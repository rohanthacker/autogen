import asyncio
import logging
import re
from asyncio import Task
from typing import Any, Mapping, Optional, Sequence, Unpack, List, AsyncGenerator, Union

from azure.ai.inference.aio import ChatCompletionsClient
from azure.ai.inference.models import (
    SystemMessage as AzureSystemMessage,
    UserMessage as AzureUserMessage,
    AssistantMessage as AzureAssistantMessage,
    ToolMessage as AzureToolMessage,
    ChatCompletions,
    ChatCompletionsToolDefinition,
    FunctionDefinition,
    ChatChoice,
)

from autogen_core.application.logging import EVENT_LOGGER_NAME, TRACE_LOGGER_NAME
from autogen_core.application.logging.events import LLMCallEvent
from autogen_core.base import CancellationToken
from autogen_core.components import Image, FunctionCall
from autogen_core.components.models import (
    ChatCompletionClient,
    CreateResult,
    LLMMessage,
    RequestUsage,
    ModelCapabilities,
    SystemMessage,
    UserMessage,
    FunctionExecutionResultMessage,
    AssistantMessage,
)
from autogen_core.components.tools import Tool, ToolSchema
from .config import AzureAIChatCompletionClientConfig

logger = logging.getLogger(EVENT_LOGGER_NAME)
trace_logger = logging.getLogger(TRACE_LOGGER_NAME)


def _tool_message_to_azure(message: FunctionExecutionResultMessage) -> Sequence[AzureToolMessage]:
    return [AzureToolMessage(content=x.content, tool_call_id=x.call_id) for x in message.content]


def to_azure_ai_message(message: LLMMessage):
    if isinstance(message, SystemMessage):
        return [AzureSystemMessage(content=message.content)]
    elif isinstance(message, UserMessage):
        return [AzureUserMessage(content=message.content)]
    elif isinstance(message, AssistantMessage):
        return [AzureAssistantMessage(content=message.content)]
    else:
        return [_tool_message_to_azure(message)]


def normalize_name(name: str) -> str:
    """
    LLMs sometimes ask functions while ignoring their own format requirements, this function should be used to replace invalid characters with "_".

    Prefer _assert_valid_name for validating user configuration or input
    """
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)[:64]


class AzureAIChatCompletionClient(ChatCompletionClient):
    def __init__(self, **kwargs: Unpack[AzureAIChatCompletionClientConfig]):
        if "endpoint" not in kwargs:
            raise ValueError("endpoint is required")
        if "credentials" not in kwargs:
            raise ValueError("credentials is required")
        if "model_capabilities" not in kwargs:
            raise ValueError("model_capabilities is required")

        copied_args = dict(kwargs).copy()

        self._model_capabilities: ModelCapabilities = copied_args.pop("model_capabilities")

        self._client = ChatCompletionsClient(
            endpoint=copied_args["endpoint"],
            credential=copied_args["credentials"],
        )

        self._total_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
        self._actual_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)

    async def create(
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[Tool | ToolSchema] = [],
        json_output: Optional[bool] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> CreateResult:
        # TODO: Valid create args
        create_args = {}

        if self.capabilities["vision"] is False:
            for message in messages:
                if isinstance(message, UserMessage):
                    if isinstance(message.content, list) and any(isinstance(x, Image) for x in message.content):
                        raise ValueError("Model does not support vision and image was provided")

        if json_output is not None:
            if self.capabilities["json_output"] is False and json_output is True:
                raise ValueError("This model does not support json_output")
            if json_output is True:
                create_args["response_format"] = {"type": "json_object"}
            else:
                create_args["response_format"] = {"type": "text"}

        if self.capabilities["json_output"] is False and json_output is True:
            raise ValueError("Model does not support JSON output")

        if self.capabilities["function_calling"] is False and len(tools) > 0:
            raise ValueError("Model does not support function calling")

        azure_messages_nested = [to_azure_ai_message(msg) for msg in messages]
        azure_messages = [item for sublist in azure_messages_nested for item in sublist]

        future: Task[ChatCompletions]

        if len(tools) > 0:
            converted_tools = self.convert_tools(tools)
            future = asyncio.ensure_future(
                self._client.complete(
                    messages=azure_messages,
                    stream=False,
                    tools=converted_tools,
                    **create_args,
                )
            )
        else:
            future = asyncio.ensure_future(
                self._client.complete(
                    messages=azure_messages,
                    stream=False,
                    **create_args,
                )
            )

        if cancellation_token is not None:
            await cancellation_token.link_future(future)

        try:
            result = await future
        except:
            raise
        finally:
            await self._client.close()

        choice: ChatChoice = result.choices[0]

        if result.usage is not None:
            logger.info(
                LLMCallEvent(
                    prompt_tokens=result.usage.prompt_tokens,
                    completion_tokens=result.usage.completion_tokens,
                )
            )

        usage = RequestUsage(
            prompt_tokens=result.usage.prompt_tokens if result.usage is not None else 0,
            completion_tokens=result.usage.completion_tokens if result.usage is not None else 0,
        )

        if choice.finish_reason == "tool_calls":
            assert choice.message.tool_calls is not None

            # NOTE: If OAI response type changes, this will need to be updated
            content = [
                FunctionCall(
                    id=x.id,
                    arguments=x.function.arguments,
                    name=normalize_name(x.function.name),
                )
                for x in choice.message.tool_calls
            ]
            finish_reason = "function_calls"
        else:
            finish_reason = choice.finish_reason.value
            content = choice.message.content or ""

        # TODO: Add support for logprobs
        # TODO: Add add_tokens

        return CreateResult(
            finish_reason=finish_reason,  # type: ignore
            usage=usage,
            content=content,
            cached=False,
        )

    def create_stream(
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[Tool | ToolSchema] = [],
        json_output: Optional[bool] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> AsyncGenerator[Union[str, CreateResult], None]:
        raise NotImplementedError()

    def actual_usage(self) -> RequestUsage:
        return self._actual_usage

    def total_usage(self) -> RequestUsage:
        return self._total_usage

    def count_tokens(self, messages: Sequence[LLMMessage], tools: Sequence[Tool | ToolSchema] = []) -> int:
        raise NotImplementedError()

    def remaining_tokens(self, messages: Sequence[LLMMessage], tools: Sequence[Tool | ToolSchema] = []) -> int:
        raise NotImplementedError()

    @staticmethod
    def convert_tools(tools: Sequence[Tool | ToolSchema]) -> List[ChatCompletionsToolDefinition]:
        result: List[ChatCompletionsToolDefinition] = []
        for tool in tools:
            if isinstance(tool, Tool):
                tool_schema = tool.schema
            else:
                assert isinstance(tool, dict)
                tool_schema = tool

            result.append(
                ChatCompletionsToolDefinition(
                    function=FunctionDefinition(
                        name=tool_schema["name"],
                        description=(tool_schema["description"] if "description" in tool_schema else ""),
                        parameters=tool_schema["parameters"] if "parameters" in tool_schema else {},
                    )
                )
            )
        return result

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._model_capabilities
