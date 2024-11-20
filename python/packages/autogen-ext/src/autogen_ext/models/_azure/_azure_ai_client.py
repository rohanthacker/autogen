import asyncio
import logging
import re
from asyncio import Task
from typing import Any, Mapping, Optional, Sequence, Unpack, List, AsyncGenerator, Union, Dict

from azure.ai.inference.aio import ChatCompletionsClient
from azure.ai.inference.models import (
    SystemMessage as AzureSystemMessage,
    UserMessage as AzureUserMessage,
    AssistantMessage as AzureAssistantMessage,
    ToolMessage as AzureToolMessage,
    FunctionCall as AzureFunctionCall,
    ChatCompletions,
    ChatCompletionsToolDefinition,
    FunctionDefinition,
    ChatChoice,
    ChatCompletionsResponseFormatJSON,
    ChatCompletionsResponseFormatText,
    ContentItem,
    TextContentItem,
    ImageContentItem,
    ImageUrl,
    ImageDetailLevel,
    ChatCompletionsToolCall,
)
from azure.core.credentials import AzureKeyCredential
from azure.core.credentials_async import AsyncTokenCredential

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
from .config import AzureAIChatCompletionClientConfig, AzureAICreateArgs, GITHUB_MODELS_ENDPOINT

logger = logging.getLogger(EVENT_LOGGER_NAME)
trace_logger = logging.getLogger(TRACE_LOGGER_NAME)


create_kwargs = set(AzureAICreateArgs.__annotations__.keys())


def _is_github_model(endpoint: str) -> bool:
    return endpoint == GITHUB_MODELS_ENDPOINT


def _func_call_to_azure(message: FunctionCall) -> ChatCompletionsToolCall:
    return ChatCompletionsToolCall(
        id=message.id,
        function=AzureFunctionCall(arguments=message.arguments, name=message.name),
    )


def _system_message_to_azure(message: SystemMessage) -> AzureSystemMessage:
    return AzureSystemMessage(content=message.content)


def _user_message_to_azure(message: UserMessage) -> AzureUserMessage:
    assert_valid_name(message.source)
    if isinstance(message.content, str):
        return AzureUserMessage(content=message.content)
    else:
        parts: List[ContentItem] = []
        for part in message.content:
            if isinstance(part, str):
                parts.append(TextContentItem(text=part))
            elif isinstance(part, Image):
                # TODO: support url based images
                # TODO: support specifying details
                parts.append(ImageContentItem(image_url=ImageUrl(url=part.data_uri, detail=ImageDetailLevel.AUTO)))
            else:
                raise ValueError(f"Unknown content type: {message.content}")
        return AzureUserMessage(content=parts)


def _assistant_message_to_azure(message: AssistantMessage) -> AzureAssistantMessage:
    assert_valid_name(message.source)
    if isinstance(message.content, list):
        return AzureAssistantMessage(
            tool_calls=[_func_call_to_azure(x) for x in message.content],
        )
    else:
        return AzureAssistantMessage(content=message.content)


def _tool_message_to_azure(message: FunctionExecutionResultMessage) -> Sequence[AzureToolMessage]:
    return [AzureToolMessage(content=x.content, tool_call_id=x.call_id) for x in message.content]


def to_azure_ai_message(message: LLMMessage):
    if isinstance(message, SystemMessage):
        return [_system_message_to_azure(message)]
    elif isinstance(message, UserMessage):
        return [_user_message_to_azure(message)]
    elif isinstance(message, AssistantMessage):
        return [_assistant_message_to_azure(message)]
    else:
        return _tool_message_to_azure(message)


def assert_valid_name(name: str) -> str:
    """
    Ensure that configured names are valid, raises ValueError if not.

    For munging LLM responses use _normalize_name to ensure LLM specified names don't break the API.
    """
    if not re.match(r"^[a-zA-Z0-9_-]+$", name):
        raise ValueError(f"Invalid name: {name}. Only letters, numbers, '_' and '-' are allowed.")
    if len(name) > 64:
        raise ValueError(f"Invalid name: {name}. Name must be less than 64 characters.")
    return name


def normalize_name(name: str) -> str:
    """
    LLMs sometimes ask functions while ignoring their own format requirements, this function should be used to replace invalid characters with "_".

    Prefer _assert_valid_name for validating user configuration or input
    """
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)[:64]


def _add_usage(usage1: RequestUsage, usage2: RequestUsage) -> RequestUsage:
    return RequestUsage(
        prompt_tokens=usage1.prompt_tokens + usage2.prompt_tokens,
        completion_tokens=usage1.completion_tokens + usage2.completion_tokens,
    )


class AzureAIChatCompletionClient(ChatCompletionClient):
    def __init__(self, **kwargs: Unpack[AzureAIChatCompletionClientConfig]):
        if "endpoint" not in kwargs:
            raise ValueError("endpoint is required")
        if "credential" not in kwargs:
            raise ValueError("credential is required")
        if "model_capabilities" not in kwargs:
            raise ValueError("model_capabilities is required")

        if _is_github_model(kwargs["endpoint"]) and "model" not in kwargs:
            raise ValueError("model is required for Github models")

        create_args: AzureAICreateArgs = dict(kwargs).copy()

        self._endpoint: str = create_args.pop("endpoint")
        self._credential: Union[AzureKeyCredential, AsyncTokenCredential] = create_args.pop("credential")
        self._model_capabilities: ModelCapabilities = create_args.pop("model_capabilities")

        # TODO: Decide how to support response_format
        # if (
        #     "response_format" in create_args
        #     and create_args["response_format"]["type"] == "json_object"
        #     and not self._model_capabilities["json_output"]
        # ):
        #     raise ValueError("Model does not support JSON output")

        self._create_args = create_args

        self._client: Optional[ChatCompletionsClient] = None

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
        extra_create_args_keys = set(extra_create_args.keys())
        if not create_kwargs.issuperset(extra_create_args_keys):
            raise ValueError(f"Invalid extra_create_args: {extra_create_args_keys - create_kwargs}")

        create_args = self._create_args.copy()
        create_args.update(extra_create_args)

        if self.capabilities["vision"] is False:
            for message in messages:
                if isinstance(message, UserMessage):
                    if isinstance(message.content, list) and any(isinstance(x, Image) for x in message.content):
                        raise ValueError("Model does not support vision and image was provided")

        if json_output is not None:
            if self.capabilities["json_output"] is False and json_output is True:
                raise ValueError("This model does not support json_output")
            if json_output is True:
                create_args["response_format"] = ChatCompletionsResponseFormatJSON()
            else:
                create_args["response_format"] = ChatCompletionsResponseFormatText()

        if self.capabilities["json_output"] is False and json_output is True:
            raise ValueError("Model does not support JSON output")

        if self.capabilities["function_calling"] is False and len(tools) > 0:
            raise ValueError("Model does not support function calling")

        azure_messages_nested = [to_azure_ai_message(msg) for msg in messages]
        azure_messages = [item for sublist in azure_messages_nested for item in sublist]

        future: Task[ChatCompletions]

        self._create_client()
        # TODO: Support tools given at create
        # TODO: Remove tools kwarg from create_args
        if len(tools) > 0:
            converted_tools = self._convert_tools(tools)
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

        result = await future
        await self._close_client()

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

        choice: ChatChoice = result.choices[0]

        if choice.finish_reason == "tool_calls":
            assert choice.message.tool_calls is not None

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

        # TODO: Find out why is this operation is performed,
        #       the return value of these functions are not used,
        #       so the following function calls are redundant
        _add_usage(self._actual_usage, usage)
        _add_usage(self._total_usage, usage)

        return CreateResult(
            finish_reason=finish_reason,  # type: ignore
            usage=usage,
            content=content,
            cached=False,
        )

    async def create_stream(
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[Tool | ToolSchema] = [],
        json_output: Optional[bool] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> AsyncGenerator[Union[str, CreateResult], None]:
        extra_create_args_keys = set(extra_create_args.keys())
        if not create_kwargs.issuperset(extra_create_args_keys):
            raise ValueError(f"Extra create args are invalid: {extra_create_args_keys - create_kwargs}")

        create_args = self._create_args.copy()
        create_args.update(extra_create_args)

        if self.capabilities["vision"] is False:
            for message in messages:
                if isinstance(message, UserMessage):
                    if isinstance(message.content, list) and any(isinstance(x, Image) for x in message.content):
                        raise ValueError("Model does not support vision and image was provided")

        if json_output is not None:
            if self.capabilities["json_output"] is False and json_output is True:
                raise ValueError("This model does not support json_output")
            if json_output is True:
                create_args["response_format"] = ChatCompletionsResponseFormatJSON()
            # else:
            #     create_args["response_format"] = ChatCompletionsResponseFormatText()

        if self.capabilities["json_output"] is False and json_output is True:
            raise ValueError("Model does not support JSON output")

        if self.capabilities["function_calling"] is False and len(tools) > 0:
            raise ValueError("Model does not support function calling")

        azure_messages_nested = [to_azure_ai_message(msg) for msg in messages]
        azure_messages = [item for sublist in azure_messages_nested for item in sublist]

        self._create_client()
        if len(tools) > 0:
            converted_tools = self._convert_tools(tools)
            stream_future = asyncio.ensure_future(
                self._client.complete(
                    messages=azure_messages,
                    stream=True,
                    tools=converted_tools,
                    **create_args,
                )
            )
        else:
            stream_future = asyncio.ensure_future(
                self._client.complete(messages=azure_messages, stream=True, **create_args)
            )

        if cancellation_token is not None:
            await cancellation_token.link_future(stream_future)

        stream = await stream_future

        finish_reason = None
        content_deltas: List[str] = []
        full_tool_calls: Dict[int, FunctionCall] = {}

        async for chunk in stream:
            choice = chunk.choices[0]
            if choice.finish_reason is not None:
                finish_reason = choice.finish_reason.value
            if choice.delta.content is not None:
                content_deltas.append(choice.delta.content)
                if len(choice.delta.content) > 1:
                    yield choice.delta.content
                continue
            if choice.delta.tool_calls is not None:
                for tool_call_chunk in choice.delta.tool_calls:
                    idx = tool_call_chunk["index"]
                    if idx not in full_tool_calls:
                        full_tool_calls[idx] = FunctionCall(
                            id="",
                            arguments="",
                            name="",
                        )
                    if tool_call_chunk.id is not None:
                        full_tool_calls[idx].id = tool_call_chunk.id

                    if tool_call_chunk.function is not None:
                        if tool_call_chunk.function.name is not None:
                            full_tool_calls[idx].name += tool_call_chunk.function.name
                        if tool_call_chunk.function.arguments is not None:
                            full_tool_calls[idx].arguments += tool_call_chunk.function.arguments

        await self._close_client()

        if finish_reason is None:
            raise ValueError("No stop reason found")

        content: Union[str, List[FunctionCall]]
        if len(content_deltas) > 0:
            content = "".join(content_deltas)
        else:
            content = list(full_tool_calls.values())

        usage = RequestUsage(prompt_tokens=0, completion_tokens=0)

        if finish_reason == "tool_calls":
            finish_reason = "function_calls"

        yield CreateResult(
            content=content,
            finish_reason=finish_reason,  # type: ignore
            usage=usage,
            cached=False,
            logprobs=None,
        )

    def actual_usage(self) -> RequestUsage:
        return self._actual_usage

    def total_usage(self) -> RequestUsage:
        return self._total_usage

    def count_tokens(self, messages: Sequence[LLMMessage], tools: Sequence[Tool | ToolSchema] = []) -> int:
        raise NotImplementedError()

    def remaining_tokens(self, messages: Sequence[LLMMessage], tools: Sequence[Tool | ToolSchema] = []) -> int:
        raise NotImplementedError()

    def _create_client(self) -> None:
        if self._client is None:
            self._client = ChatCompletionsClient(self._endpoint, self._credential, **self._create_args)

    async def _close_client(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    @staticmethod
    def _convert_tools(tools: Sequence[Tool | ToolSchema]) -> List[ChatCompletionsToolDefinition]:
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
