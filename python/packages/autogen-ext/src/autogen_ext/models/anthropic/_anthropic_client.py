import asyncio
import inspect
import json
import math
from asyncio import Task
from typing import Any, AsyncGenerator, Dict, List, Literal, Mapping, Optional, Sequence, Set, Union, Unpack

from anthropic import AsyncAnthropic
from anthropic.resources import AsyncMessages
from anthropic.types import (
    ImageBlockParam,
    Message,
    MessageParam,
    MessageTokensCount,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStreamEvent,
    TextBlockParam,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlock,
)
from anthropic.types.message_create_params import MessageCreateParamsBase
from autogen_core import CancellationToken, FunctionCall, Image
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    CreateResult,
    FinishReasons,
    FunctionExecutionResultMessage,
    LLMMessage,
    ModelInfo,
    RequestUsage,
    SystemMessage,
    UserMessage,
)
from autogen_core.tools import Tool, ToolSchema

from . import _model_info
from .config import AnthropicChatCompletionClientConfig

anthropic_init_kwargs = set(inspect.getfullargspec(AsyncAnthropic.__init__).kwonlyargs)

create_kwargs = set(MessageCreateParamsBase.__annotations__.keys())
disallowed_create_args = {"stream", "messages"}
required_create_args: Set[str] = {"model"}


def get_media_type(data_uri: str) -> str:
    if data_uri.startswith("data:image/png"):
        return "image/png"
    elif data_uri.startswith("data:image/jpeg"):
        return "image/jpeg"
    elif data_uri.startswith("data:image/gif"):
        return "image/gif"
    elif data_uri.startswith("data:image/webp"):
        return "image/webp"
    else:
        raise ValueError("Unknown media type")


def _anthropic_client_from_config(config: Mapping[str, Any]) -> AsyncAnthropic:
    # Shave down the config to just the Anthropic kwargs
    anthropic_config = {k: v for k, v in config.items() if k in anthropic_init_kwargs}
    return AsyncAnthropic(**anthropic_config)


def _create_args_from_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    create_args = {k: v for k, v in config.items() if k in create_kwargs}
    create_args_keys = set(create_args.keys())
    if not required_create_args.issubset(create_args_keys):
        raise ValueError(f"Required create args are missing: {required_create_args - create_args_keys}")
    if disallowed_create_args.intersection(create_args_keys):
        raise ValueError(f"Disallowed create args are present: {disallowed_create_args.intersection(create_args_keys)}")
    return create_args


def convert_tools(tools: Sequence[Tool | ToolSchema]) -> List[ToolParam]:
    result: List[ToolParam] = []
    for tool in tools:
        if isinstance(tool, Tool):
            tool_schema = tool.schema.copy()
        else:
            assert isinstance(tool, dict)
            tool_schema = tool.copy()

        result.append(
            ToolParam(
                name=tool_schema["name"],
                description=tool_schema["description"],
                input_schema=tool_schema["parameters"],
            )
        )
    return result


def _system_message_to_anthropic_message(message: SystemMessage) -> MessageParam:
    return MessageParam(
        content=message.content,
        role="assistant",
    )


def _user_message_to_anthropic_message(message: UserMessage) -> MessageParam:
    if isinstance(message.content, str):
        return MessageParam(
            content=message.content,
            role="user",
        )
    else:
        content: List = []
        for part in message.content:
            if isinstance(part, str):
                content.append(TextBlockParam(text=part, type="text"))
            elif isinstance(part, Image):
                media_type = get_media_type(part.data_uri)
                image_part = ImageBlockParam(
                    source={
                        "data": part.to_base64(),
                        "media_type": media_type,
                        "type": "base64",
                    },
                    type="image",
                )
                content.append(image_part)
            else:
                raise ValueError(f"Unknown content type: {message.content}")
        return MessageParam(content=content, role="user")


def tool_message_to_anthropic(message: FunctionExecutionResultMessage) -> Sequence[ToolResultBlockParam]:
    return [
        ToolResultBlockParam(
            tool_use_id=x.call_id,
            type="tool_result",
            content=x.content,
        )
        for x in message.content
    ]


#     return [_system_message_to_anthropic_message(message)]
# elif isinstance(message, UserMessage):


def _to_anthropic_message(message: LLMMessage) -> MessageParam:
    if isinstance(message, (SystemMessage, UserMessage)):
        return _user_message_to_anthropic_message(message)
    elif isinstance(message, AssistantMessage):
        if isinstance(message.content, list):
            return MessageParam(
                content=[
                    ToolUseBlock(id=x.id, input=json.dumps(x.arguments), name=x.name, type="tool_use")
                    for x in message.content
                ],
                role="assistant",
            )
        else:
            return MessageParam(content=message.content, role="assistant")
    else:
        return MessageParam(content=tool_message_to_anthropic(message), role="assistant")


def normalize_finish_reason(stop_reason: str | None) -> FinishReasons:
    if stop_reason is None:
        return "unknown"

    stop_reason = stop_reason.lower()

    KNOWN_STOP_REASONS: Dict[str, FinishReasons] = {
        "end_turn": "stop",
        "max_tokens": "length",
        "stop_sequence": "stop",
        "tool_use": "function_calls",
    }

    return KNOWN_STOP_REASONS.get(stop_reason, "unknown")


class AnthropicChatCompletionClient(ChatCompletionClient):
    def __init__(self, **kwargs: Unpack[AnthropicChatCompletionClientConfig]):
        if "model" not in kwargs:
            raise ValueError("model is required for AnthropicChatCompletionClient")

        copied_args = dict(kwargs).copy()

        self._model = _model_info.resolve_model(kwargs["model"])

        self._model_info: Optional[ModelInfo] = None
        if "model_info" in kwargs:
            self._model_info = kwargs["model_info"]
            del copied_args["model_info"]
        else:
            try:
                self._model_info = _model_info.get_info(kwargs["model"])
            except KeyError as err:
                raise ValueError("model_info is required when model name is not a valid Anthropic model") from err

        self._client = _anthropic_client_from_config(copied_args)
        self._create_args = _create_args_from_config(copied_args)

        self._total_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
        self._actual_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)

    def _validate_model_info(
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[Tool | ToolSchema],
        json_output: Optional[bool],
    ) -> None:
        if self.model_info["vision"] is False:
            for message in messages:
                if isinstance(message, UserMessage):
                    if isinstance(message.content, list) and any(isinstance(x, Image) for x in message.content):
                        raise ValueError("Model does not support vision and image was provided")

        if json_output is not None:
            if self.model_info["json_output"] is False:
                raise ValueError("Model does not support JSON output.")

        if self.model_info["function_calling"] is False and len(tools) > 0:
            raise ValueError("Model does not support function calling")

    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
        json_output: Optional[bool] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> CreateResult:
        extra_create_args_keys = set(extra_create_args.keys())
        if not create_kwargs.issuperset(extra_create_args_keys):
            raise ValueError(f"Extra create args contains invalid keys: {extra_create_args_keys - create_kwargs}")

        # Copy the create args and overwrite anything in extra_create_args
        create_args = self._create_args.copy()
        create_args.update(extra_create_args)

        self._validate_model_info(messages, tools, json_output)

        anthropic_messages = [_to_anthropic_message(msg) for msg in messages]
        # anthropic_messages = [item for sublist in anthropic_messages_nested for item in sublist]

        task: Task[Message]

        if len(tools) > 0:
            converted_tools = convert_tools(tools)
            task = asyncio.create_task(
                self._client.messages.create(
                    messages=anthropic_messages,
                    tools=converted_tools,
                    # model=self._model,
                    **create_args,
                )
            )
        else:
            task = asyncio.create_task(
                self._client.messages.create(
                    messages=anthropic_messages,
                    # model=self._model,
                    **create_args,
                )
            )

        if cancellation_token is not None:
            cancellation_token.link_future(task)

        response = await task

        usage = RequestUsage(
            prompt_tokens=response.usage.input_tokens if response.usage.input_tokens is not None else 0,
            completion_tokens=response.usage.output_tokens if response.usage.output_tokens is not None else 0,
        )

        choice = response.content[0]

        if response.stop_reason == "tool_use":
            # TODO: We've dropped a text block. Lets see how we can maintain this.
            tool_use_blocks = filter(lambda msg: isinstance(msg, ToolUseBlock), response.content)
            content: Union[str, List[FunctionCall]] = [
                FunctionCall(id=x.id, name=x.name, arguments=json.dumps(x.input)) for x in tool_use_blocks
            ]
            finish_reason = "tool_use"
        else:
            content = choice.text
            finish_reason = response.stop_reason

        result = CreateResult(
            finish_reason=normalize_finish_reason(finish_reason),
            content=content,
            usage=usage,
            cached=False,
        )
        return result

    async def create_stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
        json_output: Optional[bool] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> AsyncGenerator[Union[str, CreateResult], None]:
        extra_create_args_keys = set(extra_create_args.keys())
        if not create_kwargs.issuperset(extra_create_args_keys):
            raise ValueError(f"Extra create args contains invalid keys: {extra_create_args_keys - create_kwargs}")

        # Copy the create args and overwrite anything in extra_create_args
        create_args = self._create_args.copy()
        create_args.update(extra_create_args)

        self._validate_model_info(messages, tools, json_output)

        anthropic_messages_nested = [_to_anthropic_message(msg) for msg in messages]
        anthropic_messages = [item for sublist in anthropic_messages_nested for item in sublist]

        if len(tools) > 0:
            converted_tools = convert_tools(tools)
            task = asyncio.create_task(
                self._client.messages.create(
                    max_tokens=1024,
                    messages=anthropic_messages,
                    tools=converted_tools,
                    model=self._model,
                    stream=True,
                    # **create_args
                )
            )
        else:
            task = asyncio.create_task(
                self._client.messages.create(
                    max_tokens=1024,
                    messages=anthropic_messages,
                    model=self._model,
                    stream=True,
                    # **create_args
                )
            )

        if cancellation_token is not None:
            cancellation_token.link_future(task)

        finish_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]] = None
        content_deltas: List[str] = []
        current_block_id: Optional[str] = None
        full_tool_calls: Dict[str, FunctionCall] = {}
        prompt_tokens = 0
        completion_tokens = 0

        # TODO check if we cover all event types
        #     RawMessageStartEvent,
        #     RawMessageDeltaEvent,
        #     RawMessageStopEvent,
        #     RawContentBlockStartEvent,
        #     RawContentBlockDeltaEvent,
        #     RawContentBlockStopEvent,
        async for event in await task:
            match event:
                case RawMessageStartEvent():
                    prompt_tokens = event.message.usage.input_tokens
                    completion_tokens = event.message.usage.output_tokens

                case RawContentBlockStartEvent(content_block=cb) if cb.type == "text":
                    pass

                case RawContentBlockStartEvent(content_block=cb) if cb.type == "tool_use":
                    current_block_id = cb.id
                    full_tool_calls[current_block_id] = FunctionCall(
                        id=cb.id,
                        name=cb.name,
                        arguments="",
                    )

                case RawContentBlockDeltaEvent(delta=delta) if delta.type == "text_delta":
                    content_deltas.append(delta.text)
                    yield delta.text

                case RawContentBlockDeltaEvent(delta=delta) if delta.type == "input_json_delta":
                    if current_block_id and current_block_id in full_tool_calls:
                        full_tool_calls[current_block_id].arguments += delta.partial_json
                    yield delta.partial_json

                case RawMessageDeltaEvent(delta=delta):
                    finish_reason = delta.stop_reason
                    completion_tokens += event.usage.output_tokens
            # if isinstance(event, RawMessageStartEvent):
            #     prompt_tokens = event.message.usage.input_tokens
            #     completion_tokens = event.message.usage.output_tokens
            # elif isinstance(event, RawContentBlockStartEvent):
            #     if event.content_block.type == "text":
            #         pass
            #     elif event.content_block.type == "tool_use":
            #         current_block_id = event.content_block.id
            #         full_tool_calls[current_block_id] = FunctionCall(
            #             id=event.content_block.id,
            #             name=event.content_block.name,
            #             arguments="",
            #         )
            # elif isinstance(event, RawContentBlockDeltaEvent):
            #     if event.delta.type == "text_delta":
            #         content_deltas.append(event.delta.text)
            #         yield event.delta.text
            #     elif event.delta.type == "input_json_delta":
            #         if current_block_id and current_block_id in full_tool_calls:
            #             full_tool_calls[current_block_id].arguments += event.delta.partial_json
            #         yield event.delta.partial_json
            # elif isinstance(event, RawMessageDeltaEvent):
            #     finish_reason = event.delta.stop_reason
            #     completion_tokens += event.usage.output_tokens

        content: Union[str, List[FunctionCall]]

        if len(full_tool_calls.values()) > 0:
            content = list(full_tool_calls.values())
        else:
            content = "".join(content_deltas)

        result = CreateResult(
            finish_reason=normalize_finish_reason(finish_reason),
            content=content,
            usage=RequestUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
            cached=False,
        )

        yield result

    def actual_usage(self) -> RequestUsage:
        return self._actual_usage

    def total_usage(self) -> RequestUsage:
        return self._total_usage

    def count_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        return 0

    def remaining_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        return 0

    @property
    def capabilities(self) -> ModelInfo:
        return self.model_info

    @property
    def model_info(self) -> ModelInfo:
        return self._model_info
