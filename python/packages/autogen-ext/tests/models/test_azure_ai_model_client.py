import asyncio
from datetime import datetime

import pytest
from azure.ai.inference.aio import ChatCompletionsClient
from azure.ai.inference.models import (
    ChatCompletions,
    ChatChoice,
    CompletionsUsage,
    ChatResponseMessage,
    StreamingChatCompletionsUpdate,
    StreamingChatChoiceUpdate,
    StreamingChatResponseMessageUpdate,
)
from azure.core.credentials import AzureKeyCredential

from autogen_core.base import CancellationToken
from autogen_core.components.models import ModelCapabilities, UserMessage, CreateResult
from autogen_ext.models import AzureAIChatCompletionClient


async def _mock_create_stream():
    chunks = ["Hello", " Another Hello", " Yet Another Hello"]
    model = ""
    for chunk in chunks:
        await asyncio.sleep(0.1)
        yield StreamingChatCompletionsUpdate(
            id="id",
            created=datetime.now(),
            choices=[
                StreamingChatChoiceUpdate(
                    finish_reason="stop",
                    index=0,
                    delta=StreamingChatResponseMessageUpdate(
                        content=chunk,
                        role="assistant",
                    ),
                )
            ],
            usage=CompletionsUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            model=model,
        )


async def _mock_create(*args, **kwargs):
    stream = kwargs.get("stream", False)
    model = kwargs.get("model", "")
    if not stream:
        return ChatCompletions(
            id="id",
            created=datetime.now(),
            model=model,
            usage=CompletionsUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            choices=[
                ChatChoice(
                    finish_reason="stop", index=0, message=ChatResponseMessage(content="Hello", role="assistant")
                )
            ],
        )
    else:
        return _mock_create_stream()


@pytest.mark.asyncio
async def test_azure_ai_chat_completion_client() -> None:
    client = AzureAIChatCompletionClient(
        endpoint="endpoint",
        credential=AzureKeyCredential("api_key"),
        model_capabilities=ModelCapabilities(
            json_output=False,
            function_calling=False,
            vision=False,
        ),
    )
    assert client


@pytest.mark.asyncio
async def test_azure_ai_chat_completion_client_create(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ChatCompletionsClient, "complete", _mock_create)
    client = AzureAIChatCompletionClient(
        endpoint="endpoint",
        credential=AzureKeyCredential("api_key"),
        model_capabilities=ModelCapabilities(
            json_output=False,
            function_calling=False,
            vision=False,
        ),
    )
    result = await client.create(messages=[UserMessage(content="Hello", source="user")])
    assert result.content == "Hello"


@pytest.mark.asyncio
async def test_azure_ai_chat_completion_client_create_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ChatCompletionsClient, "complete", _mock_create)
    client = AzureAIChatCompletionClient(
        endpoint="endpoint",
        credential=AzureKeyCredential("api_key"),
        model_capabilities=ModelCapabilities(
            json_output=False,
            function_calling=False,
            vision=False,
        ),
    )
    chunks = []
    async for chunk in client.create_stream(messages=[UserMessage(content="Hello", source="user")]):
        chunks.append(chunk)

    assert chunks[0] == "Hello"
    assert chunks[1] == " Another Hello"
    assert chunks[2] == " Yet Another Hello"
    assert isinstance(chunks[-1], CreateResult)
    assert chunks[-1].content == "Hello Another Hello Yet Another Hello"


@pytest.mark.asyncio
async def test_openai_chat_completion_client_create_cancel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ChatCompletionsClient, "complete", _mock_create)
    client = AzureAIChatCompletionClient(
        endpoint="endpoint",
        credential=AzureKeyCredential("api_key"),
        model_capabilities=ModelCapabilities(
            json_output=False,
            function_calling=False,
            vision=False,
        ),
    )
    cancellation_token = CancellationToken()
    task = asyncio.create_task(
        client.create(messages=[UserMessage(content="Hello", source="user")], cancellation_token=cancellation_token)
    )
    cancellation_token.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
