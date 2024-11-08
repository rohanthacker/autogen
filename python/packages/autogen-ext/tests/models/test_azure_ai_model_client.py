import asyncio
from datetime import datetime

import pytest
from azure.ai.inference.aio import ChatCompletionsClient
from azure.ai.inference.models import ChatCompletions, ChatChoice, CompletionsUsage, ChatResponseMessage
from azure.core.credentials import AzureKeyCredential

from autogen_core.base import CancellationToken
from autogen_core.components.models import ModelCapabilities, UserMessage
from autogen_ext.models import AzureAIChatCompletionClient

endpoint = "endpoint"
api_key = "api_key"

async def _mock_create(*args, **kwargs):
    # TODO: Add Mock for Streaming Client
    model = kwargs.get("model", "")
    return ChatCompletions(
        id="id",
        created=datetime.now(),
        model=model,
        usage=CompletionsUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        choices=[
            ChatChoice(finish_reason="stop", index=0, message=ChatResponseMessage(content="Hello", role="assistant"))
        ],
    )


@pytest.mark.asyncio
async def test_azure_ai_chat_completion_client() -> None:
    client = AzureAIChatCompletionClient(
        endpoint="endpoint",
        credentials=AzureKeyCredential("api_key"),
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
        endpoint=endpoint,
        credentials=AzureKeyCredential(api_key),
        model_capabilities=ModelCapabilities(
            json_output=False,
            function_calling=False,
            vision=False,
        ),
    )
    result = await client.create(messages=[UserMessage(content="Hello", source="user")])
    assert result.content == "Hello"


@pytest.mark.asyncio
async def test_openai_chat_completion_client_create_cancel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ChatCompletionsClient, "complete", _mock_create)
    client = AzureAIChatCompletionClient(
        endpoint=endpoint,
        credentials=AzureKeyCredential(api_key),
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
