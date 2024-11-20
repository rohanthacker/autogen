from typing import TypedDict, Required, Union, Optional, List, Dict, Any

from azure.core.credentials import AzureKeyCredential
from azure.core.credentials_async import AsyncTokenCredential
from azure.ai.inference.models import (
    ChatCompletionsResponseFormat,
    ChatCompletionsToolDefinition,
    ChatCompletionsToolChoicePreset,
    ChatCompletionsNamedToolChoice,
)

from autogen_core.components.models import ModelCapabilities


GITHUB_MODELS_ENDPOINT = "https://models.inference.ai.azure.com"


class AzureAICreateArgs(TypedDict, total=False):
    frequency_penalty: Optional[float]
    presence_penalty: Optional[float]
    temperature: Optional[float]
    top_p: Optional[float]
    max_tokens: Optional[int]
    response_format: Optional[ChatCompletionsResponseFormat]
    stop: Optional[List[str]]
    tools: Optional[List[ChatCompletionsToolDefinition]]
    tool_choice: Optional[Union[str, ChatCompletionsToolChoicePreset, ChatCompletionsNamedToolChoice]]
    seed: Optional[int]
    model: Optional[str]
    model_extras: Optional[Dict[str, Any]]


class AzureAIChatCompletionClientConfig(AzureAICreateArgs, total=False):
    endpoint: Required[str]
    credential: Required[Union[AzureKeyCredential | AsyncTokenCredential]]
    model_capabilities: Required[ModelCapabilities]
