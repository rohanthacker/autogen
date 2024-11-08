from typing import TypedDict, Required, Union

from azure.core.credentials import AzureKeyCredential
from azure.core.credentials_async import AsyncTokenCredential

from autogen_core.components.models import ModelCapabilities


class AzureAIChatCompletionClientConfig(TypedDict, total=False):
    endpoint: Required[str]
    credentials: Required[Union[AzureKeyCredential | AsyncTokenCredential]]
    model_capabilities: Required[ModelCapabilities]
