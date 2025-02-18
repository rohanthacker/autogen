from typing import Mapping, Optional, TypedDict, Union

import httpx
from anthropic import Timeout
from autogen_core.models import ModelInfo


class AnthropicChatCompletionClientConfig(TypedDict, total=False):
    model: str
    max_tokens: int
    model_info: ModelInfo
    api_key: Optional[str]
    auth_token: Optional[str]
    base_url: Optional[Union[str | httpx.URL]]
    timeout: Optional[Union[float, Timeout]]
    max_retries: Optional[int]
    default_headers: Optional[Mapping[str, str]]
    default_query: Optional[Mapping[str, object]]
    http_client: Optional[httpx.AsyncClient]
