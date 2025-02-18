from typing import Dict

from autogen_core.models import ModelFamily, ModelInfo

# Based on: https://docs.anthropic.com/en/docs/about-claude/models
# https://docs.anthropic.com/en/docs/resources/model-deprecations
_MODEL_POINTERS = {
    "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
    "claude-3.5-haiku": "claude-3-5-haiku-20241022",
    "claude-3-opus": "claude-3-opus-20240229",
    "claude-3-sonnet": "claude-3-sonnet-20240229",
    "claude-3-haiku": "claude-3-haiku-20240307",
}

_MODEL_INFO: Dict[str, ModelInfo] = {
    "claude-3-5-sonnet-20241022": {
        "vision": True,
        "function_calling": True,
        "json_output": False,
        "family": ModelFamily.CLAUDE_3_5_SONNET,
    },
    "claude-3-5-haiku-20241022": {
        "vision": False,
        "function_calling": True,
        "json_output": False,
        "family": ModelFamily.CLAUDE_3_5_HAIKU,
    },
    "claude-3-opus-20240229": {
        "vision": True,
        "function_calling": True,
        "json_output": False,
        "family": ModelFamily.CLAUDE_3_OPUS,
    },
    "claude-3-sonnet-20240229": {
        "vision": True,
        "function_calling": True,
        "json_output": False,
        "family": ModelFamily.CLAUDE_3_SONNET,
    },
    "claude-3-haiku-20240307": {
        "vision": True,
        "function_calling": True,
        "json_output": False,
        "family": ModelFamily.CLAUDE_3_HAIKU,
    },
}

_MODEL_TOKEN_LIMITS: Dict[str, int] = {
    "claude-3-5-sonnet-20241022": 8192,
    "claude-3-5-haiku-20241022": 8192,
    "claude-3-opus-20240229": 4096,
    "claude-3-sonnet-20240229": 4096,
    "claude-3-haiku-20240307": 4096,
}


def resolve_model(model: str) -> str:
    if model in _MODEL_POINTERS:
        return _MODEL_POINTERS[model]
    return model


def get_info(model: str) -> ModelInfo:
    resolved_model = resolve_model(model)
    return _MODEL_INFO[resolved_model]


def get_token_limit(model: str) -> int:
    resolved_model = resolve_model(model)
    return _MODEL_TOKEN_LIMITS[resolved_model]
