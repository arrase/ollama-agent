"""Core module with types, models and shared utilities."""

from .common import (
    DEFAULT_REASONING_EFFORT,
    ReasoningEffortValue,
    atomic_write_text,
    render_prompt_template,
    shorten,
    validate_identifier,
)
from .models import (
    REASONING_HIDE_VALUES,
    ModelCapabilityError,
    ModelContextWindowError,
    OllamaChatModel,
    check_ollama_version,
    create_ollama_async_client,
    create_ollama_chat_model,
    ensure_model_supports_tools,
    get_model_thinking_config,
    model_supports_tools,
    parse_context_window_input,
    validate_and_normalize_effort,
    validate_reasoning_effort,
)
from .parsers import (
    ThinkTagParser,
)
from .prompt_processor import (
    PromptProcessingError,
    process_prompt_mentions,
)
from .resource_manager import (
    BaseFileStoreManager,
    require_text,
    resolve_unique_match,
)

__all__ = [
    "DEFAULT_REASONING_EFFORT",
    "REASONING_HIDE_VALUES",
    "BaseFileStoreManager",
    "ModelCapabilityError",
    "ModelContextWindowError",
    "OllamaChatModel",
    "PromptProcessingError",
    "ReasoningEffortValue",
    "ThinkTagParser",
    "atomic_write_text",
    "check_ollama_version",
    "create_ollama_async_client",
    "create_ollama_chat_model",
    "ensure_model_supports_tools",
    "get_model_thinking_config",
    "model_supports_tools",
    "parse_context_window_input",
    "process_prompt_mentions",
    "render_prompt_template",
    "require_text",
    "resolve_unique_match",
    "shorten",
    "validate_and_normalize_effort",
    "validate_identifier",
    "validate_reasoning_effort",
]
