"""Core module with types, models and shared utilities."""

from .common import (
    DEFAULT_REASONING_EFFORT,
    ReasoningEffortValue,
    atomic_write_text,
    extract_text,
    render_prompt_template,
    shorten,
    validate_identifier,
)
from .models import (
    REASONING_OFF_VALUES,
    REASONING_ON_VALUES,
    ModelCapabilityError,
    ModelContextWindowError,
    OllamaChatModel,
    OllamaVersionError,
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
    # Types
    "BaseFileStoreManager",
    "DEFAULT_REASONING_EFFORT",
    "ReasoningEffortValue",
    # Models
    "REASONING_OFF_VALUES",
    "REASONING_ON_VALUES",
    "ModelCapabilityError",
    "ModelContextWindowError",
    "OllamaChatModel",
    "OllamaVersionError",
    "check_ollama_version",
    "create_ollama_async_client",
    "create_ollama_chat_model",
    "ensure_model_supports_tools",
    "get_model_thinking_config",
    "model_supports_tools",
    "parse_context_window_input",
    "validate_and_normalize_effort",
    "validate_reasoning_effort",
    # Utils
    "atomic_write_text",
    "extract_text",
    "render_prompt_template",
    "require_text",
    "resolve_unique_match",
    "shorten",
    "validate_identifier",
    # Prompt Processor
    "PromptProcessingError",
    "process_prompt_mentions",
]
