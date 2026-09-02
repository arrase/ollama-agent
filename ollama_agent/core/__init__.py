"""Core module with types, models and shared utilities."""

from .common import (
    ALLOWED_REASONING_EFFORTS,
    DEFAULT_REASONING_EFFORT,
    RAGToolResult,
    ReasoningEffortValue,
    atomic_write_text,
    extract_text,
    validate_identifier,
)
from .models import (
    ModelCapabilityError,
    ModelContextWindowError,
    OllamaChatModel,
    create_ollama_chat_model,
    ensure_model_supports_tools,
    get_model_capabilities,
    get_model_creation_kwargs,
    model_supports_thinking,
    model_supports_tools,
    resolve_context_window,
    resolve_model_parameters,
    resolve_ollama_reasoning,
    validate_reasoning_effort,
)
from .prompt_processor import (
    ContextLimitExceededError,
    FileTooLargeError,
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
    "ALLOWED_REASONING_EFFORTS",
    "BaseFileStoreManager",
    "DEFAULT_REASONING_EFFORT",
    "RAGToolResult",
    "ReasoningEffortValue",
    # Models
    "ModelCapabilityError",
    "ModelContextWindowError",
    "OllamaChatModel",
    "create_ollama_chat_model",
    "ensure_model_supports_tools",
    "get_model_capabilities",
    "get_model_creation_kwargs",
    "model_supports_tools",
    "model_supports_thinking",
    "resolve_context_window",
    "resolve_model_parameters",
    "resolve_ollama_reasoning",
    "validate_reasoning_effort",
    # Utils
    "atomic_write_text",
    "extract_text",
    "require_text",
    "resolve_unique_match",
    "validate_identifier",
    # Prompt Processor
    "ContextLimitExceededError",
    "FileTooLargeError",
    "PromptProcessingError",
    "process_prompt_mentions",
]
