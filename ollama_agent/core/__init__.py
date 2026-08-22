"""Core module with types, models and shared utilities."""

from .common import (
    ALLOWED_REASONING_EFFORTS,
    DEFAULT_REASONING_EFFORT,
    CommandResult,
    RAGToolResult,
    ReasoningEffortValue,
    assistant_text_from_messages,
    extract_text,
    final_text_from_state,
    validate_identifier,
)
from .models import (
    ModelCapabilityError,
    ModelContextWindowError,
    OLLAMA_PARAM_DEFAULTS,
    OllamaChatModel,
    create_ollama_chat_model,
    ensure_model_supports_tools,
    get_model_capabilities,
    model_supports_tools,
    model_supports_thinking,
    resolve_context_window,
    resolve_model_parameters,
    resolve_ollama_reasoning,
    validate_reasoning_effort,
)
from .prompt_processor import (
    PromptProcessingError,
    process_prompt_mentions,
)
from .resource_manager import BaseFileStoreManager

__all__ = [
    # Types
    "ALLOWED_REASONING_EFFORTS",
    "BaseFileStoreManager",
    "CommandResult",
    "DEFAULT_REASONING_EFFORT",
    "RAGToolResult",
    "ReasoningEffortValue",
    # Models
    "ModelCapabilityError",
    "ModelContextWindowError",
    "OLLAMA_PARAM_DEFAULTS",
    "OllamaChatModel",
    "create_ollama_chat_model",
    "ensure_model_supports_tools",
    "get_model_capabilities",
    "model_supports_tools",
    "model_supports_thinking",
    "resolve_context_window",
    "resolve_model_parameters",
    "resolve_ollama_reasoning",
    "validate_reasoning_effort",
    # Utils
    "assistant_text_from_messages",
    "extract_text",
    "final_text_from_state",
    "validate_identifier",
    # Prompt Processor
    "PromptProcessingError",
    "process_prompt_mentions",
]
