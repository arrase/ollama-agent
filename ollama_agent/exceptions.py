"""Custom exceptions for the Ollama agent."""


class ModelCapabilityError(RuntimeError):
    """Raised when the selected model cannot run tool calls."""
