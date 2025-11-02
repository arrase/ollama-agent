"""Utility functions shared across the application."""

import logging
from typing import Literal, cast

# Type definitions
ReasoningEffortValue = Literal["low", "medium", "high"]
ALLOWED_REASONING_EFFORTS: tuple[ReasoningEffortValue, ...] = ("low", "medium", "high")
DEFAULT_REASONING_EFFORT: ReasoningEffortValue = "medium"

# Configure logging
logger = logging.getLogger(__name__)


def validate_reasoning_effort(effort: str) -> ReasoningEffortValue:
    """
    Validate and normalize reasoning effort value.
    
    Args:
        effort: Effort level string to validate.
        
    Returns:
        Valid reasoning effort value.
    """
    if effort in ALLOWED_REASONING_EFFORTS:
        return cast(ReasoningEffortValue, effort)
    logger.warning(f"Invalid reasoning effort '{effort}', using default '{DEFAULT_REASONING_EFFORT}'")
    return DEFAULT_REASONING_EFFORT
