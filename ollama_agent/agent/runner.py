"""Agent execution logic for the Ollama agent."""

import logging
from typing import Any, AsyncGenerator, Optional

from agents import Runner
from ..exceptions import ModelCapabilityError
from .agent_manager import AgentManager
from .events import event_payloads
from .session_manager import SessionManager

logger = logging.getLogger(__name__)


async def run_async_streamed(
    agent_manager: AgentManager,
    session_manager: SessionManager,
    prompt: str,
    model: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
) -> AsyncGenerator[dict[str, Any], None]:
    try:
        agent = await agent_manager.get_agent(model, reasoning_effort)
    except ModelCapabilityError as exc:
        logger.error("Model capability error for streamed execution: %s", exc)
        yield {"type": "error", "content": str(exc)}
        return
    try:
        result = Runner.run_streamed(
            agent, input=prompt, session=session_manager.get_session()
        )
        async for event in result.stream_events():
            for payload in event_payloads(event):
                yield payload
    except Exception as exc:  # noqa: BLE001
        logger.error("Error running streamed agent: %s", exc)
        yield {"type": "error", "content": str(exc)}
