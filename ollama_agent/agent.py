"""AI agent using openai-agents and Ollama."""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Iterable, Optional, cast

from agents import (
    Agent,
    ModelSettings,
    Runner,
    SQLiteSession,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)
from openai import AsyncOpenAI
from openai.types.responses import ResponseReasoningTextDeltaEvent, ResponseTextDeltaEvent
from openai.types.shared import Reasoning

from .settings.configini import load_instructions
from .settings.mcp import RunningMCPServer, cleanup_mcp_servers, initialize_mcp_servers
from .tools import execute_command
from .utils import ReasoningEffortValue, extract_text, validate_reasoning_effort

logger = logging.getLogger(__name__)


def _raw_event_payloads(data: Any) -> Iterable[dict[str, Any]]:
    if isinstance(data, ResponseReasoningTextDeltaEvent) and data.delta:
        yield {"type": "reasoning_delta", "content": data.delta}
    elif isinstance(data, ResponseTextDeltaEvent) and data.delta:
        yield {"type": "text_delta", "content": data.delta}


def _item_event_payloads(item: Any) -> Iterable[dict[str, Any]]:
    item_type = getattr(item, "type", "")
    if item_type == "tool_call_item":
        yield {"type": "tool_call", "name": getattr(item, "name", "unknown")}
    elif item_type == "tool_call_output_item":
        yield {"type": "tool_output", "output": str(getattr(item, "output", ""))}
    elif item_type == "reasoning":
        summary = getattr(item, "summary", "")
        if summary:
            yield {"type": "reasoning_summary", "content": summary}


@dataclass(slots=True)
class OllamaAgent:
    model: str
    base_url: str = "http://localhost:11434/v1/"
    api_key: str = "ollama"
    reasoning_effort: ReasoningEffortValue = "medium"
    database_path: Optional[Path] = None
    mcp_config_path: Optional[Path] = None
    instructions: str = field(init=False)
    client: AsyncOpenAI = field(init=False)
    agent: Agent = field(init=False)
    session: Optional[SQLiteSession] = field(init=False, default=None)
    session_id: Optional[str] = field(init=False, default=None)
    mcp_servers: list[RunningMCPServer] = field(init=False, default_factory=list)
    storage_path: Path = field(init=False)
    _db_path: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.reasoning_effort = validate_reasoning_effort(self.reasoning_effort)
        self.storage_path = self.database_path or Path.home() / ".ollama-agent" / "sessions.db"
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.database_path = self.storage_path
        self._db_path = str(self.storage_path)
        self.instructions = load_instructions()

        set_tracing_disabled(True)
        set_default_openai_api("chat_completions")
        self.client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
        set_default_openai_client(self.client, use_for_tracing=False)

        self.agent = self._create_agent()
        self.reset_session()

    @staticmethod
    def _extract_preview_text(message_blob: Optional[str]) -> str:
        if not message_blob:
            return "No messages"
        try:
            message_data = json.loads(message_blob)
        except (json.JSONDecodeError, TypeError):
            return "No content"

        content: Any = message_data.get("content") if isinstance(message_data, dict) else message_data
        text_preview = extract_text(content)
        return text_preview[:50] if text_preview else str(message_data)[:50]

    def _create_agent(
        self,
        *,
        model: Optional[str] = None,
        reasoning_effort: Optional[ReasoningEffortValue] = None,
    ) -> Agent:
        return Agent(
            name="Ollama Assistant",
            instructions=self.instructions,
            model=model or self.model,
            tools=[execute_command],
            mcp_servers=[entry.server for entry in self.mcp_servers],
            model_settings=ModelSettings(
                reasoning=Reasoning(
                    effort=cast(
                        Any, reasoning_effort or self.reasoning_effort
                    )
                )
            ),
        )

    async def _ensure_mcp_servers_initialized(self) -> None:
        if not self.mcp_servers and self.mcp_config_path:
            self.mcp_servers = await initialize_mcp_servers(self.mcp_config_path)
            if self.mcp_servers:
                self.agent = self._create_agent()

    async def _get_agent(self, model: Optional[str], reasoning_effort: Optional[str]) -> Agent:
        await self._ensure_mcp_servers_initialized()
        if not model and not reasoning_effort:
            return self.agent
        effort: ReasoningEffortValue = (
            validate_reasoning_effort(reasoning_effort)
            if reasoning_effort
            else self.reasoning_effort
        )
        return self._create_agent(model=model, reasoning_effort=effort)

    async def cleanup(self) -> None:
        if self.mcp_servers:
            await cleanup_mcp_servers(self.mcp_servers)
            self.mcp_servers.clear()

    async def run_async(
        self,
        prompt: str,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        agent = await self._get_agent(model, reasoning_effort)
        try:
            result = await Runner.run(agent, input=prompt, session=self.session)
            return str(result.final_output)
        except Exception as exc:  # noqa: BLE001
            logger.error("Error running agent: %s", exc)
            return f"Error: {exc}"

    async def run_async_streamed(
        self,
        prompt: str,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        agent = await self._get_agent(model, reasoning_effort)
        try:
            result = Runner.run_streamed(agent, input=prompt, session=self.session)
            async for event in result.stream_events():
                match event.type:
                    case "raw_response_event":
                        for payload in _raw_event_payloads(event.data):
                            yield payload
                    case "run_item_stream_event":
                        for payload in _item_event_payloads(event.item):
                            yield payload
                    case "agent_updated_stream_event":
                        yield {"type": "agent_update", "name": event.new_agent.name}
        except Exception as exc:  # noqa: BLE001
            logger.error("Error running streamed agent: %s", exc)
            yield {"type": "error", "content": str(exc)}

    def reset_session(self) -> str:
        self.session_id = str(uuid.uuid4())
        self.session = SQLiteSession(self.session_id, self._db_path)
        return self.session_id

    def load_session(self, session_id: str) -> None:
        self.session_id = session_id
        self.session = SQLiteSession(session_id, self._db_path)

    def get_session_id(self) -> Optional[str]:
        return self.session_id

    def list_sessions(self) -> list[dict[str, Any]]:
        if not self.storage_path.exists():
            return []
        try:
            with sqlite3.connect(self._db_path) as conn:
                rows = conn.execute(
                    """
                    SELECT s.session_id,
                           COUNT(m.id) AS message_count,
                           s.created_at,
                           s.updated_at,
                           (
                               SELECT message_data
                               FROM agent_messages
                               WHERE session_id = s.session_id
                               ORDER BY created_at ASC
                               LIMIT 1
                           ) AS first_message
                    FROM agent_sessions s
                    LEFT JOIN agent_messages m ON s.session_id = m.session_id
                    GROUP BY s.session_id
                    ORDER BY s.updated_at DESC
                    """
                ).fetchall()

            return [
                {
                    "session_id": session_id,
                    "message_count": message_count,
                    "first_message": first_time or "Unknown",
                    "last_message": last_time or "Unknown",
                    "preview": self._extract_preview_text(first_message),
                }
                for session_id, message_count, first_time, last_time, first_message in rows
            ]
        except Exception as exc:  # noqa: BLE001
            logger.error("Error listing sessions: %s", exc)
            return []

    async def get_session_history(self, session_id: Optional[str] = None) -> list[Any]:
        session_id = session_id or self.session_id
        if not session_id:
            return []
        temp_session = SQLiteSession(session_id, self._db_path)
        try:
            return list(await temp_session.get_items())
        except Exception as exc:  # noqa: BLE001
            logger.error("Error getting session history: %s", exc)
            return []

    def delete_session(self, session_id: str) -> bool:
        if not self.storage_path.exists():
            return False
        try:
            with sqlite3.connect(self._db_path) as conn:
                for table in ("agent_messages", "agent_sessions"):
                    conn.execute(f"DELETE FROM {table} WHERE session_id = ?", (session_id,))
            if session_id == self.session_id:
                self.reset_session()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Error deleting session: %s", exc)
            return False
