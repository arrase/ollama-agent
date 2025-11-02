"""AI agent using openai-agents and Ollama."""

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Literal, Optional, cast

from agents import Agent, ModelSettings, Runner, SQLiteSession, set_default_openai_api, set_default_openai_client, set_tracing_disabled
from openai import AsyncOpenAI
from openai.types.shared import Reasoning

from .tools import execute_command

ReasoningEffortValue = Literal["low", "medium", "high"]
ALLOWED_REASONING_EFFORTS: tuple[ReasoningEffortValue, ...] = ("low", "medium", "high")
DEFAULT_REASONING_EFFORT: ReasoningEffortValue = "medium"


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
    return DEFAULT_REASONING_EFFORT


class OllamaAgent:
    """AI agent that connects to Ollama."""
    
    model: str
    base_url: str
    api_key: str
    reasoning_effort: ReasoningEffortValue
    database_path: Path
    client: AsyncOpenAI
    agent: Agent
    session: Optional[SQLiteSession]
    session_id: Optional[str]

    def __init__(
        self, 
        model: str, 
        base_url: str = "http://localhost:11434/v1/", 
        api_key: str = "ollama", 
        reasoning_effort: str = "medium",
        database_path: Optional[Path] = None
    ):
        """
        Initialize the agent.

        Args:
            model: Name of the model to use.
            base_url: Base URL of the Ollama server.
            api_key: API key (required but ignored by Ollama).
            reasoning_effort: Reasoning effort level (low, medium, high).
            database_path: Path to the SQLite database for session storage.
        """
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.reasoning_effort = validate_reasoning_effort(reasoning_effort)
        self.database_path = database_path or Path.home() / ".ollama-agent" / "sessions.db"
        
        # Ensure database directory exists
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create OpenAI client and agent
        self.client = self._create_client()
        self.agent = self._create_agent()
        self.session = None
        self.session_id = None
        
        # Initialize with a new session
        self.reset_session()

    def _connect(self) -> sqlite3.Connection:
        """Create a SQLite connection for the session database."""
        return sqlite3.connect(str(self.database_path))

    @staticmethod
    def _extract_preview_text(message_blob: Optional[str]) -> str:
        """Parse the first message of a session into a short preview."""
        if not message_blob:
            return "No messages"

        try:
            message_data = json.loads(message_blob)
        except (json.JSONDecodeError, TypeError):
            return "No content"

        if isinstance(message_data, dict):
            content = message_data.get("content")
            if isinstance(content, str):
                return content[:50]
            if isinstance(content, list):
                text = " ".join(
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("text")
                ).strip()
                return text[:50] if text else "No content"
            return str(message_data)[:50]

        return str(message_data)[:50]

    def _create_client(self) -> AsyncOpenAI:
        """
        Create and configure OpenAI client for Ollama.
        
        Returns:
            Configured AsyncOpenAI client.
        """
        set_tracing_disabled(True)
        set_default_openai_api("chat_completions")
        
        client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
        set_default_openai_client(client, use_for_tracing=False)
        
        return client

    def _create_agent(self) -> Agent:
        """
        Create the AI agent with tools and settings.
        
        Returns:
            Configured Agent instance.
        """
        return Agent(
            name="Ollama Assistant",
            instructions=(
                "You are a helpful AI assistant that can help with various tasks. "
                "You have access to a tool that allows you to execute operating system commands."
            ),
            model=self.model,
            tools=[execute_command],
            model_settings=ModelSettings(
                reasoning=Reasoning(effort=self.reasoning_effort)
            )
        )

    async def run_async(self, prompt: str) -> str:
        """
        Run the agent asynchronously.

        Args:
            prompt: The user's prompt.

        Returns:
            The agent's response.
        """
        try:
            result = await Runner.run(self.agent, input=prompt, session=self.session)
            return str(result.final_output)
        except Exception as e:
            return f"Error: {str(e)}"
    
    def reset_session(self) -> str:
        """
        Reset conversation with a new session.
        
        Returns:
            The new session ID.
        """
        self.session_id = str(uuid.uuid4())
        self.session = SQLiteSession(self.session_id, str(self.database_path))
        return self.session_id
    
    def load_session(self, session_id: str) -> None:
        """
        Load an existing session.
        
        Args:
            session_id: ID of the session to load.
        """
        self.session_id = session_id
        self.session = SQLiteSession(session_id, str(self.database_path))
    
    def get_session_id(self) -> Optional[str]:
        """
        Get the current session ID.
        
        Returns:
            The current session ID or None if no session is active.
        """
        return self.session_id
    
    def list_sessions(self) -> list[dict[str, Any]]:
        """
        List all available sessions in the database.
        
        Returns:
            List of dictionaries containing session information.
            Each dictionary has: session_id, message_count, first_message, last_message
        """
        if not self.database_path.exists():
            return []
        
        try:
            with self._connect() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT s.session_id,
                           COUNT(m.id) as message_count,
                           s.created_at as first_message_time,
                           s.updated_at as last_message_time
                    FROM agent_sessions s
                    LEFT JOIN agent_messages m ON s.session_id = m.session_id
                    GROUP BY s.session_id
                    ORDER BY s.updated_at DESC
                    """
                )

                rows = cursor.fetchall()

                sessions: list[dict[str, Any]] = []
                for session_id, count, first_time, last_time in rows:
                    cursor.execute(
                        """
                        SELECT message_data FROM agent_messages
                        WHERE session_id = ?
                        ORDER BY created_at ASC
                        LIMIT 1
                        """,
                        (session_id,),
                    )

                    preview_row = cursor.fetchone()
                    preview = self._extract_preview_text(preview_row[0] if preview_row else None)

                    sessions.append({
                        "session_id": session_id,
                        "message_count": count,
                        "first_message": first_time or "Unknown",
                        "last_message": last_time or "Unknown",
                        "preview": preview,
                    })

                return sessions
        except Exception as e:
            print(f"Error listing sessions: {e}")
            return []
    
    async def get_session_history(self, session_id: Optional[str] = None) -> list[Any]:
        """
        Get the conversation history for a session.
        
        Args:
            session_id: ID of the session. If None, uses current session.
            
        Returns:
            List of messages in the session.
        """
        if session_id is None:
            session_id = self.session_id
        
        if session_id is None:
            return []
        
        temp_session = SQLiteSession(session_id, str(self.database_path))
        try:
            items = await temp_session.get_items()
            return list(items)
        except Exception as e:
            print(f"Error getting session history: {e}")
            return []
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session from the database.
        
        Args:
            session_id: ID of the session to delete.
            
        Returns:
            True if successful, False otherwise.
        """
        if not self.database_path.exists():
            return False
        
        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM agent_messages WHERE session_id = ?", (session_id,))
                cursor.execute("DELETE FROM agent_sessions WHERE session_id = ?", (session_id,))

            # If we deleted the current session, create a new one
            if session_id == self.session_id:
                self.reset_session()
            
            return True
        except Exception as e:
            print(f"Error deleting session: {e}")
            return False
