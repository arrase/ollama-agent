"""AI agent using openai-agents and Ollama."""

import json
import logging
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Optional

from agents import Agent, ModelSettings, Runner, SQLiteSession, set_default_openai_api, set_default_openai_client, set_tracing_disabled
from openai import AsyncOpenAI
from openai.types.shared import Reasoning

from .mcp_config import initialize_mcp_servers, cleanup_mcp_servers, MCPServer
from .tools import execute_command
from .utils import validate_reasoning_effort, ReasoningEffortValue

logger = logging.getLogger(__name__)


class OllamaAgent:
    """AI agent that connects to Ollama."""
    
    model: str
    base_url: str
    api_key: str
    reasoning_effort: ReasoningEffortValue
    database_path: Path
    mcp_config_path: Optional[Path]
    client: AsyncOpenAI
    agent: Agent
    session: Optional[SQLiteSession]
    session_id: Optional[str]
    mcp_servers: list[MCPServer]

    def __init__(
        self, 
        model: str, 
        base_url: str = "http://localhost:11434/v1/", 
        api_key: str = "ollama", 
        reasoning_effort: str = "medium",
        database_path: Optional[Path] = None,
        mcp_config_path: Optional[Path] = None
    ):
        """
        Initialize the agent.

        Args:
            model: Name of the model to use.
            base_url: Base URL of the Ollama server.
            api_key: API key (required but ignored by Ollama).
            reasoning_effort: Reasoning effort level (low, medium, high).
            database_path: Path to the SQLite database for session storage.
            mcp_config_path: Path to MCP servers configuration file.
        """
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.reasoning_effort = validate_reasoning_effort(reasoning_effort)
        self.database_path = database_path or Path.home() / ".ollama-agent" / "sessions.db"
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.mcp_config_path = mcp_config_path
        self.mcp_servers = []
        
        # Configure OpenAI client
        set_tracing_disabled(True)
        set_default_openai_api("chat_completions")
        self.client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
        set_default_openai_client(self.client, use_for_tracing=False)
        
        # Create agent and session (MCP servers will be initialized when needed)
        self.agent = self._create_agent()
        self.session = None
        self.session_id = None
        self.reset_session()

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

    def _create_agent(self, model: Optional[str] = None, reasoning_effort: Optional[ReasoningEffortValue] = None) -> Agent:
        """
        Create the AI agent with tools and settings.
        
        Args:
            model: Optional model override. Uses instance model if not provided.
            reasoning_effort: Optional reasoning effort override. Uses instance effort if not provided.
        
        Returns:
            Configured Agent instance.
        """
        return Agent(
            name="Ollama Assistant",
            instructions=(
                "You are a helpful AI assistant that can help with various tasks. "
                "You have access to a tool that allows you to execute operating system commands."
            ),
            model=model or self.model,
            tools=[execute_command],
            mcp_servers=self.mcp_servers,
            model_settings=ModelSettings(
                reasoning=Reasoning(effort=reasoning_effort or self.reasoning_effort)
            )
        )
    
    async def _ensure_mcp_servers_initialized(self) -> None:
        """Initialize MCP servers if not already done."""
        if not self.mcp_servers and self.mcp_config_path:
            self.mcp_servers = await initialize_mcp_servers(self.mcp_config_path)
            # Recreate agent with MCP servers if any were initialized
            if self.mcp_servers:
                self.agent = self._create_agent()
    
    async def cleanup(self) -> None:
        """Cleanup resources, including MCP server connections."""
        if self.mcp_servers:
            await cleanup_mcp_servers(self.mcp_servers)
            self.mcp_servers = []

    async def run_async(self, prompt: str, model: Optional[str] = None, reasoning_effort: Optional[str] = None) -> str:
        """
        Run the agent asynchronously.

        Args:
            prompt: The user's prompt.
            model: Optional model override.
            reasoning_effort: Optional reasoning effort override.

        Returns:
            The agent's response.
        """
        # Ensure MCP servers are initialized
        await self._ensure_mcp_servers_initialized()
        
        # Create agent with overrides if needed
        if model or reasoning_effort:
            validated_effort = validate_reasoning_effort(reasoning_effort) if reasoning_effort else None
            agent = self._create_agent(model=model, reasoning_effort=validated_effort)
        else:
            agent = self.agent
        
        try:
            result = await Runner.run(agent, input=prompt, session=self.session)
            return str(result.final_output)
        except Exception as e:
            logger.error(f"Error running agent: {e}")
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
            with sqlite3.connect(str(self.database_path)) as conn:
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
            logger.error(f"Error listing sessions: {e}")
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
            logger.error(f"Error getting session history: {e}")
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
            with sqlite3.connect(str(self.database_path)) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM agent_messages WHERE session_id = ?", (session_id,))
                cursor.execute("DELETE FROM agent_sessions WHERE session_id = ?", (session_id,))

            # If we deleted the current session, create a new one
            if session_id == self.session_id:
                self.reset_session()
            
            return True
        except Exception as e:
            logger.error(f"Error deleting session: {e}")
            return False
