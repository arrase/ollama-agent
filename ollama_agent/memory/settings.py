"""Mem0 settings configuration."""

from dataclasses import dataclass


@dataclass(eq=True, slots=True)
class Mem0Settings:
    """Configuration for Mem0 persistent memory integration."""

    collection_name: str = "ollama-agent"
    host: str = "localhost"
    port: int = 6333
    embedding_model_dims: int = 768
    llm_model: str = "llama3.1:latest"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2000
    ollama_base_url: str = "http://localhost:11434"
    embedder_model: str = "nomic-embed-text:latest"
    embedder_base_url: str = "http://localhost:11434"
    user_id: str = "default"
