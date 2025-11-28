"""Utilities to ensure the Mem0 Qdrant backend is available via Docker."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import docker  # type: ignore
from docker.errors import APIError, DockerException, NotFound  # type: ignore

if TYPE_CHECKING:
    from .settings import Mem0Settings

logger = logging.getLogger(__name__)

QDRANT_IMAGE = "qdrant/qdrant:latest"
QDRANT_PORT = "6333/tcp"
CONTAINER_PREFIX = "ollama-agent-qdrant"


class MemoryBootstrapError(RuntimeError):
    """Raised when the Mem0 backend cannot be started."""


def ensure_qdrant_service(settings: Mem0Settings) -> None:
    """Ensure the Qdrant container is running."""
    port = settings.port
    name = f"{CONTAINER_PREFIX}-{port}"

    try:
        client = docker.from_env()
    except DockerException as e:
        raise MemoryBootstrapError("Cannot connect to Docker daemon") from e

    try:
        container = client.containers.get(name)
        if container.status != "running":
            logger.info("Starting Qdrant container: %s", name)
            container.start()
        logger.debug("Qdrant container ready: %s", name)
    except NotFound:
        logger.info("Creating Qdrant container: %s on port %d", name, port)
        _create_container(client, name, port)
    except (APIError, DockerException) as e:
        raise MemoryBootstrapError(f"Qdrant container error: {e}") from e


def _create_container(client, name: str, port: int) -> None:
    """Create and start a new Qdrant container."""
    try:
        client.containers.run(
            QDRANT_IMAGE,
            name=name,
            detach=True,
            ports={QDRANT_PORT: port},
            restart_policy={"Name": "unless-stopped"},
        )
    except (APIError, DockerException) as e:
        raise MemoryBootstrapError(f"Failed to create Qdrant container: {e}") from e
