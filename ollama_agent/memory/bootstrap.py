"""Utilities to ensure the Mem0 Qdrant backend is available via Docker."""

from __future__ import annotations

import logging
import docker  # type: ignore
from typing import Any, Dict, Iterable
from ..settings.configini import Mem0Settings
from docker.errors import APIError, DockerException, NotFound  # type: ignore


logger = logging.getLogger(__name__)

QDRANT_IMAGE = "qdrant/qdrant:latest"
QDRANT_INTERNAL_PORT = "6333/tcp"
CONTAINER_NAME_PREFIX = "ollama-agent-qdrant"


class MemoryBootstrapError(RuntimeError):
    """Raised when the Mem0 backend cannot be started or found."""


def ensure_qdrant_service(settings: Mem0Settings) -> None:
    """Ensure the Qdrant container required by Mem0 is running."""
    if docker is None:  # pragma: no cover - handled via runtime dependency check
        raise MemoryBootstrapError(
            "The docker library is required to manage the Qdrant backend."
        )

    host_port = settings.port or 63333
    container_name = f"{CONTAINER_NAME_PREFIX}-{host_port}"

    try:
        client = docker.from_env()
    except DockerException as exc:  # type: ignore[attr-defined]
        raise MemoryBootstrapError(
            "Could not connect to the Docker daemon; ensure it is running."
        ) from exc

    try:
        container = client.containers.get(container_name)
        _ensure_container_running(container, host_port)
    except NotFound:  # type: ignore[attr-defined]
        logger.info(
            "Creating Qdrant container %s mapped to host port %s", container_name, host_port
        )
        _run_container(client, container_name, host_port)
    except (APIError, DockerException) as exc:  # type: ignore[attr-defined]
        raise MemoryBootstrapError(
            f"Error al comprobar el contenedor Qdrant {container_name}: {exc}"
        ) from exc


def _ensure_container_running(container: Any, host_port: int) -> None:
    container.reload()
    status = getattr(container, "status", "unknown")
    if status != "running":
        logger.info("Starting existing Qdrant container %s", container.name)
        container.start()
        container.reload()
    _validate_port_mapping(container, host_port)
    logger.debug(
        "Contenedor Qdrant %s operativo en el puerto %s", container.name, host_port
    )


def _run_container(client: Any, container_name: str, host_port: int) -> None:
    ports = {QDRANT_INTERNAL_PORT: host_port}
    try:
        client.containers.run(  # type: ignore[call-arg]
            QDRANT_IMAGE,
            name=container_name,
            detach=True,
            ports=ports,
            restart_policy={"Name": "unless-stopped"},
        )
    except (APIError, DockerException) as exc:  # type: ignore[attr-defined]
        raise MemoryBootstrapError(
            f"No se pudo crear el contenedor Qdrant {container_name}: {exc}"
        ) from exc


def _validate_port_mapping(container: Any, expected_host_port: int) -> None:
    container.reload()
    network_settings: Dict[str, Any] = getattr(container, "attrs", {}).get(
        "NetworkSettings", {}
    )
    ports: Dict[str, Iterable[Dict[str, Any]] |
                None] = network_settings.get("Ports", {})
    bindings = ports.get(QDRANT_INTERNAL_PORT) if ports else None
    if not bindings:
        raise MemoryBootstrapError(
            "The Qdrant container does not expose the required port on the host."
        )

    bound_ports = {
        int(binding.get("HostPort", "0"))
        for binding in bindings
        if isinstance(binding, dict) and binding.get("HostPort")
    }
    if expected_host_port not in bound_ports:
        raise MemoryBootstrapError(
            f"The Qdrant container is not publishing port {expected_host_port}."
        )
