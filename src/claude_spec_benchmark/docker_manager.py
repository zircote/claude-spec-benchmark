"""Docker container isolation manager.

Provides isolated environments for running SWE-bench tasks safely.
Each task runs in its own container with the repository checked out.
"""

import asyncio
import logging
import tarfile
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any

import docker
from docker.errors import ContainerError, ImageNotFound, NotFound
from docker.models.containers import Container

from claude_spec_benchmark.models import SWEBenchTask

logger = logging.getLogger(__name__)


class DockerError(Exception):
    """Raised when Docker operations fail."""


class DockerManager:
    """Manages Docker containers for isolated task execution.

    Creates containers with:
    - Repository cloned and checked out to base_commit
    - Python environment with required dependencies
    - Volume mounts for persisting results

    Example:
        >>> manager = DockerManager()
        >>> container_id = await manager.create_task_container(task)
        >>> result = await manager.run_command(container_id, ["pytest"])
        >>> await manager.cleanup(container_id)
    """

    DEFAULT_IMAGE = "python:3.11-slim"
    DEFAULT_TIMEOUT = 3600  # 1 hour max container lifetime

    def __init__(
        self,
        base_image: str = DEFAULT_IMAGE,
        timeout_seconds: int = DEFAULT_TIMEOUT,
        network_enabled: bool = True,
        memory_limit: str = "4g",
        cpu_count: int = 2,
    ) -> None:
        """Initialize Docker manager.

        Args:
            base_image: Base Docker image to use.
            timeout_seconds: Maximum container lifetime.
            network_enabled: Whether containers have network access.
            memory_limit: Container memory limit (e.g., '4g', '8g').
            cpu_count: Number of CPUs allocated per container.
        """
        self._base_image = base_image
        self._timeout = timeout_seconds
        self._network_enabled = network_enabled
        self._memory_limit = memory_limit
        self._cpu_count = cpu_count
        self._client = docker.from_env()
        self._containers: dict[str, Container] = {}

    def _ensure_image(self) -> None:
        """Pull base image if not available locally."""
        try:
            self._client.images.get(self._base_image)
            logger.debug("Image %s found locally", self._base_image)
        except ImageNotFound:
            logger.info("Pulling image %s", self._base_image)
            self._client.images.pull(self._base_image)

    async def create_task_container(
        self,
        task: SWEBenchTask,
        workdir: Path | None = None,
    ) -> str:
        """Create an isolated container for a task.

        Args:
            task: The SWE-bench task to create container for.
            workdir: Optional host directory to mount.

        Returns:
            Container ID.
        """
        self._ensure_image()

        # Create unique container name
        container_name = f"swe-bench-{task.instance_id.replace('/', '-').replace('__', '-')}"

        # Build setup script to clone and checkout repo
        setup_script = self._build_setup_script(task)

        volumes = {}
        if workdir:
            volumes[str(workdir)] = {"bind": "/workspace", "mode": "rw"}

        try:
            # Create container
            container = self._client.containers.create(
                image=self._base_image,
                name=container_name,
                command=["sleep", "infinity"],  # Keep alive
                detach=True,
                working_dir="/repo",
                mem_limit=self._memory_limit,
                cpu_count=self._cpu_count,
                network_disabled=not self._network_enabled,
                volumes=volumes,
                labels={
                    "swe-bench": "true",
                    "task-id": task.instance_id,
                    "repo": task.repo,
                },
            )

            # Start container
            container.start()
            self._containers[container.id] = container

            # Run setup script
            await self._run_setup(container, setup_script)

            logger.info(
                "Created container %s for task %s",
                container.short_id,
                task.instance_id,
            )
            return container.id

        except Exception as e:
            msg = f"Failed to create container for {task.instance_id}: {e}"
            raise DockerError(msg) from e

    def _build_setup_script(self, task: SWEBenchTask) -> str:
        """Build shell script to set up repository in container.

        Args:
            task: The SWE-bench task.

        Returns:
            Shell script as string.
        """
        # Parse repo owner and name
        repo_parts = task.repo.split("/")
        repo_url = f"https://github.com/{task.repo}.git"

        return f"""#!/bin/bash
set -e

# Install git if not present
apt-get update -qq && apt-get install -y -qq git > /dev/null 2>&1

# Clone repository (shallow clone for speed)
git clone --depth 100 {repo_url} /repo

# Checkout specific commit
cd /repo
git fetch --depth 100 origin {task.base_commit}
git checkout {task.base_commit}

# Install Python dependencies if present
if [ -f requirements.txt ]; then
    pip install -q -r requirements.txt || true
fi
if [ -f setup.py ]; then
    pip install -q -e . || true
fi
if [ -f pyproject.toml ]; then
    pip install -q -e . || true
fi

echo "Setup complete for {task.instance_id}"
"""

    async def _run_setup(self, container: Container, script: str) -> None:
        """Run setup script in container.

        Args:
            container: Docker container.
            script: Setup script to execute.
        """
        # Copy script to container
        script_tar = self._create_tar_bytes("setup.sh", script.encode())
        container.put_archive("/tmp", script_tar)

        # Execute setup
        exit_code, output = container.exec_run(
            ["bash", "/tmp/setup.sh"],
            demux=True,
        )

        if exit_code != 0:
            stdout, stderr = output if output else (b"", b"")
            error_msg = (stderr or stdout or b"Unknown error").decode()
            msg = f"Setup failed (exit {exit_code}): {error_msg[:500]}"
            raise DockerError(msg)

    def _create_tar_bytes(self, filename: str, content: bytes) -> bytes:
        """Create a tar archive in memory.

        Args:
            filename: Name of file in archive.
            content: File content.

        Returns:
            Tar archive as bytes.
        """
        tar_stream = BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            file_data = BytesIO(content)
            info = tarfile.TarInfo(name=filename)
            info.size = len(content)
            info.mode = 0o755
            tar.addfile(info, file_data)
        tar_stream.seek(0)
        return tar_stream.read()

    async def run_command(
        self,
        container_id: str,
        command: list[str],
        workdir: str = "/repo",
        timeout: int | None = None,
    ) -> tuple[int, str, str]:
        """Run a command in a container.

        Args:
            container_id: Container to run in.
            command: Command as list of strings.
            workdir: Working directory.
            timeout: Command timeout in seconds.

        Returns:
            Tuple of (exit_code, stdout, stderr).
        """
        container = self._containers.get(container_id)
        if not container:
            try:
                container = self._client.containers.get(container_id)
            except NotFound:
                msg = f"Container not found: {container_id}"
                raise DockerError(msg)

        try:
            exit_code, output = container.exec_run(
                command,
                workdir=workdir,
                demux=True,
            )

            stdout, stderr = output if output else (b"", b"")
            return (
                exit_code or 0,
                (stdout or b"").decode("utf-8", errors="replace"),
                (stderr or b"").decode("utf-8", errors="replace"),
            )

        except Exception as e:
            msg = f"Command execution failed: {e}"
            raise DockerError(msg) from e

    async def apply_patch(
        self,
        container_id: str,
        patch: str,
    ) -> tuple[bool, str]:
        """Apply a patch to the repository in container.

        Args:
            container_id: Container ID.
            patch: Unified diff patch content.

        Returns:
            Tuple of (success, output).
        """
        container = self._containers.get(container_id)
        if not container:
            container = self._client.containers.get(container_id)

        # Write patch to container
        patch_tar = self._create_tar_bytes("patch.diff", patch.encode())
        container.put_archive("/tmp", patch_tar)

        # Apply patch
        exit_code, stdout, stderr = await self.run_command(
            container_id,
            ["git", "apply", "--verbose", "/tmp/patch.diff"],
        )

        success = exit_code == 0
        output = stdout + stderr
        return success, output

    async def get_diff(self, container_id: str) -> str:
        """Get current diff from container repository.

        Args:
            container_id: Container ID.

        Returns:
            Git diff output.
        """
        exit_code, stdout, stderr = await self.run_command(
            container_id,
            ["git", "diff", "--no-color"],
        )
        return stdout

    async def cleanup(self, container_id: str) -> None:
        """Remove a container.

        Args:
            container_id: Container to remove.
        """
        try:
            container = self._containers.pop(container_id, None)
            if not container:
                container = self._client.containers.get(container_id)

            container.stop(timeout=5)
            container.remove(force=True)
            logger.debug("Removed container %s", container_id[:12])

        except NotFound:
            logger.debug("Container %s already removed", container_id[:12])
        except Exception as e:
            logger.warning("Failed to cleanup container %s: %s", container_id[:12], e)

    async def cleanup_all(self) -> None:
        """Remove all managed containers."""
        for container_id in list(self._containers.keys()):
            await self.cleanup(container_id)

        # Also cleanup any orphaned swe-bench containers
        try:
            containers = self._client.containers.list(
                all=True,
                filters={"label": "swe-bench=true"},
            )
            for container in containers:
                try:
                    container.stop(timeout=5)
                    container.remove(force=True)
                except Exception:
                    pass
        except Exception as e:
            logger.warning("Failed to cleanup orphaned containers: %s", e)

    def __enter__(self) -> "DockerManager":
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit - cleanup containers."""
        asyncio.get_event_loop().run_until_complete(self.cleanup_all())
