"""Docker container isolation manager.

Provides isolated environments for running SWE-bench tasks safely.
Each task runs in its own container with the repository checked out.
"""

import asyncio
import logging
import tarfile
from io import BytesIO
from pathlib import Path
from typing import Any

import docker
from docker.errors import ImageNotFound, NotFound
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
        container_name = (
            f"swe-bench-{task.instance_id.replace('/', '-').replace('__', '-')}"
        )

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
            return str(container.id)

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
        # Copy script to container (container /tmp is isolated)
        script_tar = self._create_tar_bytes("setup.sh", script.encode())
        container.put_archive("/tmp", script_tar)  # noqa: S108

        # Execute setup
        exit_code, output = container.exec_run(
            ["bash", "/tmp/setup.sh"],  # noqa: S108
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
        timeout: int | None = None,  # noqa: ARG002 - Reserved for future use
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
            except NotFound as e:
                msg = f"Container not found: {container_id}"
                raise DockerError(msg) from e

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

        # Write patch to container (container /tmp is isolated)
        patch_tar = self._create_tar_bytes("patch.diff", patch.encode())
        container.put_archive("/tmp", patch_tar)  # noqa: S108

        # Apply patch
        exit_code, stdout, stderr = await self.run_command(
            container_id,
            ["git", "apply", "--verbose", "/tmp/patch.diff"],  # noqa: S108
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
                except Exception as cleanup_err:
                    logger.debug("Cleanup failed for container: %s", cleanup_err)
        except Exception as e:
            logger.warning("Failed to cleanup orphaned containers: %s", e)

    def __enter__(self) -> DockerManager:
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit - cleanup containers."""
        asyncio.get_event_loop().run_until_complete(self.cleanup_all())

    # =========================================================================
    # Image Management
    # =========================================================================

    def get_image_status(self, image_name: str | None = None) -> dict[str, Any]:
        """Get status information for an image.

        Args:
            image_name: Image to check (default: base image).

        Returns:
            Dictionary with image status information.

        Example:
            >>> manager = DockerManager()
            >>> status = manager.get_image_status()
            >>> print(f"Image: {status['name']}, Size: {status['size_mb']:.1f} MB")
        """
        image_name = image_name or self._base_image

        try:
            image = self._client.images.get(image_name)
            return {
                "name": image_name,
                "id": image.short_id,
                "exists": True,
                "size_bytes": image.attrs.get("Size", 0),
                "size_mb": image.attrs.get("Size", 0) / (1024 * 1024),
                "created": image.attrs.get("Created"),
                "tags": image.tags,
            }
        except ImageNotFound:
            return {
                "name": image_name,
                "id": None,
                "exists": False,
                "size_bytes": 0,
                "size_mb": 0,
                "created": None,
                "tags": [],
            }

    def pull_image(
        self,
        image_name: str | None = None,
        progress_callback: Any | None = None,
    ) -> dict[str, Any]:
        """Pull an image with progress reporting.

        Args:
            image_name: Image to pull (default: base image).
            progress_callback: Optional callback for progress updates.
                Called with (layer_id, status, progress_pct).

        Returns:
            Dictionary with pull result information.

        Example:
            >>> def on_progress(layer, status, pct):
            ...     print(f"{layer}: {status} ({pct:.0f}%)")
            >>> result = manager.pull_image(progress_callback=on_progress)
        """
        image_name = image_name or self._base_image

        try:
            # Check if already exists
            before_status = self.get_image_status(image_name)

            # Pull with streaming
            pull_output = []
            layer_progress: dict[str, float] = {}

            for line in self._client.api.pull(image_name, stream=True, decode=True):
                pull_output.append(line)

                if progress_callback and "id" in line:
                    layer_id = line.get("id", "")
                    status = line.get("status", "")

                    # Calculate progress from progressDetail
                    detail = line.get("progressDetail", {})
                    current = detail.get("current", 0)
                    total = detail.get("total", 0)
                    pct = (current / total * 100) if total > 0 else 0

                    layer_progress[layer_id] = pct
                    progress_callback(layer_id, status, pct)

            after_status = self.get_image_status(image_name)

            return {
                "image": image_name,
                "success": True,
                "already_existed": before_status["exists"],
                "size_mb": after_status["size_mb"],
                "layers_pulled": len(layer_progress),
            }

        except Exception as e:
            return {
                "image": image_name,
                "success": False,
                "error": str(e),
                "already_existed": False,
                "size_mb": 0,
                "layers_pulled": 0,
            }

    def list_swebench_images(self) -> list[dict[str, Any]]:
        """List all SWE-bench related images.

        Returns:
            List of image information dictionaries.
        """
        images = []
        for image in self._client.images.list():
            # Check if it's a SWE-bench image (by tag or label)
            tags = image.tags or []
            is_swebench = any(
                "swe-bench" in tag.lower() or "sweb" in tag.lower() for tag in tags
            )

            # Also include python base images commonly used
            is_python_base = any(
                tag.startswith("python:") for tag in tags
            )

            if is_swebench or is_python_base:
                images.append({
                    "id": image.short_id,
                    "tags": tags,
                    "size_mb": image.attrs.get("Size", 0) / (1024 * 1024),
                    "created": image.attrs.get("Created"),
                    "is_swebench": is_swebench,
                })

        return images

    # =========================================================================
    # Container Status and Cleanup
    # =========================================================================

    def list_swebench_containers(
        self,
        include_stopped: bool = True,
    ) -> list[dict[str, Any]]:
        """List all SWE-bench containers.

        Args:
            include_stopped: Whether to include stopped containers.

        Returns:
            List of container information dictionaries.
        """
        containers = self._client.containers.list(
            all=include_stopped,
            filters={"label": "swe-bench=true"},
        )

        result = []
        for container in containers:
            result.append({
                "id": container.short_id,
                "name": container.name,
                "status": container.status,
                "task_id": container.labels.get("task-id", "unknown"),
                "repo": container.labels.get("repo", "unknown"),
                "created": container.attrs.get("Created"),
                "is_running": container.status == "running",
            })

        return result

    def get_orphan_containers(self) -> list[dict[str, Any]]:
        """Get orphaned SWE-bench containers (not in our tracking).

        Orphan containers are those with swe-bench label but not
        tracked by this manager instance.

        Returns:
            List of orphan container information.
        """
        all_containers = self.list_swebench_containers(include_stopped=True)
        tracked_ids = {c.short_id for c in self._containers.values()}

        return [c for c in all_containers if c["id"] not in tracked_ids]

    def cleanup_orphan_containers(self, force: bool = False) -> dict[str, Any]:
        """Remove orphaned SWE-bench containers.

        Args:
            force: Force removal even if running.

        Returns:
            Summary of cleanup operation.
        """
        orphans = self.get_orphan_containers()
        removed = []
        failed = []

        for container_info in orphans:
            try:
                container = self._client.containers.get(container_info["id"])

                if container_info["is_running"]:
                    if force:
                        container.stop(timeout=5)
                    else:
                        failed.append({
                            "id": container_info["id"],
                            "reason": "Running (use force=True to stop)",
                        })
                        continue

                container.remove(force=force)
                removed.append(container_info["id"])
                logger.info("Removed orphan container %s", container_info["id"])

            except Exception as e:
                failed.append({
                    "id": container_info["id"],
                    "reason": str(e),
                })

        return {
            "orphans_found": len(orphans),
            "removed": removed,
            "failed": failed,
            "removed_count": len(removed),
            "failed_count": len(failed),
        }

    # =========================================================================
    # Disk Usage Reporting
    # =========================================================================

    def get_disk_usage(self) -> dict[str, Any]:
        """Get Docker disk usage summary.

        Returns:
            Dictionary with disk usage information.

        Example:
            >>> usage = manager.get_disk_usage()
            >>> print(f"Images: {usage['images_size_mb']:.1f} MB")
            >>> print(f"Containers: {usage['containers_size_mb']:.1f} MB")
        """
        try:
            df = self._client.df()

            # Calculate totals
            images_size = sum(
                img.get("Size", 0) for img in df.get("Images", [])
            )
            containers_size = sum(
                c.get("SizeRw", 0) for c in df.get("Containers", [])
            )
            volumes_size = sum(
                v.get("UsageData", {}).get("Size", 0)
                for v in df.get("Volumes", [])
            )
            build_cache_size = sum(
                bc.get("Size", 0) for bc in df.get("BuildCache", [])
            )

            total_size = images_size + containers_size + volumes_size + build_cache_size

            # SWE-bench specific
            swebench_images = self.list_swebench_images()
            swebench_images_size = sum(img["size_mb"] for img in swebench_images)

            swebench_containers = self.list_swebench_containers()
            swebench_container_count = len(swebench_containers)

            return {
                "images_count": len(df.get("Images", [])),
                "images_size_bytes": images_size,
                "images_size_mb": images_size / (1024 * 1024),
                "containers_count": len(df.get("Containers", [])),
                "containers_size_bytes": containers_size,
                "containers_size_mb": containers_size / (1024 * 1024),
                "volumes_count": len(df.get("Volumes", [])),
                "volumes_size_bytes": volumes_size,
                "volumes_size_mb": volumes_size / (1024 * 1024),
                "build_cache_size_bytes": build_cache_size,
                "build_cache_size_mb": build_cache_size / (1024 * 1024),
                "total_size_bytes": total_size,
                "total_size_mb": total_size / (1024 * 1024),
                "total_size_gb": total_size / (1024 * 1024 * 1024),
                # SWE-bench specific
                "swebench_images_count": len(swebench_images),
                "swebench_images_size_mb": swebench_images_size,
                "swebench_containers_count": swebench_container_count,
            }

        except Exception as e:
            logger.warning("Failed to get disk usage: %s", e)
            return {
                "error": str(e),
                "images_size_mb": 0,
                "containers_size_mb": 0,
                "total_size_mb": 0,
            }

    def prune_unused(
        self,
        images: bool = True,
        containers: bool = True,
        volumes: bool = False,
    ) -> dict[str, Any]:
        """Prune unused Docker resources.

        Args:
            images: Prune dangling images.
            containers: Prune stopped containers.
            volumes: Prune unused volumes (dangerous!).

        Returns:
            Summary of pruned resources.
        """
        result: dict[str, Any] = {
            "containers_pruned": 0,
            "containers_space_mb": 0,
            "images_pruned": 0,
            "images_space_mb": 0,
            "volumes_pruned": 0,
            "volumes_space_mb": 0,
        }

        try:
            if containers:
                prune_result = self._client.containers.prune()
                result["containers_pruned"] = len(
                    prune_result.get("ContainersDeleted", []) or []
                )
                result["containers_space_mb"] = (
                    prune_result.get("SpaceReclaimed", 0) / (1024 * 1024)
                )

            if images:
                prune_result = self._client.images.prune()
                result["images_pruned"] = len(
                    prune_result.get("ImagesDeleted", []) or []
                )
                result["images_space_mb"] = (
                    prune_result.get("SpaceReclaimed", 0) / (1024 * 1024)
                )

            if volumes:
                prune_result = self._client.volumes.prune()
                result["volumes_pruned"] = len(
                    prune_result.get("VolumesDeleted", []) or []
                )
                result["volumes_space_mb"] = (
                    prune_result.get("SpaceReclaimed", 0) / (1024 * 1024)
                )

            result["total_space_mb"] = (
                result["containers_space_mb"]
                + result["images_space_mb"]
                + result["volumes_space_mb"]
            )

        except Exception as e:
            result["error"] = str(e)

        return result
