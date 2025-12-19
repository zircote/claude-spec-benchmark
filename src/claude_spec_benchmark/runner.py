"""Claude Code subprocess runner.

Executes Claude Code CLI as a subprocess to generate patches for SWE-bench tasks.

Security Note: Uses asyncio.create_subprocess_exec (not shell=True) which passes
arguments as an array, preventing shell injection. This is the Python equivalent
of Node.js execFile().
"""

import asyncio
import logging
import shutil
from datetime import datetime
from pathlib import Path

from claude_spec_benchmark.models import SWEBenchTask, TaskRun, TaskStatus

logger = logging.getLogger(__name__)


class RunnerError(Exception):
    """Raised when runner execution fails."""


class ClaudeCodeRunner:
    """Runs Claude Code CLI to generate patches for SWE-bench tasks.

    Spawns claude-code subprocess with the problem statement as input,
    captures the generated patch from stdout/files.

    Example:
        >>> runner = ClaudeCodeRunner()
        >>> result = await runner.run_task(task, workdir)
        >>> print(result.generated_patch)
    """

    DEFAULT_TIMEOUT = 1800  # 30 minutes
    DEFAULT_CLI_PATH = "claude"

    def __init__(
        self,
        cli_path: str = DEFAULT_CLI_PATH,
        timeout_seconds: int = DEFAULT_TIMEOUT,
        model: str | None = None,
        extra_args: list[str] | None = None,
    ) -> None:
        """Initialize the runner.

        Args:
            cli_path: Path to claude-code CLI executable.
            timeout_seconds: Maximum time for task execution.
            model: Optional model override (e.g., 'opus-4', 'sonnet-4').
            extra_args: Additional CLI arguments.
        """
        self._cli_path = cli_path
        self._timeout = timeout_seconds
        self._model = model
        self._extra_args = extra_args or []
        self._validate_cli()

    def _validate_cli(self) -> None:
        """Verify claude-code CLI is available."""
        if not shutil.which(self._cli_path):
            msg = f"Claude Code CLI not found at: {self._cli_path}"
            raise RunnerError(msg)

    def _build_prompt(self, task: SWEBenchTask) -> str:
        """Build the prompt for Claude Code.

        Args:
            task: The SWE-bench task to solve.

        Returns:
            Formatted prompt string.
        """
        prompt_parts = [
            "You are solving a software engineering task from SWE-bench.",
            "",
            "## Repository",
            f"Repository: {task.repo}",
            f"Base commit: {task.base_commit}",
            f"Version: {task.version}" if task.version else "",
            "",
            "## Problem Statement",
            task.problem_statement,
        ]

        if task.hints_text:
            prompt_parts.extend(
                [
                    "",
                    "## Hints",
                    task.hints_text,
                ]
            )

        prompt_parts.extend(
            [
                "",
                "## Instructions",
                "1. Analyze the problem and understand what needs to be fixed",
                "2. Make the minimal changes necessary to solve the issue",
                "3. Generate a unified diff patch that can be applied with `git apply`",
                "4. Output ONLY the patch, starting with `diff --git`",
                "",
                "Generate the patch now:",
            ]
        )

        return "\n".join(prompt_parts)

    def _build_command(self, prompt_file: Path) -> list[str]:
        """Build the CLI command as argument list (no shell interpolation).

        Args:
            prompt_file: Path to file containing the prompt.

        Returns:
            Command as list of strings for subprocess_exec.
        """
        cmd = [
            self._cli_path,
            "--print",  # Non-interactive mode, print output
            "--dangerously-skip-permissions",  # Allow file operations
        ]

        if self._model:
            cmd.extend(["--model", self._model])

        cmd.extend(self._extra_args)

        # Add prompt from file
        cmd.extend(["--input-file", str(prompt_file)])

        return cmd

    async def run_task(
        self,
        task: SWEBenchTask,
        workdir: Path,
        env: dict[str, str] | None = None,
    ) -> TaskRun:
        """Execute Claude Code on a task.

        Args:
            task: The SWE-bench task to solve.
            workdir: Directory containing the checked-out repository.
            env: Optional environment variables.

        Returns:
            TaskRun with execution results.
        """
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        started_at = datetime.now()

        logger.info("Starting task %s (run=%s)", task.instance_id, run_id)

        # Write prompt to temporary file
        prompt = self._build_prompt(task)
        prompt_file = workdir / ".claude_prompt.txt"
        prompt_file.write_text(prompt)

        try:
            cmd = self._build_command(prompt_file)
            logger.debug("Running command: %s", " ".join(cmd))

            # SECURITY: create_subprocess_exec passes args as array (no shell)
            # This is equivalent to Node.js execFile() - safe from injection
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=workdir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**(env or {}), "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"},
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self._timeout,
                )
                stdout = stdout_bytes.decode("utf-8", errors="replace")
                stderr = stderr_bytes.decode("utf-8", errors="replace")
                exit_code = process.returncode

                status = TaskStatus.COMPLETED if exit_code == 0 else TaskStatus.FAILED
                generated_patch = self._extract_patch(stdout)

            except TimeoutError:
                process.kill()
                await process.wait()
                stdout = ""
                stderr = "Task timed out"
                exit_code = -1
                status = TaskStatus.TIMEOUT
                generated_patch = None

        except Exception as e:
            logger.exception("Runner error for task %s", task.instance_id)
            stdout = ""
            stderr = str(e)
            exit_code = -1
            status = TaskStatus.FAILED
            generated_patch = None

        finally:
            # Cleanup prompt file
            if prompt_file.exists():
                prompt_file.unlink()

        completed_at = datetime.now()
        duration = (completed_at - started_at).total_seconds()

        return TaskRun(
            task_id=task.instance_id,
            run_id=run_id,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration,
            generated_patch=generated_patch,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
        )

    def _extract_patch(self, output: str) -> str | None:
        """Extract unified diff patch from Claude's output.

        Args:
            output: Raw output from Claude Code.

        Returns:
            Extracted patch or None if not found.
        """
        lines = output.split("\n")
        patch_lines: list[str] = []
        in_patch = False

        for line in lines:
            # Start of patch
            if line.startswith("diff --git"):
                in_patch = True
                patch_lines = [line]
            elif in_patch:
                # End conditions
                if line.startswith("```") and patch_lines:
                    break
                # Patch content
                if (
                    line.startswith(("---", "+++", "@@", "+", "-", " ", "\\"))
                    or line.startswith("diff --git")
                    or line.startswith("index ")
                    or line.startswith("new file mode")
                    or line.startswith("deleted file mode")
                ) or line.strip() == "":
                    patch_lines.append(line)
                else:
                    # Non-patch line ends the patch
                    break

        if patch_lines:
            return "\n".join(patch_lines).strip()
        return None

    async def run_batch(
        self,
        tasks: list[SWEBenchTask],
        workdirs: dict[str, Path],
        max_concurrent: int = 4,
    ) -> list[TaskRun]:
        """Run multiple tasks concurrently.

        Args:
            tasks: List of tasks to execute.
            workdirs: Mapping of task_id -> workdir.
            max_concurrent: Maximum parallel executions.

        Returns:
            List of TaskRun results.
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def run_with_semaphore(task: SWEBenchTask) -> TaskRun:
            async with semaphore:
                workdir = workdirs.get(task.instance_id)
                if not workdir:
                    return TaskRun(
                        task_id=task.instance_id,
                        status=TaskStatus.FAILED,
                        error_message=f"No workdir for task {task.instance_id}",
                    )
                return await self.run_task(task, workdir)

        coros = [run_with_semaphore(task) for task in tasks]
        results = await asyncio.gather(*coros)
        return list(results)
