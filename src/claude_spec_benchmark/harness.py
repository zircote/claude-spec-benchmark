"""SWE-bench harness integration for prediction output and assessment.

Provides JSONL output compatibility with SWE-bench assessment harness,
allowing predictions from this framework to be assessed using the
standard swebench.harness.run_assessment() function.

SWE-bench Prediction Format:
    Each line is a JSON object with:
    - instance_id: SWE-bench task identifier
    - model_name_or_path: Model identifier used for generation
    - model_patch: Git-compatible unified diff patch

Example:
    >>> writer = PredictionWriter("predictions.jsonl", model="claude-opus-4")
    >>> writer.write(task_run)
    >>> writer.close()

    # Then use with swebench harness:
    # from swebench.harness.run_assessment import main as run_assess
    # run_assess(predictions_path="predictions.jsonl", ...)
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from claude_spec_benchmark.models import (
    SDDBenchResult,
    SWEBenchTask,
    TaskRun,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SWEBenchPrediction:
    """A single SWE-bench prediction in the expected output format.

    This matches the schema expected by swebench.harness.run_assessment().

    Attributes:
        instance_id: SWE-bench task identifier (e.g., 'django__django-11099')
        model_name_or_path: Model identifier (e.g., 'claude-opus-4')
        model_patch: Git unified diff patch to apply

    Example:
        >>> pred = SWEBenchPrediction(
        ...     instance_id="django__django-11099",
        ...     model_name_or_path="claude-opus-4",
        ...     model_patch="diff --git a/file.py b/file.py\\n...",
        ... )
        >>> pred.to_jsonl()
        '{"instance_id": "django__django-11099", ...}'
    """

    instance_id: str
    model_name_or_path: str
    model_patch: str

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary for JSON serialization."""
        return {
            "instance_id": self.instance_id,
            "model_name_or_path": self.model_name_or_path,
            "model_patch": self.model_patch,
        }

    def to_jsonl(self) -> str:
        """Convert to JSONL line (no trailing newline)."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_task_run(
        cls,
        task_run: TaskRun,
        model_name: str = "claude-code",
    ) -> SWEBenchPrediction:
        """Create prediction from a TaskRun result.

        Args:
            task_run: Completed task run with generated patch.
            model_name: Model identifier to use in output.

        Returns:
            SWEBenchPrediction instance.

        Raises:
            ValueError: If task_run has no generated_patch.
        """
        if not task_run.generated_patch:
            msg = f"TaskRun {task_run.task_id} has no generated patch"
            raise ValueError(msg)

        return cls(
            instance_id=task_run.task_id,
            model_name_or_path=model_name,
            model_patch=task_run.generated_patch,
        )

    @classmethod
    def from_sdd_result(
        cls,
        result: SDDBenchResult,
        model_name: str = "claude-code",
    ) -> SWEBenchPrediction | None:
        """Create prediction from an SDDBenchResult.

        Args:
            result: Complete SDD pipeline result.
            model_name: Model identifier to use in output.

        Returns:
            SWEBenchPrediction if implementation produced a patch, else None.
        """
        # Look for implementation artifacts with patch
        for phase_result in result.phase_results:
            if phase_result.phase == "implementation":
                patch = phase_result.artifacts.get("patch")
                if patch:
                    return cls(
                        instance_id=result.instance_id,
                        model_name_or_path=model_name,
                        model_patch=patch,
                    )
        return None


class PredictionWriter:
    """Writes SWE-bench predictions to JSONL file.

    Supports streaming writes for large benchmark runs.

    Example:
        >>> with PredictionWriter("preds.jsonl", "claude-opus-4") as writer:
        ...     for task_run in results:
        ...         writer.write_task_run(task_run)
    """

    def __init__(
        self,
        output_path: str | Path,
        model_name: str = "claude-code",
        append: bool = False,
    ) -> None:
        """Initialize the prediction writer.

        Args:
            output_path: Path to output JSONL file.
            model_name: Model identifier for all predictions.
            append: If True, append to existing file; else overwrite.
        """
        self._path = Path(output_path)
        self._model_name = model_name
        self._append = append
        self._file: Any = None
        self._count = 0

    def __enter__(self) -> PredictionWriter:
        """Context manager entry."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if self._append else "w"
        self._file = self._path.open(mode, encoding="utf-8")
        return self

    def __exit__(self, *_: object) -> None:
        """Context manager exit."""
        self.close()

    def write(self, prediction: SWEBenchPrediction) -> None:
        """Write a single prediction.

        Args:
            prediction: Prediction to write.
        """
        if self._file is None:
            msg = "Writer not opened. Use context manager or call open()."
            raise RuntimeError(msg)
        self._file.write(prediction.to_jsonl() + "\n")
        self._count += 1

    def write_task_run(self, task_run: TaskRun) -> bool:
        """Write prediction from TaskRun if it has a patch.

        Args:
            task_run: Task execution result.

        Returns:
            True if prediction was written, False if skipped (no patch).
        """
        if not task_run.generated_patch:
            logger.debug("Skipping %s: no patch generated", task_run.task_id)
            return False

        prediction = SWEBenchPrediction.from_task_run(task_run, self._model_name)
        self.write(prediction)
        return True

    def write_sdd_result(self, result: SDDBenchResult) -> bool:
        """Write prediction from SDDBenchResult if it has a patch.

        Args:
            result: SDD pipeline result.

        Returns:
            True if prediction was written, False if skipped.
        """
        prediction = SWEBenchPrediction.from_sdd_result(result, self._model_name)
        if prediction is None:
            logger.debug("Skipping %s: no patch in SDD result", result.instance_id)
            return False

        self.write(prediction)
        return True

    def close(self) -> None:
        """Close the output file."""
        if self._file is not None:
            self._file.close()
            self._file = None

    @property
    def count(self) -> int:
        """Number of predictions written."""
        return self._count


def load_predictions(path: str | Path) -> Iterator[SWEBenchPrediction]:
    """Load predictions from a JSONL file.

    Args:
        path: Path to predictions JSONL file.

    Yields:
        SWEBenchPrediction instances.

    Raises:
        FileNotFoundError: If file doesn't exist.
        json.JSONDecodeError: If JSON parsing fails.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                yield SWEBenchPrediction(
                    instance_id=data["instance_id"],
                    model_name_or_path=data["model_name_or_path"],
                    model_patch=data["model_patch"],
                )
            except (KeyError, json.JSONDecodeError) as e:
                logger.warning("Failed to parse line %d: %s", line_num, e)
                raise


def validate_predictions(predictions_path: str | Path) -> dict[str, Any]:
    """Validate predictions file format for SWE-bench compatibility.

    Args:
        predictions_path: Path to predictions JSONL file.

    Returns:
        Validation result with stats and any errors.
    """
    predictions_path = Path(predictions_path)
    result: dict[str, Any] = {
        "valid": True,
        "total_lines": 0,
        "valid_predictions": 0,
        "errors": [],
        "instance_ids": [],
    }

    if not predictions_path.exists():
        result["valid"] = False
        result["errors"].append(f"File not found: {predictions_path}")
        return result

    with predictions_path.open(encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            result["total_lines"] += 1
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                result["valid"] = False
                result["errors"].append(f"Line {line_num}: Invalid JSON - {e}")
                continue

            # Check required fields
            required = ["instance_id", "model_name_or_path", "model_patch"]
            missing = [f for f in required if f not in data]
            if missing:
                result["valid"] = False
                result["errors"].append(
                    f"Line {line_num}: Missing fields: {missing}"
                )
                continue

            # Validate field types
            for field_name in required:
                if not isinstance(data[field_name], str):
                    result["valid"] = False
                    result["errors"].append(
                        f"Line {line_num}: {field_name} must be string"
                    )

            # Check patch format (should start with diff --git or be empty)
            patch = data.get("model_patch", "")
            if patch and not (
                patch.startswith("diff --git") or patch.startswith("---")
            ):
                result["errors"].append(
                    f"Line {line_num}: Unusual patch format (warning)"
                )

            result["valid_predictions"] += 1
            result["instance_ids"].append(data.get("instance_id", ""))

    return result


@dataclass
class HarnessResult:
    """Result from SWE-bench harness assessment.

    Wraps the output from swebench.harness.run_assessment() into
    a structured format.
    """

    instance_id: str
    resolved: bool
    apply_status: Literal["success", "failed", "timeout"]
    test_status: Literal["passed", "failed", "error"]
    log_output: str = ""
    duration_seconds: float = 0.0
    error_message: str | None = None


@dataclass
class HarnessRunSummary:
    """Summary of a complete harness assessment run."""

    total: int = 0
    resolved: int = 0
    unresolved: int = 0
    apply_failed: int = 0
    test_failed: int = 0
    errors: int = 0
    results: list[HarnessResult] = field(default_factory=list)

    @property
    def resolve_rate(self) -> float:
        """Fraction of instances resolved."""
        return self.resolved / self.total if self.total > 0 else 0.0


def convert_batch_to_jsonl(
    task_runs: Iterable[TaskRun],
    output_path: str | Path,
    model_name: str = "claude-code",
) -> int:
    """Convert a batch of TaskRun results to SWE-bench JSONL format.

    Args:
        task_runs: Iterable of task execution results.
        output_path: Output JSONL file path.
        model_name: Model identifier for predictions.

    Returns:
        Number of predictions written.
    """
    with PredictionWriter(output_path, model_name) as writer:
        for task_run in task_runs:
            writer.write_task_run(task_run)
    return writer.count


def convert_sdd_results_to_jsonl(
    sdd_results: Iterable[SDDBenchResult],
    output_path: str | Path,
    model_name: str = "claude-code",
) -> int:
    """Convert SDD pipeline results to SWE-bench JSONL format.

    Args:
        sdd_results: Iterable of SDD assessment results.
        output_path: Output JSONL file path.
        model_name: Model identifier for predictions.

    Returns:
        Number of predictions written.
    """
    with PredictionWriter(output_path, model_name) as writer:
        for result in sdd_results:
            writer.write_sdd_result(result)
    return writer.count


# =============================================================================
# SWE-bench Harness Integration (for direct harness calls)
# =============================================================================


def create_predictions_for_harness(
    tasks: Iterable[SWEBenchTask],
    task_runs: dict[str, TaskRun],
    output_path: str | Path,
    model_name: str = "claude-code",
) -> Path:
    """Create predictions file for use with SWE-bench harness.

    This function creates a predictions.jsonl file in the format
    expected by swebench.harness.run_assessment().

    Args:
        tasks: Original SWE-bench tasks.
        task_runs: Mapping of instance_id -> TaskRun results.
        output_path: Directory or file path for output.
        model_name: Model identifier for predictions.

    Returns:
        Path to the created predictions file.

    Example:
        >>> predictions_path = create_predictions_for_harness(
        ...     tasks=tasks,
        ...     task_runs=results,
        ...     output_path="./output",
        ...     model_name="claude-opus-4",
        ... )
        >>> # Then run harness:
        >>> # from swebench.harness.run_assessment import main
        >>> # main(predictions_path=str(predictions_path), ...)
    """
    output_path = Path(output_path)

    # If directory, create predictions.jsonl inside
    if output_path.is_dir() or not output_path.suffix:
        output_path.mkdir(parents=True, exist_ok=True)
        predictions_file = output_path / "predictions.jsonl"
    else:
        predictions_file = output_path
        predictions_file.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with predictions_file.open("w", encoding="utf-8") as f:
        for task in tasks:
            task_run = task_runs.get(task.instance_id)
            if task_run and task_run.generated_patch:
                prediction = SWEBenchPrediction(
                    instance_id=task.instance_id,
                    model_name_or_path=model_name,
                    model_patch=task_run.generated_patch,
                )
                f.write(prediction.to_jsonl() + "\n")
                count += 1

    logger.info(
        "Created predictions file: %s (%d predictions)",
        predictions_file,
        count,
    )
    return predictions_file


def generate_harness_command(
    predictions_path: str | Path,
    dataset: str = "princeton-nlp/SWE-bench_Lite",
    split: str = "test",
    max_workers: int = 4,
    run_id: str | None = None,
) -> list[str]:
    """Generate the command to run SWE-bench harness assessment.

    This creates the command line that would invoke the swebench harness
    on the predictions file.

    Args:
        predictions_path: Path to predictions JSONL file.
        dataset: HuggingFace dataset identifier.
        split: Dataset split to use.
        max_workers: Number of parallel workers.
        run_id: Optional run identifier.

    Returns:
        Command as list of strings.

    Example:
        >>> cmd = generate_harness_command("predictions.jsonl")
        >>> subprocess.run(cmd)
    """
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    return [
        "python",
        "-m",
        "swebench.harness.run_assessment",
        "--predictions_path",
        str(predictions_path),
        "--swe_bench_tasks",
        dataset,
        "--split",
        split,
        "--max_workers",
        str(max_workers),
        "--run_id",
        run_id,
    ]
