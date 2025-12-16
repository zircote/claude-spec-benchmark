"""Data models for SWE-bench test harness.

Defines Pydantic models for tasks, evaluation results, and metrics.
Uses frozen models for immutability where appropriate.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TaskStatus(str, Enum):
    """Status of a benchmark task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


class EvaluationResult(str, Enum):
    """Result of task evaluation."""

    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    PARTIAL = "partial"


class SWEBenchTask(BaseModel):
    """A single SWE-bench task from the dataset.

    Maps to the SWE-bench Lite dataset schema from HuggingFace.
    """

    model_config = ConfigDict(frozen=True)

    instance_id: str = Field(..., description="Unique task identifier (e.g., 'django__django-11099')")
    repo: str = Field(..., description="Repository name (e.g., 'django/django')")
    base_commit: str = Field(..., description="Git commit SHA to check out")
    problem_statement: str = Field(..., description="Issue description / problem to solve")
    hints_text: str = Field(default="", description="Optional hints for solving")
    created_at: str = Field(default="", description="Issue creation timestamp")
    patch: str = Field(..., description="Gold patch (solution)")
    test_patch: str = Field(default="", description="Test patch to apply")
    version: str = Field(default="", description="Repository version/tag")
    environment_setup_commit: str = Field(default="", description="Commit for environment setup")
    fail_to_pass: str = Field(default="", description="Tests that should go from fail to pass")
    pass_to_pass: str = Field(default="", description="Tests that should remain passing")


class TaskRun(BaseModel):
    """Record of a single task execution attempt."""

    model_config = ConfigDict(frozen=True)

    task_id: str
    run_id: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    status: TaskStatus = TaskStatus.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_seconds: float | None = None
    generated_patch: str | None = None
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    container_id: str | None = None
    error_message: str | None = None


class TestResult(BaseModel):
    """Result of running a test suite."""

    model_config = ConfigDict(frozen=True)

    test_name: str
    passed: bool
    duration_seconds: float
    output: str = ""
    error: str | None = None


class DiffMetrics(BaseModel):
    """Metrics comparing generated patch to gold patch."""

    model_config = ConfigDict(frozen=True)

    exact_match: bool = False
    lines_added_match: float = 0.0
    lines_removed_match: float = 0.0
    files_modified_match: float = 0.0
    hunks_match: float = 0.0
    semantic_similarity: float = 0.0


class EvaluationMetrics(BaseModel):
    """Complete evaluation metrics for a task."""

    model_config = ConfigDict(frozen=True)

    task_id: str
    result: EvaluationResult
    tests_passed: int = 0
    tests_failed: int = 0
    tests_total: int = 0
    test_pass_rate: float = 0.0
    diff_metrics: DiffMetrics = Field(default_factory=DiffMetrics)
    fail_to_pass_resolved: int = 0
    fail_to_pass_total: int = 0
    pass_to_pass_maintained: int = 0
    pass_to_pass_total: int = 0
    custom_metrics: dict[str, Any] = Field(default_factory=dict)


class BenchmarkConfig(BaseModel):
    """Configuration for a benchmark run."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(default="swe-bench-lite", description="Benchmark name")
    dataset: str = Field(default="princeton-nlp/SWE-bench_Lite", description="HuggingFace dataset")
    split: str = Field(default="test", description="Dataset split to use")
    task_ids: list[str] | None = Field(default=None, description="Specific tasks to run (None = all)")
    timeout_seconds: int = Field(default=1800, description="Per-task timeout (30 min default)")
    max_workers: int = Field(default=4, description="Parallel task workers")
    docker_image: str = Field(default="python:3.11-slim", description="Base Docker image")
    output_dir: Path = Field(default=Path("./results"), description="Results output directory")
    claude_code_path: str = Field(default="claude", description="Path to claude-code CLI")


class BenchmarkRun(BaseModel):
    """Record of a complete benchmark run."""

    run_id: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    config: BenchmarkConfig
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: datetime | None = None
    total_tasks: int = 0
    completed_tasks: int = 0
    passed_tasks: int = 0
    failed_tasks: int = 0
    error_tasks: int = 0
    timeout_tasks: int = 0
    task_runs: list[TaskRun] = Field(default_factory=list)
    evaluations: list[EvaluationMetrics] = Field(default_factory=list)


class AggregateMetrics(BaseModel):
    """Aggregate metrics across all tasks in a benchmark run."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    total_tasks: int
    pass_rate: float = 0.0
    partial_rate: float = 0.0
    fail_rate: float = 0.0
    error_rate: float = 0.0
    timeout_rate: float = 0.0
    avg_duration_seconds: float = 0.0
    median_duration_seconds: float = 0.0
    total_duration_seconds: float = 0.0
    avg_test_pass_rate: float = 0.0
    avg_diff_similarity: float = 0.0
    fail_to_pass_rate: float = 0.0
    pass_to_pass_rate: float = 0.0
