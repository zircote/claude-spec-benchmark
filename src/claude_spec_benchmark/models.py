"""Data models for SWE-bench test harness and SDD-Bench.

Defines Pydantic models for tasks, evaluation results, metrics,
and SDD-specific constructs (degradation, elicitation, pipeline).

Uses frozen models for immutability where appropriate.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal

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

    PASS = "pass"  # noqa: S105  # Not a password, it's an enum value
    FAIL = "fail"
    ERROR = "error"
    PARTIAL = "partial"


class SWEBenchTask(BaseModel):
    """A single SWE-bench task from the dataset.

    Maps to the SWE-bench Lite dataset schema from HuggingFace.
    """

    model_config = ConfigDict(frozen=True)

    instance_id: str = Field(
        ..., description="Unique task identifier (e.g., 'django__django-11099')"
    )
    repo: str = Field(..., description="Repository name (e.g., 'django/django')")
    base_commit: str = Field(..., description="Git commit SHA to check out")
    problem_statement: str = Field(
        ..., description="Issue description / problem to solve"
    )
    hints_text: str = Field(default="", description="Optional hints for solving")
    created_at: str = Field(default="", description="Issue creation timestamp")
    patch: str = Field(..., description="Gold patch (solution)")
    test_patch: str = Field(default="", description="Test patch to apply")
    version: str = Field(default="", description="Repository version/tag")
    environment_setup_commit: str = Field(
        default="", description="Commit for environment setup"
    )
    fail_to_pass: str = Field(
        default="", description="Tests that should go from fail to pass"
    )
    pass_to_pass: str = Field(
        default="", description="Tests that should remain passing"
    )


class TaskRun(BaseModel):
    """Record of a single task execution attempt."""

    model_config = ConfigDict(frozen=True)

    task_id: str
    run_id: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S")
    )
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
    dataset: str = Field(
        default="princeton-nlp/SWE-bench_Lite", description="HuggingFace dataset"
    )
    split: str = Field(default="test", description="Dataset split to use")
    task_ids: list[str] | None = Field(
        default=None, description="Specific tasks to run (None = all)"
    )
    timeout_seconds: int = Field(
        default=1800, description="Per-task timeout (30 min default)"
    )
    max_workers: int = Field(default=4, description="Parallel task workers")
    docker_image: str = Field(
        default="python:3.11-slim", description="Base Docker image"
    )
    output_dir: Path = Field(
        default=Path("./results"), description="Results output directory"
    )
    claude_code_path: str = Field(
        default="claude", description="Path to claude-code CLI"
    )


class BenchmarkRun(BaseModel):
    """Record of a complete benchmark run."""

    run_id: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S")
    )
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


# =============================================================================
# SDD-Bench Models (Specification-Driven Development)
# =============================================================================


class SpecDegradationLevel(str, Enum):
    """Level of specification degradation for SDD-Bench.

    Controls how much information is removed from the original SWE-bench issue
    to simulate incomplete requirements.

    Levels (from most to least information):
        FULL: Original issue text (baseline, no degradation)
        PARTIAL: Code snippets, stack traces, file paths removed
        VAGUE: Only high-level problem description retained
        MINIMAL: Single sentence summary of the issue
        AMBIGUOUS: Intentionally unclear or contradictory description
    """

    FULL = "full"
    PARTIAL = "partial"
    VAGUE = "vague"
    MINIMAL = "minimal"
    AMBIGUOUS = "ambiguous"


class DegradedSpec(BaseModel):
    """Result of degrading a specification.

    Contains both the degraded text and metadata about what was hidden,
    enabling scoring of elicitation effectiveness.

    Example:
        >>> spec = DegradedSpec(
        ...     degraded_text="Something is wrong with the login",
        ...     hidden_details=["File: auth/views.py", "Error: KeyError"],
        ...     original_text="...",
        ...     level=SpecDegradationLevel.VAGUE,
        ...     seed=42,
        ... )
    """

    model_config = ConfigDict(frozen=True)

    degraded_text: str = Field(..., description="The degraded specification text")
    hidden_details: list[str] = Field(
        default_factory=list,
        description="Details removed during degradation (for scoring)",
    )
    original_text: str = Field(..., description="Original full specification")
    level: SpecDegradationLevel = Field(..., description="Degradation level applied")
    seed: int | None = Field(
        default=None, description="Random seed for reproducibility"
    )


class QuestionCategory(str, Enum):
    """Category of Socratic question for elicitation.

    Based on Paul & Elder's Socratic questioning taxonomy.
    Used to classify agent questions and measure question diversity.
    """

    CLARIFICATION = "clarification"  # "What do you mean by...?"
    ASSUMPTION = "assumption"  # "What are you assuming when...?"
    EVIDENCE = "evidence"  # "What evidence supports...?"
    VIEWPOINT = "viewpoint"  # "What is an alternative...?"
    IMPLICATION = "implication"  # "What would happen if...?"
    META = "meta"  # "Why is this question important?"
    UNKNOWN = "unknown"  # Could not classify


class Requirement(BaseModel):
    """A single extracted requirement from an issue.

    Represents an atomic, testable requirement that can be used
    for elicitation scoring and spec-to-test mapping.

    Example:
        >>> req = Requirement(
        ...     id="REQ-001",
        ...     text="The login form should validate email format",
        ...     category="functional",
        ...     keywords=["login", "email", "validation"],
        ... )
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(..., description="Unique requirement identifier (e.g., 'REQ-001')")
    text: str = Field(..., description="Requirement text")
    category: Literal["functional", "non-functional", "constraint"] = Field(
        default="functional",
        description="Requirement category",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Keywords for relevance matching",
    )
    discoverable: bool = Field(
        default=True,
        description="Whether this can be discovered through elicitation",
    )
    source_span: tuple[int, int] | None = Field(
        default=None,
        description="Character span in original text (start, end)",
    )


class OracleResponse(BaseModel):
    """Response from the elicitation oracle.

    Represents the oracle's answer to an agent's question,
    including relevance scoring and any requirements revealed.

    Example:
        >>> response = OracleResponse(
        ...     answer="The login form expects email format with @",
        ...     relevance_score=0.85,
        ...     revealed_requirements=["REQ-001"],
        ...     question_category=QuestionCategory.CLARIFICATION,
        ... )
    """

    model_config = ConfigDict(frozen=True)

    answer: str = Field(..., description="Oracle's response text")
    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="How relevant the question was (0.0-1.0)",
    )
    revealed_requirements: list[str] = Field(
        default_factory=list,
        description="Requirement IDs revealed by this response",
    )
    question_category: QuestionCategory = Field(
        default=QuestionCategory.UNKNOWN,
        description="Classified category of the question",
    )


class ElicitationMetrics(BaseModel):
    """Metrics summarizing elicitation performance.

    Captures how effectively an agent discovered requirements
    through dialogue with the oracle.

    Example:
        >>> metrics = ElicitationMetrics(
        ...     discovery_rate=0.75,
        ...     question_efficiency=0.15,
        ...     total_questions=20,
        ...     question_distribution={"clarification": 8, "assumption": 5},
        ...     revealed_requirements=["REQ-001", "REQ-002", "REQ-003"],
        ...     hidden_requirements=["REQ-004"],
        ... )
    """

    model_config = ConfigDict(frozen=True)

    discovery_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of discoverable requirements found",
    )
    question_efficiency: float = Field(
        default=0.0,
        description="Discoveries per question asked",
    )
    total_questions: int = Field(default=0, description="Total questions asked")
    question_distribution: dict[str, int] = Field(
        default_factory=dict,
        description="Count of questions by category",
    )
    revealed_requirements: list[str] = Field(
        default_factory=list,
        description="Requirement IDs that were discovered",
    )
    hidden_requirements: list[str] = Field(
        default_factory=list,
        description="Requirement IDs that remained hidden",
    )
    avg_relevance_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Average relevance score of questions",
    )


class SDDPhaseResult(BaseModel):
    """Result of a single SDD pipeline phase.

    Each phase (degrade, elicit, parse, test, implement, refine, validate)
    produces one of these results.

    Example:
        >>> result = SDDPhaseResult(
        ...     phase="elicitation",
        ...     success=True,
        ...     duration_seconds=45.2,
        ...     artifacts={"elicitation_metrics": {...}},
        ... )
    """

    model_config = ConfigDict(frozen=True)

    phase: str = Field(
        ..., description="Phase name (e.g., 'elicitation', 'implementation')"
    )
    success: bool = Field(..., description="Whether the phase completed successfully")
    duration_seconds: float = Field(
        default=0.0, description="Time taken for this phase"
    )
    artifacts: dict[str, Any] = Field(
        default_factory=dict,
        description="Phase-specific output artifacts",
    )
    error: str | None = Field(default=None, description="Error message if failed")


class SDDBenchResult(BaseModel):
    """Complete result of an SDD-Bench evaluation run.

    Aggregates results across all pipeline phases for a single task instance.

    Example:
        >>> result = SDDBenchResult(
        ...     instance_id="django__django-11099",
        ...     degradation_level=SpecDegradationLevel.VAGUE,
        ...     phase_results=[...],
        ...     final_status=EvaluationResult.PASS,
        ...     total_duration_seconds=180.5,
        ... )
    """

    model_config = ConfigDict(frozen=True)

    instance_id: str = Field(..., description="SWE-bench instance ID")
    degradation_level: SpecDegradationLevel = Field(
        ...,
        description="Degradation level used for this run",
    )
    phase_results: list[SDDPhaseResult] = Field(
        default_factory=list,
        description="Results from each pipeline phase",
    )
    final_status: EvaluationResult = Field(
        default=EvaluationResult.FAIL,
        description="Overall evaluation result",
    )
    elicitation_metrics: ElicitationMetrics | None = Field(
        default=None,
        description="Elicitation phase metrics (if applicable)",
    )
    test_metrics: EvaluationMetrics | None = Field(
        default=None,
        description="Test generation metrics (if applicable)",
    )
    implementation_metrics: EvaluationMetrics | None = Field(
        default=None,
        description="Implementation evaluation metrics",
    )
    total_duration_seconds: float = Field(
        default=0.0,
        description="Total time for complete pipeline",
    )
