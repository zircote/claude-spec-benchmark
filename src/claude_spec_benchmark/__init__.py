"""SWE-bench test harness for evaluating Claude's software engineering capabilities.

Also includes SDD-Bench: Spec-Driven Development benchmarking suite.

Example:
    >>> from claude_spec_benchmark import TaskLoader, ClaudeCodeRunner
    >>> loader = TaskLoader()
    >>> for task in loader.iter_tasks(limit=5):
    ...     print(task.instance_id)

SDD-Bench Example:
    >>> from claude_spec_benchmark import DegradationEngine, SpecDegradationLevel
    >>> engine = DegradationEngine()
    >>> result = engine.degrade(issue_text, SpecDegradationLevel.VAGUE, seed=42)
    >>> print(result.degraded_text)
"""

__version__ = "0.1.0"

from claude_spec_benchmark.degradation.engine import DegradationEngine
from claude_spec_benchmark.degradation.patterns import DegradationPatterns
from claude_spec_benchmark.docker_manager import DockerError, DockerManager
from claude_spec_benchmark.elicitation.extraction import RequirementsExtractor
from claude_spec_benchmark.elicitation.oracle import (
    ElicitationOracle,
    OraclePersonality,
)
from claude_spec_benchmark.elicitation.scoring import QuestionScorer
from claude_spec_benchmark.evaluator import EvaluationError, Evaluator, MetricPlugin
from claude_spec_benchmark.harness import (
    HarnessResult,
    HarnessRunSummary,
    PredictionWriter,
    SWEBenchPrediction,
    convert_batch_to_jsonl,
    convert_sdd_results_to_jsonl,
    create_predictions_for_harness,
    generate_harness_command,
    load_predictions,
    validate_predictions,
)
from claude_spec_benchmark.metrics import (
    DegradationLevelMetrics,
    MetricsCollector,
    PhaseTimingMetrics,
    ReportGenerator,
    SDDAggregateMetrics,
    SDDMetricsCollector,
)
from claude_spec_benchmark.models import (
    AggregateMetrics,
    BenchmarkConfig,
    BenchmarkRun,
    DegradedSpec,
    DiffMetrics,
    ElicitationMetrics,
    EvaluationMetrics,
    EvaluationResult,
    OracleResponse,
    QuestionCategory,
    Requirement,
    SDDBenchResult,
    SDDPhaseResult,
    SpecDegradationLevel,
    SWEBenchTask,
    TaskRun,
    TaskStatus,
    TestResult,
)
from claude_spec_benchmark.runner import ClaudeCodeRunner, RunnerError
from claude_spec_benchmark.swt_metrics import (
    ApplicabilityResult,
    CoverageInfo,
    GeneratedTest,
    SWTMetrics,
    TestApplicability,
    TestOutcome,
    TestRunResult,
    aggregate_swt_metrics,
    compute_applicability,
    compute_coverage_delta,
    compute_fail_to_pass_rate,
    compute_success_rate,
    compute_swt_metrics,
)
from claude_spec_benchmark.task_loader import TaskLoader, TaskLoadError

__all__ = [
    # Version
    "__version__",
    # SWE-bench Models
    "AggregateMetrics",
    "BenchmarkConfig",
    "BenchmarkRun",
    "DiffMetrics",
    "EvaluationMetrics",
    "EvaluationResult",
    "SWEBenchTask",
    "TaskRun",
    "TaskStatus",
    "TestResult",
    # SDD-Bench Models
    "DegradedSpec",
    "ElicitationMetrics",
    "OracleResponse",
    "QuestionCategory",
    "Requirement",
    "SDDBenchResult",
    "SDDPhaseResult",
    "SpecDegradationLevel",
    # Task Loading
    "TaskLoader",
    "TaskLoadError",
    # Execution
    "ClaudeCodeRunner",
    "RunnerError",
    # Docker
    "DockerManager",
    "DockerError",
    # Evaluation
    "Evaluator",
    "EvaluationError",
    "MetricPlugin",
    # Metrics
    "MetricsCollector",
    "ReportGenerator",
    "SDDMetricsCollector",
    "SDDAggregateMetrics",
    "DegradationLevelMetrics",
    "PhaseTimingMetrics",
    # SDD-Bench Degradation
    "DegradationEngine",
    "DegradationPatterns",
    # SDD-Bench Elicitation
    "ElicitationOracle",
    "OraclePersonality",
    "QuestionScorer",
    "RequirementsExtractor",
    # SWE-bench Harness Integration
    "SWEBenchPrediction",
    "PredictionWriter",
    "HarnessResult",
    "HarnessRunSummary",
    "load_predictions",
    "validate_predictions",
    "convert_batch_to_jsonl",
    "convert_sdd_results_to_jsonl",
    "create_predictions_for_harness",
    "generate_harness_command",
    # SWT-bench Metrics
    "SWTMetrics",
    "TestApplicability",
    "TestOutcome",
    "GeneratedTest",
    "TestRunResult",
    "ApplicabilityResult",
    "CoverageInfo",
    "compute_applicability",
    "compute_success_rate",
    "compute_fail_to_pass_rate",
    "compute_coverage_delta",
    "compute_swt_metrics",
    "aggregate_swt_metrics",
]
