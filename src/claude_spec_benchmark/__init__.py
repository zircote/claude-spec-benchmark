"""SWE-bench test harness for evaluating Claude's software engineering capabilities.

Example:
    >>> from claude_spec_benchmark import TaskLoader, ClaudeCodeRunner
    >>> loader = TaskLoader()
    >>> for task in loader.iter_tasks(limit=5):
    ...     print(task.instance_id)
"""

__version__ = "0.1.0"

from claude_spec_benchmark.docker_manager import DockerError, DockerManager
from claude_spec_benchmark.evaluator import Evaluator, EvaluationError, MetricPlugin
from claude_spec_benchmark.metrics import MetricsCollector, ReportGenerator
from claude_spec_benchmark.models import (
    AggregateMetrics,
    BenchmarkConfig,
    BenchmarkRun,
    DiffMetrics,
    EvaluationMetrics,
    EvaluationResult,
    SWEBenchTask,
    TaskRun,
    TaskStatus,
    TestResult,
)
from claude_spec_benchmark.runner import ClaudeCodeRunner, RunnerError
from claude_spec_benchmark.task_loader import TaskLoadError, TaskLoader

__all__ = [
    # Version
    "__version__",
    # Models
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
]
