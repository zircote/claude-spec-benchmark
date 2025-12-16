"""Metrics aggregation and reporting.

Collects evaluation metrics across benchmark runs and generates reports.
"""

import json
import logging
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from claude_spec_benchmark.models import (
    AggregateMetrics,
    BenchmarkRun,
    EvaluationMetrics,
    EvaluationResult,
)

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects and aggregates metrics from benchmark runs.

    Example:
        >>> collector = MetricsCollector()
        >>> collector.add_evaluation(metrics)
        >>> aggregate = collector.compute_aggregate()
        >>> collector.save_report(Path("results/report.json"))
    """

    def __init__(self, run_id: str | None = None) -> None:
        """Initialize collector.

        Args:
            run_id: Optional run identifier.
        """
        self._run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self._evaluations: list[EvaluationMetrics] = []
        self._durations: list[float] = []

    def add_evaluation(
        self,
        evaluation: EvaluationMetrics,
        duration_seconds: float | None = None,
    ) -> None:
        """Add an evaluation result.

        Args:
            evaluation: Task evaluation metrics.
            duration_seconds: Optional task duration.
        """
        self._evaluations.append(evaluation)
        if duration_seconds is not None:
            self._durations.append(duration_seconds)

    def compute_aggregate(self) -> AggregateMetrics:
        """Compute aggregate metrics across all evaluations.

        Returns:
            Aggregate metrics summary.
        """
        if not self._evaluations:
            return AggregateMetrics(
                run_id=self._run_id,
                total_tasks=0,
            )

        total = len(self._evaluations)
        results = [e.result for e in self._evaluations]

        pass_count = results.count(EvaluationResult.PASS)
        partial_count = results.count(EvaluationResult.PARTIAL)
        fail_count = results.count(EvaluationResult.FAIL)
        error_count = results.count(EvaluationResult.ERROR)

        # Duration stats
        avg_duration = statistics.mean(self._durations) if self._durations else 0.0
        median_duration = statistics.median(self._durations) if self._durations else 0.0
        total_duration = sum(self._durations)

        # Test pass rate average
        test_pass_rates = [e.test_pass_rate for e in self._evaluations if e.tests_total > 0]
        avg_test_pass_rate = statistics.mean(test_pass_rates) if test_pass_rates else 0.0

        # Diff similarity average
        diff_sims = [e.diff_metrics.semantic_similarity for e in self._evaluations]
        avg_diff_similarity = statistics.mean(diff_sims) if diff_sims else 0.0

        # Fail-to-pass rate
        f2p_resolved = sum(e.fail_to_pass_resolved for e in self._evaluations)
        f2p_total = sum(e.fail_to_pass_total for e in self._evaluations)
        f2p_rate = f2p_resolved / f2p_total if f2p_total > 0 else 0.0

        # Pass-to-pass rate
        p2p_maintained = sum(e.pass_to_pass_maintained for e in self._evaluations)
        p2p_total = sum(e.pass_to_pass_total for e in self._evaluations)
        p2p_rate = p2p_maintained / p2p_total if p2p_total > 0 else 0.0

        return AggregateMetrics(
            run_id=self._run_id,
            total_tasks=total,
            pass_rate=pass_count / total,
            partial_rate=partial_count / total,
            fail_rate=fail_count / total,
            error_rate=error_count / total,
            timeout_rate=0.0,  # Computed separately from task runs
            avg_duration_seconds=avg_duration,
            median_duration_seconds=median_duration,
            total_duration_seconds=total_duration,
            avg_test_pass_rate=avg_test_pass_rate,
            avg_diff_similarity=avg_diff_similarity,
            fail_to_pass_rate=f2p_rate,
            pass_to_pass_rate=p2p_rate,
        )

    def to_dict(self) -> dict[str, Any]:
        """Export all data as dictionary.

        Returns:
            Dictionary with run_id, aggregate, and individual evaluations.
        """
        return {
            "run_id": self._run_id,
            "timestamp": datetime.now().isoformat(),
            "aggregate": self.compute_aggregate().model_dump(),
            "evaluations": [e.model_dump() for e in self._evaluations],
        }

    def save_report(self, path: Path) -> None:
        """Save metrics report to JSON file.

        Args:
            path: Output file path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        path.write_text(json.dumps(data, indent=2, default=str))
        logger.info("Saved metrics report to %s", path)


class ReportGenerator:
    """Generates human-readable reports from metrics."""

    def __init__(self, console: Console | None = None) -> None:
        """Initialize report generator.

        Args:
            console: Optional Rich console for output.
        """
        self._console = console or Console()

    def print_summary(self, aggregate: AggregateMetrics) -> None:
        """Print summary table to console.

        Args:
            aggregate: Aggregate metrics to display.
        """
        table = Table(title=f"Benchmark Run: {aggregate.run_id}")

        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Tasks", str(aggregate.total_tasks))
        table.add_row("Pass Rate", f"{aggregate.pass_rate:.1%}")
        table.add_row("Partial Rate", f"{aggregate.partial_rate:.1%}")
        table.add_row("Fail Rate", f"{aggregate.fail_rate:.1%}")
        table.add_row("Error Rate", f"{aggregate.error_rate:.1%}")
        table.add_row("", "")
        table.add_row("Avg Duration", f"{aggregate.avg_duration_seconds:.1f}s")
        table.add_row("Median Duration", f"{aggregate.median_duration_seconds:.1f}s")
        table.add_row("Total Duration", f"{aggregate.total_duration_seconds:.1f}s")
        table.add_row("", "")
        table.add_row("Avg Test Pass Rate", f"{aggregate.avg_test_pass_rate:.1%}")
        table.add_row("Avg Diff Similarity", f"{aggregate.avg_diff_similarity:.1%}")
        table.add_row("Fail→Pass Rate", f"{aggregate.fail_to_pass_rate:.1%}")
        table.add_row("Pass→Pass Rate", f"{aggregate.pass_to_pass_rate:.1%}")

        self._console.print(table)

    def print_task_results(
        self,
        evaluations: list[EvaluationMetrics],
        show_all: bool = False,
    ) -> None:
        """Print per-task results table.

        Args:
            evaluations: List of task evaluations.
            show_all: If False, only show non-passing tasks.
        """
        table = Table(title="Task Results")

        table.add_column("Task ID", style="cyan", max_width=40)
        table.add_column("Result", style="bold")
        table.add_column("Tests", style="green")
        table.add_column("F2P", style="yellow")
        table.add_column("Diff Sim", style="blue")

        for e in evaluations:
            if not show_all and e.result == EvaluationResult.PASS:
                continue

            result_style = {
                EvaluationResult.PASS: "green",
                EvaluationResult.PARTIAL: "yellow",
                EvaluationResult.FAIL: "red",
                EvaluationResult.ERROR: "red bold",
            }.get(e.result, "white")

            table.add_row(
                e.task_id,
                f"[{result_style}]{e.result.value}[/{result_style}]",
                f"{e.tests_passed}/{e.tests_total}",
                f"{e.fail_to_pass_resolved}/{e.fail_to_pass_total}",
                f"{e.diff_metrics.semantic_similarity:.1%}",
            )

        self._console.print(table)

    def generate_markdown_report(
        self,
        aggregate: AggregateMetrics,
        evaluations: list[EvaluationMetrics],
    ) -> str:
        """Generate Markdown format report.

        Args:
            aggregate: Aggregate metrics.
            evaluations: Individual task evaluations.

        Returns:
            Markdown formatted report.
        """
        lines = [
            f"# Benchmark Report: {aggregate.run_id}",
            "",
            "## Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Tasks | {aggregate.total_tasks} |",
            f"| Pass Rate | {aggregate.pass_rate:.1%} |",
            f"| Partial Rate | {aggregate.partial_rate:.1%} |",
            f"| Fail Rate | {aggregate.fail_rate:.1%} |",
            f"| Avg Duration | {aggregate.avg_duration_seconds:.1f}s |",
            f"| Fail→Pass Rate | {aggregate.fail_to_pass_rate:.1%} |",
            "",
            "## Results by Task",
            "",
            "| Task | Result | Tests | Diff Sim |",
            "|------|--------|-------|----------|",
        ]

        for e in evaluations:
            emoji = {"pass": "✅", "partial": "🟡", "fail": "❌", "error": "💥"}.get(
                e.result.value, "❓"
            )
            lines.append(
                f"| {e.task_id} | {emoji} {e.result.value} | "
                f"{e.tests_passed}/{e.tests_total} | "
                f"{e.diff_metrics.semantic_similarity:.1%} |"
            )

        return "\n".join(lines)

    def save_markdown_report(
        self,
        path: Path,
        aggregate: AggregateMetrics,
        evaluations: list[EvaluationMetrics],
    ) -> None:
        """Save Markdown report to file.

        Args:
            path: Output file path.
            aggregate: Aggregate metrics.
            evaluations: Individual task evaluations.
        """
        content = self.generate_markdown_report(aggregate, evaluations)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        logger.info("Saved Markdown report to %s", path)
