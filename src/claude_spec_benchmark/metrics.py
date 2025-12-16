"""Metrics aggregation and reporting.

Collects evaluation metrics across benchmark runs and generates reports.
Includes SDD-specific metrics collection and per-degradation-level breakdowns.
"""

from __future__ import annotations

import json
import logging
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console
from rich.table import Table

from claude_spec_benchmark.models import (
    AggregateMetrics,
    ElicitationMetrics,
    EvaluationMetrics,
    EvaluationResult,
    SDDBenchResult,
    SpecDegradationLevel,
)

if TYPE_CHECKING:
    from claude_spec_benchmark.swt_metrics import SWTMetrics

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
        test_pass_rates = [
            e.test_pass_rate for e in self._evaluations if e.tests_total > 0
        ]
        avg_test_pass_rate = (
            statistics.mean(test_pass_rates) if test_pass_rates else 0.0
        )

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
            "| Metric | Value |",
            "|--------|-------|",
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


# =============================================================================
# SDD-Bench Metrics Models
# =============================================================================


class DegradationLevelMetrics(BaseModel):
    """Aggregated metrics for a single degradation level.

    Tracks performance at each specification degradation level
    (full, partial, vague, minimal, ambiguous).

    Example:
        >>> level_metrics = DegradationLevelMetrics(
        ...     level=SpecDegradationLevel.VAGUE,
        ...     task_count=10,
        ...     pass_rate=0.60,
        ...     avg_discovery_rate=0.75,
        ... )
    """

    model_config = ConfigDict(frozen=True)

    level: SpecDegradationLevel = Field(..., description="Degradation level")
    task_count: int = Field(default=0, description="Number of tasks at this level")
    pass_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Pass rate")
    partial_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Partial rate")
    fail_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Fail rate")
    avg_discovery_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Average requirement discovery rate",
    )
    avg_question_efficiency: float = Field(
        default=0.0, description="Average questions per discovery"
    )
    avg_duration_seconds: float = Field(default=0.0, description="Average pipeline time")


class PhaseTimingMetrics(BaseModel):
    """Timing metrics for each pipeline phase.

    Tracks how long each phase of the SDD pipeline takes on average.

    Example:
        >>> timing = PhaseTimingMetrics(
        ...     degradation_avg=0.1,
        ...     elicitation_avg=30.5,
        ...     implementation_avg=120.0,
        ... )
    """

    model_config = ConfigDict(frozen=True)

    degradation_avg: float = Field(default=0.0, description="Avg degradation time (s)")
    elicitation_avg: float = Field(default=0.0, description="Avg elicitation time (s)")
    parsing_avg: float = Field(default=0.0, description="Avg parsing time (s)")
    test_gen_avg: float = Field(default=0.0, description="Avg test generation time (s)")
    implementation_avg: float = Field(default=0.0, description="Avg impl time (s)")
    refinement_avg: float = Field(default=0.0, description="Avg refinement time (s)")
    validation_avg: float = Field(default=0.0, description="Avg validation time (s)")
    total_avg: float = Field(default=0.0, description="Avg total pipeline time (s)")


class SDDAggregateMetrics(BaseModel):
    """Complete SDD-Bench aggregate metrics.

    Extends standard metrics with SDD-specific breakdowns.

    Example:
        >>> agg = SDDAggregateMetrics(
        ...     run_id="20251216_120000",
        ...     total_tasks=100,
        ...     overall_pass_rate=0.45,
        ...     by_degradation_level=[...],
        ... )
    """

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(..., description="Run identifier")
    total_tasks: int = Field(default=0, description="Total tasks evaluated")

    # Overall rates
    overall_pass_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    overall_partial_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    overall_fail_rate: float = Field(default=0.0, ge=0.0, le=1.0)

    # Elicitation aggregates
    avg_discovery_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Average requirement discovery rate",
    )
    avg_questions_asked: float = Field(default=0.0, description="Avg questions per task")
    avg_question_efficiency: float = Field(
        default=0.0, description="Discoveries per question"
    )

    # Per-level breakdowns
    by_degradation_level: list[DegradationLevelMetrics] = Field(
        default_factory=list, description="Metrics by degradation level"
    )

    # Phase timing
    phase_timing: PhaseTimingMetrics = Field(
        default_factory=PhaseTimingMetrics, description="Per-phase timing"
    )

    # Total timing
    total_duration_seconds: float = Field(default=0.0, description="Total run time")


# =============================================================================
# SDD Metrics Collector
# =============================================================================


class SDDMetricsCollector:
    """Collects and aggregates SDD-Bench specific metrics.

    Extends MetricsCollector with SDD-specific functionality:
    - Per-degradation-level breakdowns
    - Elicitation metrics aggregation
    - Phase timing tracking
    - SWT-bench test metrics integration

    Example:
        >>> collector = SDDMetricsCollector("run_001")
        >>> collector.add_sdd_result(result)
        >>> aggregate = collector.compute_sdd_aggregate()
        >>> collector.save_sdd_report(Path("results/sdd_metrics.json"))
    """

    def __init__(self, run_id: str | None = None) -> None:
        """Initialize SDD metrics collector.

        Args:
            run_id: Optional run identifier.
        """
        self._run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self._sdd_results: list[SDDBenchResult] = []
        self._elicitation_metrics: list[ElicitationMetrics] = []
        self._swt_metrics: list[SWTMetrics] = []
        self._phase_timings: dict[str, list[float]] = defaultdict(list)

    @property
    def run_id(self) -> str:
        """Get the run ID."""
        return self._run_id

    def add_sdd_result(self, result: SDDBenchResult) -> None:
        """Add an SDD pipeline result.

        Args:
            result: Complete SDD-Bench result for a task.
        """
        self._sdd_results.append(result)

        # Extract elicitation metrics
        if result.elicitation_metrics:
            self._elicitation_metrics.append(result.elicitation_metrics)

        # Extract phase timings
        for phase_result in result.phase_results:
            self._phase_timings[phase_result.phase].append(phase_result.duration_seconds)

    def add_swt_metrics(self, metrics: SWTMetrics) -> None:
        """Add SWT-bench test generation metrics.

        Args:
            metrics: SWT metrics for a task.
        """
        self._swt_metrics.append(metrics)

    def compute_sdd_aggregate(self) -> SDDAggregateMetrics:
        """Compute aggregate SDD metrics.

        Returns:
            Complete SDDAggregateMetrics.
        """
        if not self._sdd_results:
            return SDDAggregateMetrics(run_id=self._run_id)

        total_tasks = len(self._sdd_results)

        # Overall result rates
        results = [r.final_status for r in self._sdd_results]
        pass_count = results.count(EvaluationResult.PASS)
        partial_count = results.count(EvaluationResult.PARTIAL)
        fail_count = results.count(EvaluationResult.FAIL)

        # Elicitation aggregates
        discovery_rates = [
            m.discovery_rate for m in self._elicitation_metrics if m.discovery_rate > 0
        ]
        question_counts = [m.total_questions for m in self._elicitation_metrics]
        efficiencies = [
            m.question_efficiency
            for m in self._elicitation_metrics
            if m.question_efficiency > 0
        ]

        avg_discovery = statistics.mean(discovery_rates) if discovery_rates else 0.0
        avg_questions = statistics.mean(question_counts) if question_counts else 0.0
        avg_efficiency = statistics.mean(efficiencies) if efficiencies else 0.0

        # Per-degradation-level metrics
        by_level = self._compute_level_metrics()

        # Phase timing
        phase_timing = self._compute_phase_timing()

        # Total duration
        total_duration = sum(r.total_duration_seconds for r in self._sdd_results)

        return SDDAggregateMetrics(
            run_id=self._run_id,
            total_tasks=total_tasks,
            overall_pass_rate=pass_count / total_tasks,
            overall_partial_rate=partial_count / total_tasks,
            overall_fail_rate=fail_count / total_tasks,
            avg_discovery_rate=avg_discovery,
            avg_questions_asked=avg_questions,
            avg_question_efficiency=avg_efficiency,
            by_degradation_level=by_level,
            phase_timing=phase_timing,
            total_duration_seconds=total_duration,
        )

    def _compute_level_metrics(self) -> list[DegradationLevelMetrics]:
        """Compute metrics broken down by degradation level.

        Returns:
            List of metrics for each degradation level.
        """
        level_results: dict[SpecDegradationLevel, list[SDDBenchResult]] = defaultdict(
            list
        )
        level_elicitation: dict[
            SpecDegradationLevel, list[ElicitationMetrics]
        ] = defaultdict(list)

        for result in self._sdd_results:
            level_results[result.degradation_level].append(result)
            if result.elicitation_metrics:
                level_elicitation[result.degradation_level].append(
                    result.elicitation_metrics
                )

        level_metrics = []
        for level in SpecDegradationLevel:
            results = level_results.get(level, [])
            if not results:
                continue

            task_count = len(results)
            statuses = [r.final_status for r in results]
            pass_count = statuses.count(EvaluationResult.PASS)
            partial_count = statuses.count(EvaluationResult.PARTIAL)
            fail_count = statuses.count(EvaluationResult.FAIL)

            # Elicitation for this level
            elicitations = level_elicitation.get(level, [])
            discovery_rates = [
                m.discovery_rate for m in elicitations if m.discovery_rate > 0
            ]
            efficiencies = [
                m.question_efficiency for m in elicitations if m.question_efficiency > 0
            ]

            # Duration for this level
            durations = [r.total_duration_seconds for r in results]

            level_metrics.append(
                DegradationLevelMetrics(
                    level=level,
                    task_count=task_count,
                    pass_rate=pass_count / task_count,
                    partial_rate=partial_count / task_count,
                    fail_rate=fail_count / task_count,
                    avg_discovery_rate=(
                        statistics.mean(discovery_rates) if discovery_rates else 0.0
                    ),
                    avg_question_efficiency=(
                        statistics.mean(efficiencies) if efficiencies else 0.0
                    ),
                    avg_duration_seconds=(
                        statistics.mean(durations) if durations else 0.0
                    ),
                )
            )

        return level_metrics

    def _compute_phase_timing(self) -> PhaseTimingMetrics:
        """Compute average timing for each pipeline phase.

        Returns:
            PhaseTimingMetrics with averages.
        """
        # Map phase names to model fields
        phase_map = {
            "degradation": "degradation_avg",
            "elicitation": "elicitation_avg",
            "parsing": "parsing_avg",
            "test_generation": "test_gen_avg",
            "implementation": "implementation_avg",
            "refinement": "refinement_avg",
            "validation": "validation_avg",
        }

        timing_data: dict[str, float] = {}
        for phase_name, field_name in phase_map.items():
            times = self._phase_timings.get(phase_name, [])
            timing_data[field_name] = statistics.mean(times) if times else 0.0

        # Total average
        all_totals = [r.total_duration_seconds for r in self._sdd_results]
        timing_data["total_avg"] = statistics.mean(all_totals) if all_totals else 0.0

        return PhaseTimingMetrics(**timing_data)

    def to_dict(self) -> dict[str, Any]:
        """Export all SDD metrics as dictionary.

        Returns:
            Complete metrics dictionary.
        """
        aggregate = self.compute_sdd_aggregate()

        # Convert SWTMetrics if available
        swt_data = []
        for swt in self._swt_metrics:
            swt_data.append(swt.model_dump())

        return {
            "run_id": self._run_id,
            "timestamp": datetime.now().isoformat(),
            "sdd_aggregate": aggregate.model_dump(),
            "sdd_results": [r.model_dump() for r in self._sdd_results],
            "elicitation_metrics": [m.model_dump() for m in self._elicitation_metrics],
            "swt_metrics": swt_data,
        }

    def save_sdd_report(self, path: Path) -> None:
        """Save SDD metrics report to JSON file.

        Args:
            path: Output file path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        path.write_text(json.dumps(data, indent=2, default=str))
        logger.info("Saved SDD metrics report to %s", path)
