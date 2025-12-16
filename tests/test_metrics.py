"""Tests for metrics collection and reporting."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from claude_spec_benchmark.metrics import (
    DegradationLevelMetrics,
    MetricsCollector,
    PhaseTimingMetrics,
    ReportGenerator,
    SDDAggregateMetrics,
    SDDMetricsCollector,
)
from claude_spec_benchmark.models import (
    DiffMetrics,
    ElicitationMetrics,
    EvaluationMetrics,
    EvaluationResult,
    SDDBenchResult,
    SDDPhaseResult,
    SpecDegradationLevel,
)


class TestMetricsCollector:
    """Tests for MetricsCollector."""

    def test_empty_collector(self):
        """Test collector with no evaluations."""
        collector = MetricsCollector()
        aggregate = collector.compute_aggregate()
        assert aggregate.total_tasks == 0
        assert aggregate.pass_rate == 0.0

    def test_add_single_evaluation(self, sample_evaluation: EvaluationMetrics):
        """Test adding single evaluation."""
        collector = MetricsCollector()
        collector.add_evaluation(sample_evaluation, duration_seconds=30.0)

        aggregate = collector.compute_aggregate()
        assert aggregate.total_tasks == 1
        assert aggregate.pass_rate == 1.0

    def test_aggregate_multiple_evaluations(self):
        """Test aggregating multiple evaluations."""
        collector = MetricsCollector()

        # Add passing evaluation
        collector.add_evaluation(
            EvaluationMetrics(
                task_id="task1",
                result=EvaluationResult.PASS,
                tests_passed=5,
                tests_total=5,
                test_pass_rate=1.0,
                diff_metrics=DiffMetrics(semantic_similarity=0.9),
            ),
            duration_seconds=30.0,
        )

        # Add failing evaluation
        collector.add_evaluation(
            EvaluationMetrics(
                task_id="task2",
                result=EvaluationResult.FAIL,
                tests_passed=0,
                tests_total=3,
                test_pass_rate=0.0,
                diff_metrics=DiffMetrics(semantic_similarity=0.2),
            ),
            duration_seconds=60.0,
        )

        aggregate = collector.compute_aggregate()
        assert aggregate.total_tasks == 2
        assert aggregate.pass_rate == 0.5  # 1 of 2 passed
        assert aggregate.fail_rate == 0.5
        assert aggregate.avg_duration_seconds == 45.0  # (30 + 60) / 2

    def test_to_dict(self, sample_evaluation: EvaluationMetrics):
        """Test export to dictionary."""
        collector = MetricsCollector(run_id="test-run")
        collector.add_evaluation(sample_evaluation)

        data = collector.to_dict()
        assert data["run_id"] == "test-run"
        assert "aggregate" in data
        assert "evaluations" in data
        assert len(data["evaluations"]) == 1

    def test_save_report(self, sample_evaluation: EvaluationMetrics):
        """Test saving report to file."""
        collector = MetricsCollector()
        collector.add_evaluation(sample_evaluation)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.json"
            collector.save_report(path)

            assert path.exists()
            data = json.loads(path.read_text())
            assert "aggregate" in data


class TestReportGenerator:
    """Tests for ReportGenerator."""

    def test_generate_markdown_report(self, sample_evaluation: EvaluationMetrics):
        """Test markdown report generation."""
        collector = MetricsCollector()
        collector.add_evaluation(sample_evaluation)
        aggregate = collector.compute_aggregate()

        generator = ReportGenerator()
        markdown = generator.generate_markdown_report(
            aggregate,
            [sample_evaluation],
        )

        assert "# Benchmark Report" in markdown
        assert sample_evaluation.task_id in markdown
        assert "Pass Rate" in markdown

    def test_save_markdown_report(self, sample_evaluation: EvaluationMetrics):
        """Test saving markdown report to file."""
        collector = MetricsCollector()
        collector.add_evaluation(sample_evaluation)
        aggregate = collector.compute_aggregate()

        generator = ReportGenerator()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "REPORT.md"
            generator.save_markdown_report(path, aggregate, [sample_evaluation])

            assert path.exists()
            content = path.read_text()
            assert "Benchmark Report" in content


# =============================================================================
# SDD Metrics Collector Tests
# =============================================================================


@pytest.fixture
def sample_sdd_result() -> SDDBenchResult:
    """Sample SDD pipeline result."""
    return SDDBenchResult(
        instance_id="django__django-11099",
        degradation_level=SpecDegradationLevel.VAGUE,
        phase_results=[
            SDDPhaseResult(phase="degradation", success=True, duration_seconds=0.1),
            SDDPhaseResult(phase="elicitation", success=True, duration_seconds=30.5),
            SDDPhaseResult(phase="implementation", success=True, duration_seconds=120.0),
        ],
        final_status=EvaluationResult.PASS,
        elicitation_metrics=ElicitationMetrics(
            discovery_rate=0.75,
            question_efficiency=0.15,
            total_questions=20,
            revealed_requirements=["REQ-001", "REQ-002", "REQ-003"],
            hidden_requirements=["REQ-004"],
        ),
        total_duration_seconds=150.6,
    )


@pytest.fixture
def sample_sdd_result_fail() -> SDDBenchResult:
    """Sample failing SDD result."""
    return SDDBenchResult(
        instance_id="django__django-11100",
        degradation_level=SpecDegradationLevel.MINIMAL,
        phase_results=[
            SDDPhaseResult(phase="degradation", success=True, duration_seconds=0.1),
            SDDPhaseResult(phase="elicitation", success=True, duration_seconds=45.0),
            SDDPhaseResult(phase="implementation", success=False, duration_seconds=90.0),
        ],
        final_status=EvaluationResult.FAIL,
        elicitation_metrics=ElicitationMetrics(
            discovery_rate=0.50,
            question_efficiency=0.10,
            total_questions=25,
            revealed_requirements=["REQ-001", "REQ-002"],
            hidden_requirements=["REQ-003", "REQ-004"],
        ),
        total_duration_seconds=135.1,
    )


class TestSDDMetricsCollector:
    """Tests for SDDMetricsCollector."""

    def test_empty_collector(self):
        """Test collector with no results."""
        collector = SDDMetricsCollector()
        aggregate = collector.compute_sdd_aggregate()

        assert aggregate.total_tasks == 0
        assert aggregate.overall_pass_rate == 0.0

    def test_add_single_result(self, sample_sdd_result: SDDBenchResult):
        """Test adding a single SDD result."""
        collector = SDDMetricsCollector("test-run")
        collector.add_sdd_result(sample_sdd_result)

        aggregate = collector.compute_sdd_aggregate()

        assert aggregate.run_id == "test-run"
        assert aggregate.total_tasks == 1
        assert aggregate.overall_pass_rate == 1.0
        assert aggregate.avg_discovery_rate == 0.75

    def test_aggregate_multiple_results(
        self,
        sample_sdd_result: SDDBenchResult,
        sample_sdd_result_fail: SDDBenchResult,
    ):
        """Test aggregating multiple SDD results."""
        collector = SDDMetricsCollector()
        collector.add_sdd_result(sample_sdd_result)
        collector.add_sdd_result(sample_sdd_result_fail)

        aggregate = collector.compute_sdd_aggregate()

        assert aggregate.total_tasks == 2
        assert aggregate.overall_pass_rate == 0.5  # 1 of 2 passed
        assert aggregate.overall_fail_rate == 0.5
        # Average of 0.75 and 0.50
        assert aggregate.avg_discovery_rate == pytest.approx(0.625)

    def test_degradation_level_breakdown(
        self,
        sample_sdd_result: SDDBenchResult,
        sample_sdd_result_fail: SDDBenchResult,
    ):
        """Test per-degradation-level metrics."""
        collector = SDDMetricsCollector()
        collector.add_sdd_result(sample_sdd_result)  # VAGUE level
        collector.add_sdd_result(sample_sdd_result_fail)  # MINIMAL level

        aggregate = collector.compute_sdd_aggregate()

        # Should have 2 levels represented
        assert len(aggregate.by_degradation_level) == 2

        # Find VAGUE level metrics
        vague_metrics = next(
            (m for m in aggregate.by_degradation_level if m.level == SpecDegradationLevel.VAGUE),
            None,
        )
        assert vague_metrics is not None
        assert vague_metrics.task_count == 1
        assert vague_metrics.pass_rate == 1.0

        # Find MINIMAL level metrics
        minimal_metrics = next(
            (m for m in aggregate.by_degradation_level if m.level == SpecDegradationLevel.MINIMAL),
            None,
        )
        assert minimal_metrics is not None
        assert minimal_metrics.task_count == 1
        assert minimal_metrics.fail_rate == 1.0

    def test_phase_timing(self, sample_sdd_result: SDDBenchResult):
        """Test phase timing extraction."""
        collector = SDDMetricsCollector()
        collector.add_sdd_result(sample_sdd_result)

        aggregate = collector.compute_sdd_aggregate()

        assert aggregate.phase_timing.degradation_avg == pytest.approx(0.1)
        assert aggregate.phase_timing.elicitation_avg == pytest.approx(30.5)
        assert aggregate.phase_timing.implementation_avg == pytest.approx(120.0)
        assert aggregate.phase_timing.total_avg == pytest.approx(150.6)

    def test_to_dict(self, sample_sdd_result: SDDBenchResult):
        """Test export to dictionary."""
        collector = SDDMetricsCollector("export-test")
        collector.add_sdd_result(sample_sdd_result)

        data = collector.to_dict()

        assert data["run_id"] == "export-test"
        assert "sdd_aggregate" in data
        assert "sdd_results" in data
        assert "elicitation_metrics" in data
        assert len(data["sdd_results"]) == 1
        assert len(data["elicitation_metrics"]) == 1

    def test_save_sdd_report(self, sample_sdd_result: SDDBenchResult):
        """Test saving SDD report to file."""
        collector = SDDMetricsCollector()
        collector.add_sdd_result(sample_sdd_result)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sdd_metrics.json"
            collector.save_sdd_report(path)

            assert path.exists()
            data = json.loads(path.read_text())
            assert "sdd_aggregate" in data
            assert data["sdd_aggregate"]["total_tasks"] == 1


class TestSDDMetricsModels:
    """Tests for SDD metrics Pydantic models."""

    def test_degradation_level_metrics(self):
        """Test DegradationLevelMetrics model."""
        metrics = DegradationLevelMetrics(
            level=SpecDegradationLevel.VAGUE,
            task_count=10,
            pass_rate=0.60,
            partial_rate=0.20,
            fail_rate=0.20,
            avg_discovery_rate=0.75,
        )

        assert metrics.level == SpecDegradationLevel.VAGUE
        assert metrics.task_count == 10
        assert metrics.pass_rate == 0.60

    def test_phase_timing_metrics(self):
        """Test PhaseTimingMetrics model."""
        timing = PhaseTimingMetrics(
            degradation_avg=0.1,
            elicitation_avg=30.5,
            implementation_avg=120.0,
            total_avg=150.6,
        )

        assert timing.degradation_avg == 0.1
        assert timing.total_avg == 150.6

    def test_sdd_aggregate_metrics(self):
        """Test SDDAggregateMetrics model."""
        agg = SDDAggregateMetrics(
            run_id="test-run",
            total_tasks=100,
            overall_pass_rate=0.45,
            avg_discovery_rate=0.70,
            total_duration_seconds=3600.0,
        )

        assert agg.run_id == "test-run"
        assert agg.total_tasks == 100
        assert agg.overall_pass_rate == 0.45
