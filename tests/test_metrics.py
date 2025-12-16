"""Tests for metrics collection and reporting."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from claude_spec_benchmark.metrics import MetricsCollector, ReportGenerator
from claude_spec_benchmark.models import (
    DiffMetrics,
    EvaluationMetrics,
    EvaluationResult,
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
