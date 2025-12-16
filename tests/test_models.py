"""Tests for data models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from claude_spec_benchmark.models import (
    BenchmarkConfig,
    DiffMetrics,
    EvaluationMetrics,
    EvaluationResult,
    SWEBenchTask,
    TaskRun,
    TaskStatus,
)


class TestSWEBenchTask:
    """Tests for SWEBenchTask model."""

    def test_create_minimal_task(self):
        """Test creating task with minimal required fields."""
        task = SWEBenchTask(
            instance_id="test__test-123",
            repo="test/test",
            base_commit="abc123",
            problem_statement="Fix the bug",
            patch="diff --git a/file.py b/file.py",
        )
        assert task.instance_id == "test__test-123"
        assert task.repo == "test/test"
        assert task.hints_text == ""

    def test_task_is_frozen(self, sample_task: SWEBenchTask):
        """Test that tasks are immutable."""
        with pytest.raises(ValidationError):
            sample_task.instance_id = "new-id"  # type: ignore[misc]

    def test_task_from_fixture(self, sample_task: SWEBenchTask):
        """Test sample task fixture."""
        assert sample_task.instance_id == "django__django-11099"
        assert sample_task.repo == "django/django"
        assert "QuerySet" in sample_task.hints_text


class TestTaskRun:
    """Tests for TaskRun model."""

    def test_create_task_run(self):
        """Test creating a task run."""
        run = TaskRun(
            task_id="test__test-123",
            status=TaskStatus.PENDING,
        )
        assert run.task_id == "test__test-123"
        assert run.status == TaskStatus.PENDING
        assert run.generated_patch is None

    def test_task_run_status_transitions(self):
        """Test valid status values."""
        for status in TaskStatus:
            run = TaskRun(task_id="test", status=status)
            assert run.status == status


class TestDiffMetrics:
    """Tests for DiffMetrics model."""

    def test_default_values(self):
        """Test default metric values."""
        metrics = DiffMetrics()
        assert metrics.exact_match is False
        assert metrics.semantic_similarity == 0.0

    def test_exact_match_metrics(self):
        """Test metrics for exact match."""
        metrics = DiffMetrics(
            exact_match=True,
            semantic_similarity=1.0,
            files_modified_match=1.0,
        )
        assert metrics.exact_match is True


class TestEvaluationMetrics:
    """Tests for EvaluationMetrics model."""

    def test_create_pass_evaluation(self, sample_evaluation: EvaluationMetrics):
        """Test passing evaluation."""
        assert sample_evaluation.result == EvaluationResult.PASS
        assert sample_evaluation.test_pass_rate == 1.0

    def test_create_fail_evaluation(self):
        """Test failing evaluation."""
        metrics = EvaluationMetrics(
            task_id="test__test-123",
            result=EvaluationResult.FAIL,
            tests_passed=0,
            tests_failed=5,
            tests_total=5,
        )
        assert metrics.result == EvaluationResult.FAIL
        assert metrics.tests_failed == 5


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig model."""

    def test_default_config(self):
        """Test default configuration values."""
        config = BenchmarkConfig()
        assert config.name == "swe-bench-lite"
        assert config.timeout_seconds == 1800
        assert config.max_workers == 4

    def test_custom_config(self):
        """Test custom configuration."""
        config = BenchmarkConfig(
            name="custom-bench",
            timeout_seconds=3600,
            max_workers=8,
            task_ids=["task1", "task2"],
        )
        assert config.name == "custom-bench"
        assert config.task_ids == ["task1", "task2"]
