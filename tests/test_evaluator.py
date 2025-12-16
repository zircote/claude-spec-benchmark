"""Tests for evaluation engine."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from claude_spec_benchmark.evaluator import Evaluator, MetricPlugin
from claude_spec_benchmark.models import (
    EvaluationResult,
    SWEBenchTask,
    TaskRun,
    TaskStatus,
    TestResult,
)


class TestDiffMetrics:
    """Tests for diff-based evaluation."""

    @pytest.fixture
    def evaluator(self, mock_docker_manager: MagicMock) -> Evaluator:
        """Create evaluator with mock Docker."""
        return Evaluator(mock_docker_manager)

    def test_exact_match_detection(
        self,
        evaluator: Evaluator,
        sample_task: SWEBenchTask,
    ):
        """Test exact patch match is detected."""
        # Use the same patch as gold
        metrics = evaluator._compute_diff_metrics(
            sample_task.patch,
            sample_task.patch,
        )
        assert metrics.exact_match is True
        assert metrics.semantic_similarity == 1.0

    def test_different_patch_not_exact(
        self,
        evaluator: Evaluator,
        sample_task: SWEBenchTask,
    ):
        """Test different patches are not exact match."""
        different_patch = """diff --git a/other.py b/other.py
--- a/other.py
+++ b/other.py
@@ -1 +1 @@
-old
+new
"""
        metrics = evaluator._compute_diff_metrics(
            sample_task.patch,
            different_patch,
        )
        assert metrics.exact_match is False
        assert metrics.semantic_similarity < 1.0

    def test_semantic_similarity_partial_match(
        self,
        evaluator: Evaluator,
    ):
        """Test semantic similarity for similar patches."""
        gold = """diff --git a/file.py b/file.py
--- a/file.py
+++ b/file.py
@@ -1,3 +1,3 @@
-def foo():
-    return 1
+def foo():
+    return 2
"""
        generated = """diff --git a/file.py b/file.py
--- a/file.py
+++ b/file.py
@@ -1,3 +1,3 @@
-def foo():
-    return 1
+def foo():
+    return 3
"""
        metrics = evaluator._compute_diff_metrics(gold, generated)
        assert metrics.exact_match is False
        # Similar structure should have high similarity
        assert metrics.semantic_similarity > 0.5


class TestTestParsing:
    """Tests for test list parsing."""

    @pytest.fixture
    def evaluator(self, mock_docker_manager: MagicMock) -> Evaluator:
        """Create evaluator with mock Docker."""
        return Evaluator(mock_docker_manager)

    def test_parse_json_test_list(self, evaluator: Evaluator):
        """Test parsing JSON format test list."""
        test_list = '["test_foo", "test_bar", "test_baz"]'
        result = evaluator._parse_test_list(test_list)
        assert result == ["test_foo", "test_bar", "test_baz"]

    def test_parse_empty_test_list(self, evaluator: Evaluator):
        """Test parsing empty test list."""
        assert evaluator._parse_test_list("") == []
        assert evaluator._parse_test_list("[]") == []

    def test_parse_comma_separated(self, evaluator: Evaluator):
        """Test fallback to comma-separated parsing."""
        test_list = "test_foo, test_bar"
        result = evaluator._parse_test_list(test_list)
        assert "test_foo" in result
        assert "test_bar" in result


class TestResultDetermination:
    """Tests for overall result determination."""

    @pytest.fixture
    def evaluator(self, mock_docker_manager: MagicMock) -> Evaluator:
        """Create evaluator with mock Docker."""
        return Evaluator(mock_docker_manager)

    def test_exact_match_is_pass(self, evaluator: Evaluator):
        """Test exact match results in PASS."""
        from claude_spec_benchmark.models import DiffMetrics

        metrics = DiffMetrics(exact_match=True, semantic_similarity=1.0)
        result = evaluator._determine_result(
            diff_metrics=metrics,
            fail_to_pass_resolved=0,
            fail_to_pass_total=0,
            pass_to_pass_maintained=0,
            pass_to_pass_total=0,
        )
        assert result == EvaluationResult.PASS

    def test_all_tests_pass_is_pass(self, evaluator: Evaluator):
        """Test all fail-to-pass resolved is PASS."""
        from claude_spec_benchmark.models import DiffMetrics

        metrics = DiffMetrics(exact_match=False, semantic_similarity=0.8)
        result = evaluator._determine_result(
            diff_metrics=metrics,
            fail_to_pass_resolved=2,
            fail_to_pass_total=2,
            pass_to_pass_maintained=3,
            pass_to_pass_total=3,
        )
        assert result == EvaluationResult.PASS

    def test_partial_tests_is_partial(self, evaluator: Evaluator):
        """Test partial test resolution is PARTIAL."""
        from claude_spec_benchmark.models import DiffMetrics

        metrics = DiffMetrics(exact_match=False, semantic_similarity=0.5)
        result = evaluator._determine_result(
            diff_metrics=metrics,
            fail_to_pass_resolved=1,
            fail_to_pass_total=3,
            pass_to_pass_maintained=2,
            pass_to_pass_total=2,
        )
        assert result == EvaluationResult.PARTIAL

    def test_no_tests_passed_is_fail(self, evaluator: Evaluator):
        """Test no tests resolved is FAIL."""
        from claude_spec_benchmark.models import DiffMetrics

        metrics = DiffMetrics(exact_match=False, semantic_similarity=0.3)
        result = evaluator._determine_result(
            diff_metrics=metrics,
            fail_to_pass_resolved=0,
            fail_to_pass_total=3,
            pass_to_pass_maintained=0,
            pass_to_pass_total=2,
        )
        assert result == EvaluationResult.FAIL


class TestMetricPlugin:
    """Tests for custom metric plugins."""

    def test_register_custom_plugin(self, mock_docker_manager: MagicMock):
        """Test registering and using custom plugins."""

        class CustomMetric(MetricPlugin):
            @property
            def name(self) -> str:
                return "custom"

            def compute(
                self,
                task: SWEBenchTask,
                task_run: TaskRun,
                test_results: list[TestResult],
            ) -> dict[str, Any]:
                return {"custom_score": 42}

        evaluator = Evaluator(mock_docker_manager)
        evaluator.register_plugin(CustomMetric())

        assert len(evaluator._plugins) == 1
        assert evaluator._plugins[0].name == "custom"
