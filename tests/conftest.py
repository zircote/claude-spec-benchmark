"""Pytest fixtures for claude-spec-benchmark tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

from claude_spec_benchmark.models import (
    DiffMetrics,
    EvaluationMetrics,
    EvaluationResult,
    SWEBenchTask,
    TaskRun,
    TaskStatus,
)

if TYPE_CHECKING:
    from click.testing import CliRunner


@pytest.fixture
def cli_runner() -> CliRunner:
    """Provide a Click CLI test runner."""
    from click.testing import CliRunner

    return CliRunner()


@pytest.fixture
def sample_task() -> SWEBenchTask:
    """Create a sample SWE-bench task for testing."""
    return SWEBenchTask(
        instance_id="django__django-11099",
        repo="django/django",
        base_commit="a1b2c3d4e5f6",
        problem_statement="Fix the bug in django.db.models where...",
        hints_text="Look at the QuerySet implementation",
        created_at="2023-01-15T10:30:00Z",
        patch="""diff --git a/django/db/models/query.py b/django/db/models/query.py
--- a/django/db/models/query.py
+++ b/django/db/models/query.py
@@ -100,6 +100,7 @@ class QuerySet:
     def filter(self, *args, **kwargs):
+        # Fixed the bug
         return self._filter_or_exclude(False, *args, **kwargs)
""",
        test_patch="",
        version="3.2",
        fail_to_pass='["test_filter_bug"]',
        pass_to_pass='["test_basic_filter", "test_exclude"]',
    )


@pytest.fixture
def sample_task_run(sample_task: SWEBenchTask) -> TaskRun:
    """Create a sample task run for testing."""
    return TaskRun(
        task_id=sample_task.instance_id,
        run_id="20240115_103000",
        status=TaskStatus.COMPLETED,
        duration_seconds=45.5,
        generated_patch="""diff --git a/django/db/models/query.py b/django/db/models/query.py
--- a/django/db/models/query.py
+++ b/django/db/models/query.py
@@ -100,6 +100,7 @@ class QuerySet:
     def filter(self, *args, **kwargs):
+        # Fixed the bug
         return self._filter_or_exclude(False, *args, **kwargs)
""",
        stdout="Patch generated successfully",
        stderr="",
        exit_code=0,
    )


@pytest.fixture
def sample_evaluation(sample_task: SWEBenchTask) -> EvaluationMetrics:
    """Create a sample evaluation result."""
    return EvaluationMetrics(
        task_id=sample_task.instance_id,
        result=EvaluationResult.PASS,
        tests_passed=3,
        tests_failed=0,
        tests_total=3,
        test_pass_rate=1.0,
        diff_metrics=DiffMetrics(
            exact_match=True,
            semantic_similarity=1.0,
        ),
        fail_to_pass_resolved=1,
        fail_to_pass_total=1,
        pass_to_pass_maintained=2,
        pass_to_pass_total=2,
    )


@pytest.fixture
def mock_docker_manager() -> MagicMock:
    """Create a mock Docker manager."""
    mock = MagicMock()
    mock.create_task_container.return_value = "container-123"
    mock.run_command.return_value = (0, "Success", "")
    mock.apply_patch.return_value = (True, "Patch applied")
    mock.get_diff.return_value = ""
    return mock


@pytest.fixture
def mock_dataset() -> list[dict[str, Any]]:
    """Create mock dataset rows."""
    return [
        {
            "instance_id": "django__django-11099",
            "repo": "django/django",
            "base_commit": "a1b2c3d4e5f6",
            "problem_statement": "Fix the bug in django.db.models",
            "hints_text": "",
            "created_at": "2023-01-15",
            "patch": "diff --git a/file.py b/file.py\n--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new",
            "test_patch": "",
            "version": "3.2",
            "FAIL_TO_PASS": '["test1"]',
            "PASS_TO_PASS": '["test2"]',
        },
        {
            "instance_id": "flask__flask-4992",
            "repo": "pallets/flask",
            "base_commit": "b2c3d4e5f6a7",
            "problem_statement": "Fix routing issue",
            "hints_text": "Check app.py",
            "created_at": "2023-02-20",
            "patch": "diff --git a/flask/app.py b/flask/app.py\n--- a/flask/app.py\n+++ b/flask/app.py\n@@ -1 +1 @@\n-old\n+new",
            "test_patch": "",
            "version": "2.3",
            "FAIL_TO_PASS": '["test_route"]',
            "PASS_TO_PASS": "[]",
        },
    ]
