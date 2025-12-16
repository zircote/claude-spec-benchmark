"""Evaluation engine for SWE-bench task results.

Implements multiple evaluation strategies:
- Test-based pass/fail (standard SWE-bench metric)
- Diff similarity scoring (patch comparison)
- Custom extensible metrics
"""

import difflib
import json
import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from unidiff import PatchSet

from claude_spec_benchmark.docker_manager import DockerManager
from claude_spec_benchmark.models import (
    DiffMetrics,
    EvaluationMetrics,
    EvaluationResult,
    SWEBenchTask,
    TaskRun,
    TestResult,
)

logger = logging.getLogger(__name__)


class EvaluationError(Exception):
    """Raised when evaluation fails."""


class MetricPlugin(ABC):
    """Base class for custom metric plugins."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique metric name."""

    @abstractmethod
    def compute(
        self,
        task: SWEBenchTask,
        task_run: TaskRun,
        test_results: Sequence[TestResult],
    ) -> dict[str, Any]:
        """Compute custom metrics.

        Args:
            task: The original task.
            task_run: Execution results.
            test_results: Test execution results.

        Returns:
            Dictionary of metric name -> value.
        """


class Evaluator:
    """Evaluates task runs against gold patches and test suites.

    Combines multiple evaluation approaches:
    1. Test-based: Run test suite, check fail-to-pass and pass-to-pass
    2. Diff-based: Compare generated patch to gold patch
    3. Custom: Pluggable metric computation

    Example:
        >>> evaluator = Evaluator(docker_manager)
        >>> metrics = await evaluator.evaluate(task, task_run)
        >>> print(metrics.result)
    """

    def __init__(
        self,
        docker_manager: DockerManager,
        metric_plugins: list[MetricPlugin] | None = None,
    ) -> None:
        """Initialize evaluator.

        Args:
            docker_manager: Docker manager for running tests.
            metric_plugins: Optional custom metric plugins.
        """
        self._docker = docker_manager
        self._plugins = metric_plugins or []

    async def evaluate(
        self,
        task: SWEBenchTask,
        task_run: TaskRun,
        container_id: str | None = None,
    ) -> EvaluationMetrics:
        """Run full evaluation on a task run.

        Args:
            task: The original SWE-bench task.
            task_run: The execution results.
            container_id: Optional container for test execution.

        Returns:
            Complete evaluation metrics.
        """
        # If no patch was generated, fail immediately
        if not task_run.generated_patch:
            return EvaluationMetrics(
                task_id=task.instance_id,
                result=EvaluationResult.FAIL,
            )

        # Compute diff metrics
        diff_metrics = self._compute_diff_metrics(
            task.patch,
            task_run.generated_patch,
        )

        # Run tests if container available
        test_results: list[TestResult] = []
        fail_to_pass_resolved = 0
        fail_to_pass_total = 0
        pass_to_pass_maintained = 0
        pass_to_pass_total = 0

        if container_id:
            test_results = await self._run_tests(container_id, task)

            # Parse fail-to-pass and pass-to-pass expectations
            fail_to_pass_tests = self._parse_test_list(task.fail_to_pass)
            pass_to_pass_tests = self._parse_test_list(task.pass_to_pass)

            fail_to_pass_total = len(fail_to_pass_tests)
            pass_to_pass_total = len(pass_to_pass_tests)

            # Check test expectations
            passed_names = {r.test_name for r in test_results if r.passed}

            for test_name in fail_to_pass_tests:
                if self._test_matches(test_name, passed_names):
                    fail_to_pass_resolved += 1

            for test_name in pass_to_pass_tests:
                if self._test_matches(test_name, passed_names):
                    pass_to_pass_maintained += 1

        # Compute custom metrics
        custom_metrics: dict[str, Any] = {}
        for plugin in self._plugins:
            try:
                plugin_metrics = plugin.compute(task, task_run, test_results)
                custom_metrics[plugin.name] = plugin_metrics
            except Exception as e:
                logger.warning("Plugin %s failed: %s", plugin.name, e)

        # Determine overall result
        tests_passed = sum(1 for r in test_results if r.passed)
        tests_failed = len(test_results) - tests_passed
        test_pass_rate = tests_passed / len(test_results) if test_results else 0.0

        result = self._determine_result(
            diff_metrics=diff_metrics,
            fail_to_pass_resolved=fail_to_pass_resolved,
            fail_to_pass_total=fail_to_pass_total,
            pass_to_pass_maintained=pass_to_pass_maintained,
            pass_to_pass_total=pass_to_pass_total,
        )

        return EvaluationMetrics(
            task_id=task.instance_id,
            result=result,
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            tests_total=len(test_results),
            test_pass_rate=test_pass_rate,
            diff_metrics=diff_metrics,
            fail_to_pass_resolved=fail_to_pass_resolved,
            fail_to_pass_total=fail_to_pass_total,
            pass_to_pass_maintained=pass_to_pass_maintained,
            pass_to_pass_total=pass_to_pass_total,
            custom_metrics=custom_metrics,
        )

    def _compute_diff_metrics(
        self,
        gold_patch: str,
        generated_patch: str,
    ) -> DiffMetrics:
        """Compare generated patch to gold patch.

        Args:
            gold_patch: Expected correct patch.
            generated_patch: Model-generated patch.

        Returns:
            Diff comparison metrics.
        """
        # Exact match check
        exact_match = gold_patch.strip() == generated_patch.strip()

        try:
            gold_patchset = PatchSet.from_string(gold_patch)
            gen_patchset = PatchSet.from_string(generated_patch)

            # File-level comparison
            gold_files = {p.path for p in gold_patchset}
            gen_files = {p.path for p in gen_patchset}
            files_overlap = len(gold_files & gen_files)
            files_match = files_overlap / len(gold_files) if gold_files else 0.0

            # Line-level comparison
            gold_added = sum(p.added for p in gold_patchset)
            gold_removed = sum(p.removed for p in gold_patchset)
            gen_added = sum(p.added for p in gen_patchset)
            gen_removed = sum(p.removed for p in gen_patchset)

            lines_added_match = min(gen_added, gold_added) / gold_added if gold_added else 1.0
            lines_removed_match = min(gen_removed, gold_removed) / gold_removed if gold_removed else 1.0

            # Hunk-level comparison
            gold_hunks = sum(len(list(p)) for p in gold_patchset)
            gen_hunks = sum(len(list(p)) for p in gen_patchset)
            hunks_match = min(gen_hunks, gold_hunks) / gold_hunks if gold_hunks else 1.0

        except Exception as e:
            logger.warning("Failed to parse patches: %s", e)
            files_match = 0.0
            lines_added_match = 0.0
            lines_removed_match = 0.0
            hunks_match = 0.0

        # Semantic similarity using difflib
        semantic_similarity = difflib.SequenceMatcher(
            None,
            gold_patch,
            generated_patch,
        ).ratio()

        return DiffMetrics(
            exact_match=exact_match,
            lines_added_match=lines_added_match,
            lines_removed_match=lines_removed_match,
            files_modified_match=files_match,
            hunks_match=hunks_match,
            semantic_similarity=semantic_similarity,
        )

    async def _run_tests(
        self,
        container_id: str,
        task: SWEBenchTask,
    ) -> list[TestResult]:
        """Run test suite in container.

        Args:
            container_id: Container with repo set up.
            task: Task with test information.

        Returns:
            List of test results.
        """
        results: list[TestResult] = []

        # Apply test patch if present
        if task.test_patch:
            success, output = await self._docker.apply_patch(
                container_id,
                task.test_patch,
            )
            if not success:
                logger.warning("Failed to apply test patch: %s", output[:200])

        # Run pytest with JSON output
        exit_code, stdout, stderr = await self._docker.run_command(
            container_id,
            [
                "python", "-m", "pytest",
                "--tb=short",
                "-v",
                "--json-report",
                "--json-report-file=/tmp/test_results.json",
            ],
        )

        # Try to parse JSON results
        try:
            _, json_stdout, _ = await self._docker.run_command(
                container_id,
                ["cat", "/tmp/test_results.json"],
            )
            test_data = json.loads(json_stdout)

            for test in test_data.get("tests", []):
                results.append(TestResult(
                    test_name=test.get("nodeid", "unknown"),
                    passed=test.get("outcome") == "passed",
                    duration_seconds=test.get("duration", 0.0),
                    output=test.get("call", {}).get("stdout", ""),
                    error=test.get("call", {}).get("longrepr"),
                ))

        except (json.JSONDecodeError, Exception) as e:
            logger.warning("Failed to parse test JSON: %s", e)
            # Fall back to parsing stdout
            results = self._parse_pytest_output(stdout + stderr)

        return results

    def _parse_pytest_output(self, output: str) -> list[TestResult]:
        """Parse pytest verbose output for test results.

        Args:
            output: pytest stdout/stderr.

        Returns:
            List of parsed test results.
        """
        results: list[TestResult] = []

        # Match patterns like "test_foo.py::test_bar PASSED" or "FAILED"
        pattern = r"(\S+::\S+)\s+(PASSED|FAILED|ERROR|SKIPPED)"

        for match in re.finditer(pattern, output):
            test_name = match.group(1)
            outcome = match.group(2)
            results.append(TestResult(
                test_name=test_name,
                passed=outcome == "PASSED",
                duration_seconds=0.0,
            ))

        return results

    def _parse_test_list(self, test_list_str: str) -> list[str]:
        """Parse test list from SWE-bench format.

        Args:
            test_list_str: JSON or comma-separated test names.

        Returns:
            List of test names.
        """
        if not test_list_str:
            return []

        try:
            # Try JSON format first
            return json.loads(test_list_str)
        except json.JSONDecodeError:
            # Fall back to comma-separated
            return [t.strip() for t in test_list_str.split(",") if t.strip()]

    def _test_matches(self, expected: str, passed_names: set[str]) -> bool:
        """Check if an expected test name matches any passed test.

        Args:
            expected: Expected test name (may be partial).
            passed_names: Set of passed test names.

        Returns:
            True if test is considered passing.
        """
        # Exact match
        if expected in passed_names:
            return True

        # Partial match (test name contains expected string)
        for passed in passed_names:
            if expected in passed:
                return True

        return False

    def _determine_result(
        self,
        diff_metrics: DiffMetrics,
        fail_to_pass_resolved: int,
        fail_to_pass_total: int,
        pass_to_pass_maintained: int,
        pass_to_pass_total: int,
    ) -> EvaluationResult:
        """Determine overall evaluation result.

        Standard SWE-bench metric: PASS if all fail-to-pass tests
        now pass AND all pass-to-pass tests still pass.

        Args:
            diff_metrics: Patch comparison metrics.
            fail_to_pass_resolved: Number of fail-to-pass tests now passing.
            fail_to_pass_total: Total fail-to-pass tests expected.
            pass_to_pass_maintained: Number of pass-to-pass tests still passing.
            pass_to_pass_total: Total pass-to-pass tests expected.

        Returns:
            Overall result (PASS, PARTIAL, FAIL).
        """
        # If exact match, it's a pass
        if diff_metrics.exact_match:
            return EvaluationResult.PASS

        # Standard SWE-bench criteria
        if fail_to_pass_total > 0:
            all_fail_to_pass = fail_to_pass_resolved == fail_to_pass_total
            all_pass_to_pass = pass_to_pass_maintained == pass_to_pass_total

            if all_fail_to_pass and all_pass_to_pass:
                return EvaluationResult.PASS
            elif fail_to_pass_resolved > 0 or pass_to_pass_maintained > 0:
                return EvaluationResult.PARTIAL

        # High semantic similarity is partial success
        if diff_metrics.semantic_similarity > 0.7:
            return EvaluationResult.PARTIAL

        return EvaluationResult.FAIL

    def register_plugin(self, plugin: MetricPlugin) -> None:
        """Register a custom metric plugin.

        Args:
            plugin: MetricPlugin instance.
        """
        self._plugins.append(plugin)
