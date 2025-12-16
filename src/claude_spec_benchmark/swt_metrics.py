"""SWT-bench (SWE-bench for Testing) metrics for test generation assessment.

SWT-bench evaluates the quality of AI-generated tests using these metrics:
- Applicability (W): Can the generated tests be applied/run?
- Success Rate (S): Do tests reveal the bug (fail on buggy, pass on fixed)?
- F→P Rate: Fail-to-pass transition rate across test iterations
- Coverage Delta: How much additional coverage do generated tests add?

Reference:
    SWT-bench: Testing and Validating Code Generation Models
    https://arxiv.org/abs/2406.12952

Example:
    >>> from claude_spec_benchmark.swt_metrics import SWTMetrics, compute_swt_metrics
    >>> metrics = compute_swt_metrics(
    ...     generated_tests=tests,
    ...     buggy_run_results=buggy_results,
    ...     fixed_run_results=fixed_results,
    ... )
    >>> print(f"Success Rate: {metrics.success_rate:.1%}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class TestApplicability(str, Enum):
    """Result of attempting to apply/run a generated test.

    Attributes:
        APPLICABLE: Test applies cleanly and runs
        SYNTAX_ERROR: Test has syntax errors
        IMPORT_ERROR: Test has import/dependency issues
        RUNTIME_ERROR: Test crashes during execution
        TIMEOUT: Test execution timed out
        NOT_APPLICABLE: Test could not be applied for other reasons
    """

    APPLICABLE = "applicable"
    SYNTAX_ERROR = "syntax_error"
    IMPORT_ERROR = "import_error"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"
    NOT_APPLICABLE = "not_applicable"


class TestOutcome(str, Enum):
    """Outcome of running a test against code.

    Attributes:
        PASS: Test passed
        FAIL: Test failed (assertion error)
        ERROR: Test errored (runtime exception)
        SKIP: Test was skipped
    """

    PASS = "pass"  # noqa: S105
    FAIL = "fail"
    ERROR = "error"
    SKIP = "skip"


class SWTMetrics(BaseModel):
    """Complete SWT-bench metrics for a test generation run.

    These metrics assess how well AI-generated tests can identify bugs.

    Attributes:
        applicability_rate: Fraction of tests that can be applied/run
        success_rate: Fraction of applicable tests that reveal the bug
        fail_to_pass_rate: Rate at which tests transition from fail to pass
        coverage_delta: Additional coverage provided by generated tests
        total_tests: Total number of tests generated
        applicable_tests: Number of tests that could be applied
        revealing_tests: Number of tests that reveal the bug

    Example:
        >>> metrics = SWTMetrics(
        ...     applicability_rate=0.85,
        ...     success_rate=0.60,
        ...     fail_to_pass_rate=0.55,
        ...     coverage_delta=0.12,
        ...     total_tests=20,
        ...     applicable_tests=17,
        ...     revealing_tests=10,
        ... )
    """

    model_config = ConfigDict(frozen=True)

    # Core SWT-bench metrics
    applicability_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of generated tests that can be applied (W metric)",
    )
    success_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of applicable tests that reveal the bug (S metric)",
    )
    fail_to_pass_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Rate of fail→pass transitions (F→P metric)",
    )
    coverage_delta: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Additional coverage provided by generated tests",
    )

    # Count metrics
    total_tests: int = Field(default=0, description="Total tests generated")
    applicable_tests: int = Field(default=0, description="Tests that could run")
    revealing_tests: int = Field(
        default=0, description="Tests that reveal the bug (fail buggy, pass fixed)"
    )
    failing_on_buggy: int = Field(default=0, description="Tests failing on buggy code")
    passing_on_fixed: int = Field(default=0, description="Tests passing on fixed code")

    # Detailed breakdown
    applicability_breakdown: dict[str, int] = Field(
        default_factory=dict,
        description="Count by applicability category",
    )
    coverage_before: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Coverage before adding tests"
    )
    coverage_after: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Coverage after adding tests"
    )


@dataclass
class GeneratedTest:
    """Represents a single generated test case.

    Attributes:
        name: Test function/method name
        code: Test source code
        file_path: Suggested file path for the test
    """

    name: str
    code: str
    file_path: str = ""


@dataclass
class TestRunResult:
    """Result of running a test against code.

    Attributes:
        test_name: Name of the test
        outcome: Pass/fail/error/skip
        duration_seconds: Execution time
        error_message: Error details if applicable
        stdout: Standard output
        stderr: Standard error
    """

    test_name: str
    outcome: TestOutcome
    duration_seconds: float = 0.0
    error_message: str | None = None
    stdout: str = ""
    stderr: str = ""


@dataclass
class ApplicabilityResult:
    """Result of attempting to apply a generated test.

    Attributes:
        test_name: Name of the test
        status: Applicability status
        error_message: Error details if not applicable
    """

    test_name: str
    status: TestApplicability
    error_message: str | None = None


@dataclass
class CoverageInfo:
    """Code coverage information.

    Attributes:
        line_rate: Fraction of lines covered
        branch_rate: Fraction of branches covered
        covered_lines: Number of covered lines
        total_lines: Total number of coverable lines
    """

    line_rate: float = 0.0
    branch_rate: float = 0.0
    covered_lines: int = 0
    total_lines: int = 0


def compute_applicability(
    applicability_results: Sequence[ApplicabilityResult],
) -> tuple[float, dict[str, int]]:
    """Compute applicability rate (W metric).

    The applicability rate measures what fraction of generated tests
    can actually be applied and run.

    Args:
        applicability_results: Results of applying each generated test.

    Returns:
        Tuple of (applicability_rate, breakdown_by_status).

    Example:
        >>> results = [
        ...     ApplicabilityResult("test_1", TestApplicability.APPLICABLE),
        ...     ApplicabilityResult("test_2", TestApplicability.SYNTAX_ERROR),
        ... ]
        >>> rate, breakdown = compute_applicability(results)
        >>> rate
        0.5
    """
    if not applicability_results:
        return 0.0, {}

    breakdown: dict[str, int] = {}
    applicable_count = 0

    for result in applicability_results:
        status_key = result.status.value
        breakdown[status_key] = breakdown.get(status_key, 0) + 1
        if result.status == TestApplicability.APPLICABLE:
            applicable_count += 1

    rate = applicable_count / len(applicability_results)
    return rate, breakdown


def compute_success_rate(
    buggy_results: Sequence[TestRunResult],
    fixed_results: Sequence[TestRunResult],
) -> tuple[float, int, int, int]:
    """Compute success rate (S metric).

    A test "succeeds" at revealing a bug if it:
    - FAILS on the buggy code (detects the bug)
    - PASSES on the fixed code (bug is resolved)

    Args:
        buggy_results: Test results on buggy code version.
        fixed_results: Test results on fixed code version.

    Returns:
        Tuple of (success_rate, revealing_count, failing_buggy, passing_fixed).

    Example:
        >>> buggy = [TestRunResult("t1", TestOutcome.FAIL), TestRunResult("t2", TestOutcome.PASS)]
        >>> fixed = [TestRunResult("t1", TestOutcome.PASS), TestRunResult("t2", TestOutcome.PASS)]
        >>> rate, revealing, fail_buggy, pass_fixed = compute_success_rate(buggy, fixed)
        >>> rate
        0.5
        >>> revealing
        1
    """
    if not buggy_results or not fixed_results:
        return 0.0, 0, 0, 0

    # Build lookup maps
    buggy_map = {r.test_name: r.outcome for r in buggy_results}
    fixed_map = {r.test_name: r.outcome for r in fixed_results}

    revealing_count = 0
    failing_on_buggy = 0
    passing_on_fixed = 0

    # Count tests that are in both runs
    common_tests = set(buggy_map.keys()) & set(fixed_map.keys())

    for test_name in common_tests:
        buggy_outcome = buggy_map[test_name]
        fixed_outcome = fixed_map[test_name]

        # Count individual conditions
        if buggy_outcome == TestOutcome.FAIL:
            failing_on_buggy += 1
        if fixed_outcome == TestOutcome.PASS:
            passing_on_fixed += 1

        # A revealing test fails on buggy AND passes on fixed
        if buggy_outcome == TestOutcome.FAIL and fixed_outcome == TestOutcome.PASS:
            revealing_count += 1

    total_applicable = len(common_tests)
    success_rate = revealing_count / total_applicable if total_applicable > 0 else 0.0

    return success_rate, revealing_count, failing_on_buggy, passing_on_fixed


def compute_fail_to_pass_rate(
    buggy_results: Sequence[TestRunResult],
    fixed_results: Sequence[TestRunResult],
) -> float:
    """Compute fail-to-pass transition rate (F→P metric).

    Measures what fraction of tests that fail on buggy code
    will pass on fixed code.

    Args:
        buggy_results: Test results on buggy code.
        fixed_results: Test results on fixed code.

    Returns:
        Fail-to-pass rate (0.0 to 1.0).

    Example:
        >>> buggy = [TestRunResult("t1", TestOutcome.FAIL), TestRunResult("t2", TestOutcome.FAIL)]
        >>> fixed = [TestRunResult("t1", TestOutcome.PASS), TestRunResult("t2", TestOutcome.FAIL)]
        >>> compute_fail_to_pass_rate(buggy, fixed)
        0.5
    """
    if not buggy_results or not fixed_results:
        return 0.0

    buggy_map = {r.test_name: r.outcome for r in buggy_results}
    fixed_map = {r.test_name: r.outcome for r in fixed_results}

    # Find tests that failed on buggy
    failing_on_buggy = [
        name for name, outcome in buggy_map.items() if outcome == TestOutcome.FAIL
    ]

    if not failing_on_buggy:
        return 0.0

    # Count how many now pass on fixed
    transitioned = sum(
        1 for name in failing_on_buggy if fixed_map.get(name) == TestOutcome.PASS
    )

    return transitioned / len(failing_on_buggy)


def compute_coverage_delta(
    coverage_before: CoverageInfo,
    coverage_after: CoverageInfo,
) -> float:
    """Compute coverage improvement from generated tests.

    Args:
        coverage_before: Coverage without the generated tests.
        coverage_after: Coverage with the generated tests.

    Returns:
        Coverage delta (can be negative if coverage decreased).

    Example:
        >>> before = CoverageInfo(line_rate=0.65)
        >>> after = CoverageInfo(line_rate=0.78)
        >>> compute_coverage_delta(before, after)
        0.13
    """
    return coverage_after.line_rate - coverage_before.line_rate


def compute_swt_metrics(
    generated_tests: Sequence[GeneratedTest],
    applicability_results: Sequence[ApplicabilityResult] | None = None,
    buggy_run_results: Sequence[TestRunResult] | None = None,
    fixed_run_results: Sequence[TestRunResult] | None = None,
    coverage_before: CoverageInfo | None = None,
    coverage_after: CoverageInfo | None = None,
) -> SWTMetrics:
    """Compute all SWT-bench metrics from test generation results.

    This is the main entry point for computing SWT-bench metrics.
    All input sequences are optional - metrics will be 0 for missing data.

    Args:
        generated_tests: The generated test cases.
        applicability_results: Results of applying tests (for W metric).
        buggy_run_results: Results of running tests on buggy code.
        fixed_run_results: Results of running tests on fixed code.
        coverage_before: Coverage before adding generated tests.
        coverage_after: Coverage after adding generated tests.

    Returns:
        Complete SWTMetrics object.

    Example:
        >>> tests = [GeneratedTest("test_bug", "def test_bug(): ...")]
        >>> app_results = [ApplicabilityResult("test_bug", TestApplicability.APPLICABLE)]
        >>> buggy = [TestRunResult("test_bug", TestOutcome.FAIL)]
        >>> fixed = [TestRunResult("test_bug", TestOutcome.PASS)]
        >>> metrics = compute_swt_metrics(
        ...     tests, app_results, buggy, fixed
        ... )
        >>> metrics.success_rate
        1.0
    """
    total_tests = len(generated_tests)

    # Compute applicability
    applicability_rate = 0.0
    applicability_breakdown: dict[str, int] = {}
    applicable_count = 0

    if applicability_results:
        applicability_rate, applicability_breakdown = compute_applicability(
            applicability_results
        )
        applicable_count = applicability_breakdown.get("applicable", 0)

    # Compute success rate
    success_rate = 0.0
    revealing_count = 0
    failing_buggy = 0
    passing_fixed = 0

    if buggy_run_results and fixed_run_results:
        success_rate, revealing_count, failing_buggy, passing_fixed = (
            compute_success_rate(buggy_run_results, fixed_run_results)
        )

    # Compute fail-to-pass rate
    f2p_rate = 0.0
    if buggy_run_results and fixed_run_results:
        f2p_rate = compute_fail_to_pass_rate(buggy_run_results, fixed_run_results)

    # Compute coverage delta
    cov_delta = 0.0
    cov_before = 0.0
    cov_after = 0.0

    if coverage_before and coverage_after:
        cov_delta = compute_coverage_delta(coverage_before, coverage_after)
        cov_before = coverage_before.line_rate
        cov_after = coverage_after.line_rate

    return SWTMetrics(
        applicability_rate=applicability_rate,
        success_rate=success_rate,
        fail_to_pass_rate=f2p_rate,
        coverage_delta=cov_delta,
        total_tests=total_tests,
        applicable_tests=applicable_count,
        revealing_tests=revealing_count,
        failing_on_buggy=failing_buggy,
        passing_on_fixed=passing_fixed,
        applicability_breakdown=applicability_breakdown,
        coverage_before=cov_before,
        coverage_after=cov_after,
    )


def aggregate_swt_metrics(metrics_list: Sequence[SWTMetrics]) -> SWTMetrics:
    """Aggregate SWT metrics across multiple test generation runs.

    Useful for computing overall metrics across a benchmark.

    Args:
        metrics_list: List of SWTMetrics from individual runs.

    Returns:
        Aggregated SWTMetrics.

    Example:
        >>> m1 = SWTMetrics(applicability_rate=0.8, success_rate=0.6, total_tests=10)
        >>> m2 = SWTMetrics(applicability_rate=0.9, success_rate=0.7, total_tests=15)
        >>> agg = aggregate_swt_metrics([m1, m2])
        >>> agg.total_tests
        25
    """
    if not metrics_list:
        return SWTMetrics()

    total_tests = sum(m.total_tests for m in metrics_list)
    applicable_tests = sum(m.applicable_tests for m in metrics_list)
    revealing_tests = sum(m.revealing_tests for m in metrics_list)
    failing_buggy = sum(m.failing_on_buggy for m in metrics_list)
    passing_fixed = sum(m.passing_on_fixed for m in metrics_list)

    # Weighted averages using ternary operators
    applicability_rate = applicable_tests / total_tests if total_tests > 0 else 0.0
    success_rate = revealing_tests / applicable_tests if applicable_tests > 0 else 0.0
    f2p_rate = revealing_tests / failing_buggy if failing_buggy > 0 else 0.0

    # Average coverage delta
    cov_deltas = [m.coverage_delta for m in metrics_list if m.coverage_delta > 0]
    avg_cov_delta = sum(cov_deltas) / len(cov_deltas) if cov_deltas else 0.0

    # Merge breakdowns
    merged_breakdown: dict[str, int] = {}
    for m in metrics_list:
        for key, count in m.applicability_breakdown.items():
            merged_breakdown[key] = merged_breakdown.get(key, 0) + count

    return SWTMetrics(
        applicability_rate=applicability_rate,
        success_rate=success_rate,
        fail_to_pass_rate=f2p_rate,
        coverage_delta=avg_cov_delta,
        total_tests=total_tests,
        applicable_tests=applicable_tests,
        revealing_tests=revealing_tests,
        failing_on_buggy=failing_buggy,
        passing_on_fixed=passing_fixed,
        applicability_breakdown=merged_breakdown,
    )
