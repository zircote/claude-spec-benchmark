"""Tests for SWT-bench metrics module."""

from __future__ import annotations

import pytest

from claude_spec_benchmark.swt_metrics import (
    ApplicabilityResult,
    CoverageInfo,
    GeneratedTest,
    SWTMetrics,
    TestApplicability,
    TestOutcome,
    TestRunResult,
    aggregate_swt_metrics,
    compute_applicability,
    compute_coverage_delta,
    compute_fail_to_pass_rate,
    compute_success_rate,
    compute_swt_metrics,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_tests() -> list[GeneratedTest]:
    """Sample generated tests."""
    return [
        GeneratedTest("test_bug_detection", "def test_bug_detection(): assert False"),
        GeneratedTest("test_edge_case", "def test_edge_case(): assert True"),
        GeneratedTest("test_validation", "def test_validation(): ..."),
    ]


@pytest.fixture
def all_applicable_results() -> list[ApplicabilityResult]:
    """All tests applicable."""
    return [
        ApplicabilityResult("test_1", TestApplicability.APPLICABLE),
        ApplicabilityResult("test_2", TestApplicability.APPLICABLE),
        ApplicabilityResult("test_3", TestApplicability.APPLICABLE),
    ]


@pytest.fixture
def mixed_applicability_results() -> list[ApplicabilityResult]:
    """Mix of applicable and non-applicable tests."""
    return [
        ApplicabilityResult("test_1", TestApplicability.APPLICABLE),
        ApplicabilityResult("test_2", TestApplicability.SYNTAX_ERROR),
        ApplicabilityResult("test_3", TestApplicability.APPLICABLE),
        ApplicabilityResult("test_4", TestApplicability.IMPORT_ERROR),
    ]


@pytest.fixture
def buggy_results() -> list[TestRunResult]:
    """Test results on buggy code."""
    return [
        TestRunResult("test_1", TestOutcome.FAIL),  # Detects bug
        TestRunResult("test_2", TestOutcome.FAIL),  # Detects bug
        TestRunResult("test_3", TestOutcome.PASS),  # Doesn't detect bug
    ]


@pytest.fixture
def fixed_results() -> list[TestRunResult]:
    """Test results on fixed code."""
    return [
        TestRunResult("test_1", TestOutcome.PASS),  # Bug fixed
        TestRunResult("test_2", TestOutcome.FAIL),  # Still failing (flaky or wrong)
        TestRunResult("test_3", TestOutcome.PASS),  # Still passing
    ]


# =============================================================================
# Test compute_applicability
# =============================================================================


class TestComputeApplicability:
    """Tests for compute_applicability function."""

    def test_all_applicable(self, all_applicable_results: list[ApplicabilityResult]):
        """Test when all tests are applicable."""
        rate, breakdown = compute_applicability(all_applicable_results)

        assert rate == 1.0
        assert breakdown["applicable"] == 3

    def test_mixed_applicability(
        self, mixed_applicability_results: list[ApplicabilityResult]
    ):
        """Test with mixed applicability."""
        rate, breakdown = compute_applicability(mixed_applicability_results)

        assert rate == 0.5
        assert breakdown["applicable"] == 2
        assert breakdown["syntax_error"] == 1
        assert breakdown["import_error"] == 1

    def test_empty_results(self):
        """Test with empty results."""
        rate, breakdown = compute_applicability([])

        assert rate == 0.0
        assert breakdown == {}

    def test_none_applicable(self):
        """Test when no tests are applicable."""
        results = [
            ApplicabilityResult("t1", TestApplicability.SYNTAX_ERROR),
            ApplicabilityResult("t2", TestApplicability.RUNTIME_ERROR),
        ]
        rate, breakdown = compute_applicability(results)

        assert rate == 0.0
        assert "applicable" not in breakdown


# =============================================================================
# Test compute_success_rate
# =============================================================================


class TestComputeSuccessRate:
    """Tests for compute_success_rate function."""

    def test_perfect_success(self):
        """Test when all tests reveal the bug."""
        buggy = [
            TestRunResult("t1", TestOutcome.FAIL),
            TestRunResult("t2", TestOutcome.FAIL),
        ]
        fixed = [
            TestRunResult("t1", TestOutcome.PASS),
            TestRunResult("t2", TestOutcome.PASS),
        ]

        rate, revealing, fail_buggy, pass_fixed = compute_success_rate(buggy, fixed)

        assert rate == 1.0
        assert revealing == 2
        assert fail_buggy == 2
        assert pass_fixed == 2

    def test_partial_success(
        self, buggy_results: list[TestRunResult], fixed_results: list[TestRunResult]
    ):
        """Test with partial success."""
        rate, revealing, fail_buggy, pass_fixed = compute_success_rate(
            buggy_results, fixed_results
        )

        # test_1: FAIL->PASS (revealing)
        # test_2: FAIL->FAIL (not revealing)
        # test_3: PASS->PASS (not revealing)
        assert revealing == 1
        assert fail_buggy == 2
        assert pass_fixed == 2
        assert rate == pytest.approx(1 / 3)

    def test_no_revealing_tests(self):
        """Test when no tests reveal the bug."""
        buggy = [TestRunResult("t1", TestOutcome.PASS)]
        fixed = [TestRunResult("t1", TestOutcome.PASS)]

        rate, revealing, fail_buggy, pass_fixed = compute_success_rate(buggy, fixed)

        assert rate == 0.0
        assert revealing == 0

    def test_empty_results(self):
        """Test with empty results."""
        rate, revealing, fail_buggy, pass_fixed = compute_success_rate([], [])

        assert rate == 0.0
        assert revealing == 0


# =============================================================================
# Test compute_fail_to_pass_rate
# =============================================================================


class TestComputeFailToPassRate:
    """Tests for compute_fail_to_pass_rate function."""

    def test_all_transitions(self):
        """Test when all failing tests transition to pass."""
        buggy = [
            TestRunResult("t1", TestOutcome.FAIL),
            TestRunResult("t2", TestOutcome.FAIL),
        ]
        fixed = [
            TestRunResult("t1", TestOutcome.PASS),
            TestRunResult("t2", TestOutcome.PASS),
        ]

        rate = compute_fail_to_pass_rate(buggy, fixed)
        assert rate == 1.0

    def test_partial_transitions(
        self, buggy_results: list[TestRunResult], fixed_results: list[TestRunResult]
    ):
        """Test with partial transitions."""
        rate = compute_fail_to_pass_rate(buggy_results, fixed_results)

        # 2 failing on buggy, 1 transitions to pass
        assert rate == 0.5

    def test_no_failing_tests(self):
        """Test when no tests fail on buggy."""
        buggy = [TestRunResult("t1", TestOutcome.PASS)]
        fixed = [TestRunResult("t1", TestOutcome.PASS)]

        rate = compute_fail_to_pass_rate(buggy, fixed)
        assert rate == 0.0

    def test_empty_results(self):
        """Test with empty results."""
        rate = compute_fail_to_pass_rate([], [])
        assert rate == 0.0


# =============================================================================
# Test compute_coverage_delta
# =============================================================================


class TestComputeCoverageDelta:
    """Tests for compute_coverage_delta function."""

    def test_positive_delta(self):
        """Test positive coverage improvement."""
        before = CoverageInfo(line_rate=0.65)
        after = CoverageInfo(line_rate=0.78)

        delta = compute_coverage_delta(before, after)
        assert delta == pytest.approx(0.13)

    def test_no_change(self):
        """Test no coverage change."""
        before = CoverageInfo(line_rate=0.70)
        after = CoverageInfo(line_rate=0.70)

        delta = compute_coverage_delta(before, after)
        assert delta == 0.0

    def test_negative_delta(self):
        """Test coverage decrease (possible with flaky tests)."""
        before = CoverageInfo(line_rate=0.80)
        after = CoverageInfo(line_rate=0.75)

        delta = compute_coverage_delta(before, after)
        assert delta == pytest.approx(-0.05)


# =============================================================================
# Test compute_swt_metrics
# =============================================================================


class TestComputeSWTMetrics:
    """Tests for compute_swt_metrics function."""

    def test_full_computation(
        self,
        sample_tests: list[GeneratedTest],
        all_applicable_results: list[ApplicabilityResult],
    ):
        """Test computing all metrics."""
        buggy = [
            TestRunResult("test_1", TestOutcome.FAIL),
            TestRunResult("test_2", TestOutcome.FAIL),
            TestRunResult("test_3", TestOutcome.PASS),
        ]
        fixed = [
            TestRunResult("test_1", TestOutcome.PASS),
            TestRunResult("test_2", TestOutcome.PASS),
            TestRunResult("test_3", TestOutcome.PASS),
        ]
        coverage_before = CoverageInfo(line_rate=0.60)
        coverage_after = CoverageInfo(line_rate=0.75)

        metrics = compute_swt_metrics(
            generated_tests=sample_tests,
            applicability_results=all_applicable_results,
            buggy_run_results=buggy,
            fixed_run_results=fixed,
            coverage_before=coverage_before,
            coverage_after=coverage_after,
        )

        assert metrics.total_tests == 3
        assert metrics.applicability_rate == 1.0
        assert metrics.applicable_tests == 3
        assert metrics.success_rate == pytest.approx(2 / 3)
        assert metrics.revealing_tests == 2
        assert metrics.fail_to_pass_rate == 1.0
        assert metrics.coverage_delta == pytest.approx(0.15)

    def test_minimal_computation(self, sample_tests: list[GeneratedTest]):
        """Test with minimal inputs."""
        metrics = compute_swt_metrics(generated_tests=sample_tests)

        assert metrics.total_tests == 3
        assert metrics.applicability_rate == 0.0
        assert metrics.success_rate == 0.0

    def test_empty_tests(self):
        """Test with no generated tests."""
        metrics = compute_swt_metrics(generated_tests=[])

        assert metrics.total_tests == 0
        assert metrics.applicability_rate == 0.0


# =============================================================================
# Test SWTMetrics model
# =============================================================================


class TestSWTMetrics:
    """Tests for SWTMetrics model."""

    def test_create_metrics(self):
        """Test creating metrics instance."""
        metrics = SWTMetrics(
            applicability_rate=0.85,
            success_rate=0.60,
            fail_to_pass_rate=0.55,
            coverage_delta=0.12,
            total_tests=20,
            applicable_tests=17,
            revealing_tests=10,
        )

        assert metrics.applicability_rate == 0.85
        assert metrics.success_rate == 0.60
        assert metrics.total_tests == 20

    def test_frozen_model(self):
        """Test that model is frozen."""
        metrics = SWTMetrics()

        with pytest.raises(Exception):  # ValidationError or similar
            metrics.success_rate = 0.5  # type: ignore

    def test_defaults(self):
        """Test default values."""
        metrics = SWTMetrics()

        assert metrics.applicability_rate == 0.0
        assert metrics.success_rate == 0.0
        assert metrics.total_tests == 0
        assert metrics.applicability_breakdown == {}


# =============================================================================
# Test aggregate_swt_metrics
# =============================================================================


class TestAggregateSWTMetrics:
    """Tests for aggregate_swt_metrics function."""

    def test_aggregate_multiple(self):
        """Test aggregating multiple metrics."""
        m1 = SWTMetrics(
            applicability_rate=0.8,
            success_rate=0.6,
            fail_to_pass_rate=0.7,
            coverage_delta=0.10,
            total_tests=10,
            applicable_tests=8,
            revealing_tests=5,
            failing_on_buggy=6,
            passing_on_fixed=7,
            applicability_breakdown={"applicable": 8, "syntax_error": 2},
        )
        m2 = SWTMetrics(
            applicability_rate=0.9,
            success_rate=0.7,
            fail_to_pass_rate=0.8,
            coverage_delta=0.15,
            total_tests=15,
            applicable_tests=14,
            revealing_tests=10,
            failing_on_buggy=11,
            passing_on_fixed=12,
            applicability_breakdown={"applicable": 14, "import_error": 1},
        )

        agg = aggregate_swt_metrics([m1, m2])

        assert agg.total_tests == 25
        assert agg.applicable_tests == 22
        assert agg.revealing_tests == 15
        assert agg.applicability_rate == pytest.approx(22 / 25)
        assert agg.success_rate == pytest.approx(15 / 22)
        assert agg.applicability_breakdown["applicable"] == 22
        assert agg.applicability_breakdown["syntax_error"] == 2
        assert agg.applicability_breakdown["import_error"] == 1

    def test_aggregate_empty(self):
        """Test aggregating empty list."""
        agg = aggregate_swt_metrics([])

        assert agg.total_tests == 0
        assert agg.applicability_rate == 0.0

    def test_aggregate_single(self):
        """Test aggregating single metrics."""
        m = SWTMetrics(
            applicability_rate=0.85,
            success_rate=0.65,
            total_tests=10,
            applicable_tests=9,
            revealing_tests=6,
        )

        agg = aggregate_swt_metrics([m])

        assert agg.total_tests == 10
        assert agg.applicable_tests == 9


# =============================================================================
# Test helper classes
# =============================================================================


class TestHelperClasses:
    """Tests for helper dataclasses."""

    def test_generated_test(self):
        """Test GeneratedTest creation."""
        test = GeneratedTest(
            name="test_example",
            code="def test_example(): assert True",
            file_path="tests/test_example.py",
        )

        assert test.name == "test_example"
        assert "assert True" in test.code

    def test_test_run_result(self):
        """Test TestRunResult creation."""
        result = TestRunResult(
            test_name="test_foo",
            outcome=TestOutcome.FAIL,
            duration_seconds=0.5,
            error_message="AssertionError: expected True",
        )

        assert result.test_name == "test_foo"
        assert result.outcome == TestOutcome.FAIL
        assert result.error_message is not None

    def test_coverage_info(self):
        """Test CoverageInfo creation."""
        cov = CoverageInfo(
            line_rate=0.85,
            branch_rate=0.70,
            covered_lines=850,
            total_lines=1000,
        )

        assert cov.line_rate == 0.85
        assert cov.covered_lines == 850
