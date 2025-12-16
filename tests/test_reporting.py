"""Tests for the reporting module.

Tests SDDReportGenerator and ReportSummary for all output formats.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from claude_spec_benchmark.models import (
    ElicitationMetrics,
    EvaluationResult,
    SDDBenchResult,
    SDDPhaseResult,
    SpecDegradationLevel,
)
from claude_spec_benchmark.reporting import ReportSummary, SDDReportGenerator

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture
def sample_results() -> list[SDDBenchResult]:
    """Create sample SDD results for testing."""
    return [
        SDDBenchResult(
            instance_id="django__django-11099",
            degradation_level=SpecDegradationLevel.FULL,
            phase_results=[
                SDDPhaseResult(phase="degrade", success=True, duration_seconds=0.1),
                SDDPhaseResult(phase="elicit", success=True, duration_seconds=30.0),
                SDDPhaseResult(phase="parse", success=True, duration_seconds=1.0),
                SDDPhaseResult(phase="test", success=True, duration_seconds=45.0),
                SDDPhaseResult(phase="implement", success=True, duration_seconds=120.0),
                SDDPhaseResult(phase="refine", success=True, duration_seconds=30.0),
                SDDPhaseResult(phase="validate", success=True, duration_seconds=15.0),
            ],
            final_status=EvaluationResult.PASS,
            elicitation_metrics=ElicitationMetrics(
                discovery_rate=0.9,
                question_efficiency=0.15,
                total_questions=10,
                question_distribution={"clarification": 5, "evidence": 3, "assumption": 2},
                revealed_requirements=["REQ-001", "REQ-002"],
                hidden_requirements=["REQ-003"],
            ),
            total_duration_seconds=241.1,
        ),
        SDDBenchResult(
            instance_id="django__django-11088",
            degradation_level=SpecDegradationLevel.VAGUE,
            phase_results=[
                SDDPhaseResult(phase="degrade", success=True, duration_seconds=0.1),
                SDDPhaseResult(phase="elicit", success=True, duration_seconds=45.0),
                SDDPhaseResult(phase="parse", success=True, duration_seconds=1.5),
                SDDPhaseResult(phase="test", success=True, duration_seconds=50.0),
                SDDPhaseResult(phase="implement", success=False, duration_seconds=180.0, error="Tests failed"),
                SDDPhaseResult(phase="refine", success=False, duration_seconds=0.0, error="Skipped"),
                SDDPhaseResult(phase="validate", success=False, duration_seconds=0.0, error="Skipped"),
            ],
            final_status=EvaluationResult.FAIL,
            elicitation_metrics=ElicitationMetrics(
                discovery_rate=0.6,
                question_efficiency=0.1,
                total_questions=15,
                question_distribution={"clarification": 8, "evidence": 4, "implication": 3},
                revealed_requirements=["REQ-001"],
                hidden_requirements=["REQ-002", "REQ-003"],
            ),
            total_duration_seconds=276.6,
        ),
        SDDBenchResult(
            instance_id="flask__flask-4992",
            degradation_level=SpecDegradationLevel.MINIMAL,
            phase_results=[
                SDDPhaseResult(phase="degrade", success=True, duration_seconds=0.1),
                SDDPhaseResult(phase="elicit", success=True, duration_seconds=60.0),
                SDDPhaseResult(phase="parse", success=False, duration_seconds=2.0, error="Parse error"),
            ],
            final_status=EvaluationResult.ERROR,
            total_duration_seconds=62.1,
        ),
    ]


@pytest.fixture
def generator() -> SDDReportGenerator:
    """Create a generator instance."""
    return SDDReportGenerator()


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Generator[Path]:
    """Create a temporary output directory."""
    output_dir = tmp_path / "reports"
    output_dir.mkdir(parents=True)
    yield output_dir


class TestReportSummary:
    """Tests for ReportSummary dataclass."""

    def test_summary_fields(self):
        """Test that ReportSummary has all expected fields."""
        summary = ReportSummary(
            total_tasks=10,
            passed=7,
            failed=2,
            error=1,
            pass_rate=70.0,
            avg_duration=150.0,
            by_degradation={"full": 80.0, "vague": 60.0},
            by_phase={"degrade": 100.0, "elicit": 90.0},
        )

        assert summary.total_tasks == 10
        assert summary.passed == 7
        assert summary.failed == 2
        assert summary.error == 1
        assert summary.pass_rate == 70.0
        assert summary.avg_duration == 150.0
        assert summary.by_degradation == {"full": 80.0, "vague": 60.0}
        assert summary.by_phase == {"degrade": 100.0, "elicit": 90.0}


class TestSDDReportGenerator:
    """Tests for SDDReportGenerator class."""

    def test_compute_summary_empty_results(self, generator: SDDReportGenerator):
        """Test compute_summary with empty results."""
        summary = generator.compute_summary([])

        assert summary.total_tasks == 0
        assert summary.passed == 0
        assert summary.failed == 0
        assert summary.error == 0
        assert summary.pass_rate == 0.0
        assert summary.avg_duration == 0.0
        assert summary.by_degradation == {}
        assert summary.by_phase == {}

    def test_compute_summary_with_results(
        self, generator: SDDReportGenerator, sample_results: list[SDDBenchResult]
    ):
        """Test compute_summary calculates correct values."""
        summary = generator.compute_summary(sample_results)

        assert summary.total_tasks == 3
        assert summary.passed == 1
        assert summary.failed == 1
        assert summary.error == 1
        # Pass rate: 1/3 = 33.3%
        assert abs(summary.pass_rate - 33.33) < 0.1
        # Avg duration: (241.1 + 276.6 + 62.1) / 3 = 193.27
        assert abs(summary.avg_duration - 193.27) < 0.1

        # Check degradation levels present
        assert "full" in summary.by_degradation
        assert "vague" in summary.by_degradation
        assert "minimal" in summary.by_degradation
        # Full has 100% pass rate (1 task, 1 pass)
        assert summary.by_degradation["full"] == 100.0
        # Vague has 0% pass rate (1 task, 1 fail)
        assert summary.by_degradation["vague"] == 0.0
        # Minimal has 0% pass rate (1 task, 1 error)
        assert summary.by_degradation["minimal"] == 0.0

        # Check phases present
        assert "degrade" in summary.by_phase
        assert "elicit" in summary.by_phase
        assert "parse" in summary.by_phase

    def test_generate_markdown_report_empty(self, generator: SDDReportGenerator):
        """Test markdown generation with empty results."""
        md = generator.generate_markdown_report([])

        assert "# SDD-Bench Evaluation Report" in md
        assert "Total Tasks | 0" in md
        assert "Passed | 0" in md

    def test_generate_markdown_report_with_results(
        self, generator: SDDReportGenerator, sample_results: list[SDDBenchResult]
    ):
        """Test markdown generation with sample results."""
        md = generator.generate_markdown_report(sample_results)

        # Check header
        assert "# SDD-Bench Evaluation Report" in md
        assert "Generated" in md

        # Check summary section
        assert "## Summary" in md
        assert "Total Tasks | 3" in md
        assert "Passed | 1" in md
        assert "Failed | 1" in md

        # Check degradation section
        assert "## Results by Degradation Level" in md
        assert "full" in md
        assert "vague" in md
        assert "minimal" in md

        # Check phase section
        assert "## Phase Success Rates" in md
        assert "degrade" in md
        assert "elicit" in md

        # Check individual results section
        assert "## Individual Results" in md
        assert "django__django-11099" in md
        assert "django__django-11088" in md
        assert "flask__flask-4992" in md

    def test_generate_markdown_custom_title(
        self, generator: SDDReportGenerator, sample_results: list[SDDBenchResult]
    ):
        """Test markdown generation with custom title."""
        md = generator.generate_markdown_report(sample_results, title="My Custom Report")

        assert "# My Custom Report" in md

    def test_generate_html_report(
        self,
        generator: SDDReportGenerator,
        sample_results: list[SDDBenchResult],
        tmp_output_dir: Path,
    ):
        """Test HTML report generation."""
        output_path = tmp_output_dir / "report.html"
        generator.generate_html_report(sample_results, output_path)

        assert output_path.exists()
        html_content = output_path.read_text()

        # Check basic HTML structure
        assert "<!DOCTYPE html>" in html_content
        assert "<html lang=\"en\">" in html_content
        assert "</html>" in html_content

        # Check title and content
        assert "SDD-Bench Evaluation Report" in html_content

        # Check Chart.js inclusion
        assert "chart.js" in html_content

        # Check data embedding
        assert "django__django-11099" in html_content
        assert "django__django-11088" in html_content

        # Check filters are present (level, status, and phase)
        assert "levelFilter" in html_content
        assert "statusFilter" in html_content
        assert "phaseFilter" in html_content

        # Check phase filter options are present
        assert "Degrade Failed" in html_content
        assert "Elicit Failed" in html_content
        assert "Implement Failed" in html_content

    def test_generate_html_report_custom_title(
        self,
        generator: SDDReportGenerator,
        sample_results: list[SDDBenchResult],
        tmp_output_dir: Path,
    ):
        """Test HTML report with custom title."""
        output_path = tmp_output_dir / "custom_report.html"
        generator.generate_html_report(
            sample_results, output_path, title="Custom Dashboard"
        )

        html_content = output_path.read_text()
        assert "Custom Dashboard" in html_content

    def test_export_json(
        self,
        generator: SDDReportGenerator,
        sample_results: list[SDDBenchResult],
        tmp_output_dir: Path,
    ):
        """Test JSON export."""
        output_path = tmp_output_dir / "results.json"
        generator.export_json(sample_results, output_path)

        assert output_path.exists()
        data = json.loads(output_path.read_text())

        # Check structure
        assert "metadata" in data
        assert "summary" in data
        assert "results" in data

        # Check metadata
        assert "generated" in data["metadata"]
        assert data["metadata"]["total_tasks"] == 3

        # Check summary
        assert data["summary"]["passed"] == 1
        assert data["summary"]["failed"] == 1
        assert data["summary"]["error"] == 1
        assert "by_degradation" in data["summary"]
        assert "by_phase" in data["summary"]

        # Check results
        assert len(data["results"]) == 3
        result = data["results"][0]
        assert "instance_id" in result
        assert "degradation_level" in result
        assert "final_status" in result
        assert "total_duration_seconds" in result
        assert "phase_results" in result

    def test_export_csv(
        self,
        generator: SDDReportGenerator,
        sample_results: list[SDDBenchResult],
        tmp_output_dir: Path,
    ):
        """Test CSV export."""
        output_path = tmp_output_dir / "results.csv"
        generator.export_csv(sample_results, output_path)

        assert output_path.exists()
        csv_content = output_path.read_text()
        lines = csv_content.strip().split("\n")

        # Check header
        header = lines[0]
        assert "instance_id" in header
        assert "degradation_level" in header
        assert "final_status" in header
        assert "total_duration" in header
        assert "degrade_success" in header
        assert "elicit_success" in header

        # Check data rows
        assert len(lines) == 4  # Header + 3 results

        # Check first data row
        row1 = lines[1]
        assert "django__django-11099" in row1
        assert "full" in row1
        assert "pass" in row1

    def test_export_csv_escapes_special_characters(
        self, generator: SDDReportGenerator, tmp_output_dir: Path
    ):
        """Test CSV properly handles commas in fields."""
        results = [
            SDDBenchResult(
                instance_id="project,with,commas__issue-123",
                degradation_level=SpecDegradationLevel.FULL,
                phase_results=[
                    SDDPhaseResult(phase="degrade", success=True, duration_seconds=0.1),
                ],
                final_status=EvaluationResult.PASS,
                total_duration_seconds=10.0,
            ),
            SDDBenchResult(
                instance_id='project__with"quotes-456',
                degradation_level=SpecDegradationLevel.VAGUE,
                phase_results=[
                    SDDPhaseResult(phase="degrade", success=True, duration_seconds=0.2),
                ],
                final_status=EvaluationResult.FAIL,
                total_duration_seconds=20.0,
            ),
        ]

        output_path = tmp_output_dir / "csv_special.csv"
        generator.export_csv(results, output_path)

        assert output_path.exists()
        csv_content = output_path.read_text()

        # Read back with csv module to verify proper escaping
        import csv
        import io

        reader = csv.reader(io.StringIO(csv_content))
        rows = list(reader)

        # Should have header + 2 data rows
        assert len(rows) == 3

        # Verify instance_ids are correctly parsed (commas and quotes handled)
        assert rows[1][0] == "project,with,commas__issue-123"
        assert rows[2][0] == 'project__with"quotes-456'

    def test_unicode_in_instance_id(
        self, generator: SDDReportGenerator, tmp_output_dir: Path
    ):
        """Test non-ASCII characters in instance_id."""
        results = [
            SDDBenchResult(
                instance_id="projet__francais-cafe-123",
                degradation_level=SpecDegradationLevel.FULL,
                phase_results=[
                    SDDPhaseResult(phase="degrade", success=True, duration_seconds=0.1),
                ],
                final_status=EvaluationResult.PASS,
                total_duration_seconds=10.0,
            ),
            SDDBenchResult(
                instance_id="projekt__deutsch-strasse-456",
                degradation_level=SpecDegradationLevel.VAGUE,
                phase_results=[
                    SDDPhaseResult(phase="degrade", success=True, duration_seconds=0.2),
                ],
                final_status=EvaluationResult.FAIL,
                total_duration_seconds=20.0,
            ),
            SDDBenchResult(
                instance_id="project__nihongo-789",
                degradation_level=SpecDegradationLevel.MINIMAL,
                phase_results=[
                    SDDPhaseResult(phase="degrade", success=True, duration_seconds=0.3),
                ],
                final_status=EvaluationResult.ERROR,
                total_duration_seconds=30.0,
            ),
        ]

        # Test markdown output
        md = generator.generate_markdown_report(results)
        assert "projet__francais-cafe-123" in md
        assert "projekt__deutsch-strasse-456" in md
        assert "project__nihongo-789" in md

        # Test HTML output
        html_path = tmp_output_dir / "unicode_report.html"
        generator.generate_html_report(results, html_path)
        html_content = html_path.read_text(encoding="utf-8")
        assert "projet__francais-cafe-123" in html_content
        assert "projekt__deutsch-strasse-456" in html_content
        assert "project__nihongo-789" in html_content

        # Test JSON output
        json_path = tmp_output_dir / "unicode_results.json"
        generator.export_json(results, json_path)
        json_data = json.loads(json_path.read_text(encoding="utf-8"))
        instance_ids = [r["instance_id"] for r in json_data["results"]]
        assert "projet__francais-cafe-123" in instance_ids
        assert "projekt__deutsch-strasse-456" in instance_ids
        assert "project__nihongo-789" in instance_ids

        # Test CSV output
        csv_path = tmp_output_dir / "unicode_results.csv"
        generator.export_csv(results, csv_path)
        csv_content = csv_path.read_text(encoding="utf-8")
        assert "projet__francais-cafe-123" in csv_content
        assert "projekt__deutsch-strasse-456" in csv_content
        assert "project__nihongo-789" in csv_content


class TestComparisonReport:
    """Tests for comparison report generation."""

    @pytest.fixture
    def baseline_results(self) -> list[SDDBenchResult]:
        """Create baseline results."""
        return [
            SDDBenchResult(
                instance_id="django__django-11099",
                degradation_level=SpecDegradationLevel.FULL,
                phase_results=[
                    SDDPhaseResult(phase="degrade", success=True, duration_seconds=0.1),
                    SDDPhaseResult(phase="implement", success=True, duration_seconds=100.0),
                ],
                final_status=EvaluationResult.PASS,
                total_duration_seconds=100.1,
            ),
            SDDBenchResult(
                instance_id="flask__flask-4992",
                degradation_level=SpecDegradationLevel.FULL,
                phase_results=[
                    SDDPhaseResult(phase="degrade", success=True, duration_seconds=0.1),
                    SDDPhaseResult(phase="implement", success=False, duration_seconds=80.0),
                ],
                final_status=EvaluationResult.FAIL,
                total_duration_seconds=80.1,
            ),
        ]

    @pytest.fixture
    def experiment_results(self) -> list[SDDBenchResult]:
        """Create experiment results (improved)."""
        return [
            SDDBenchResult(
                instance_id="django__django-11099",
                degradation_level=SpecDegradationLevel.FULL,
                phase_results=[
                    SDDPhaseResult(phase="degrade", success=True, duration_seconds=0.1),
                    SDDPhaseResult(phase="elicit", success=True, duration_seconds=30.0),
                    SDDPhaseResult(phase="implement", success=True, duration_seconds=90.0),
                ],
                final_status=EvaluationResult.PASS,
                total_duration_seconds=120.1,
            ),
            SDDBenchResult(
                instance_id="flask__flask-4992",
                degradation_level=SpecDegradationLevel.FULL,
                phase_results=[
                    SDDPhaseResult(phase="degrade", success=True, duration_seconds=0.1),
                    SDDPhaseResult(phase="elicit", success=True, duration_seconds=25.0),
                    SDDPhaseResult(phase="implement", success=True, duration_seconds=70.0),
                ],
                final_status=EvaluationResult.PASS,
                total_duration_seconds=95.1,
            ),
        ]

    def test_comparison_report_structure(
        self,
        generator: SDDReportGenerator,
        baseline_results: list[SDDBenchResult],
        experiment_results: list[SDDBenchResult],
    ):
        """Test comparison report has correct structure."""
        md = generator.generate_comparison_report(
            baseline_results, experiment_results, "Passthrough", "ClaudeCode"
        )

        assert "# SDD-Bench Comparison Report" in md
        assert "## Overall Comparison" in md
        assert "Passthrough" in md
        assert "ClaudeCode" in md
        assert "Delta" in md

    def test_comparison_report_deltas(
        self,
        generator: SDDReportGenerator,
        baseline_results: list[SDDBenchResult],
        experiment_results: list[SDDBenchResult],
    ):
        """Test comparison report calculates deltas correctly."""
        md = generator.generate_comparison_report(
            baseline_results, experiment_results, "Baseline", "Experiment"
        )

        # Baseline: 1/2 = 50%, Experiment: 2/2 = 100%
        # Delta: +50%
        assert "Passed | 1 | 2 | +1" in md
        assert "100.0%" in md  # Experiment pass rate
        assert "+50.0%" in md  # Delta in pass rate

        # Validate the delta calculation is mathematically correct
        baseline_summary = generator.compute_summary(baseline_results)
        experiment_summary = generator.compute_summary(experiment_results)

        # Verify baseline: 1 pass out of 2 = 50%
        assert baseline_summary.passed == 1
        assert baseline_summary.total_tasks == 2
        assert baseline_summary.pass_rate == 50.0

        # Verify experiment: 2 passes out of 2 = 100%
        assert experiment_summary.passed == 2
        assert experiment_summary.total_tasks == 2
        assert experiment_summary.pass_rate == 100.0

        # Verify delta: 100% - 50% = +50%
        expected_delta = experiment_summary.pass_rate - baseline_summary.pass_rate
        assert expected_delta == 50.0

        # Verify delta appears correctly formatted in report
        assert f"+{expected_delta:.1f}%" in md

    def test_comparison_report_improvement_conclusion(
        self,
        generator: SDDReportGenerator,
        baseline_results: list[SDDBenchResult],
        experiment_results: list[SDDBenchResult],
    ):
        """Test comparison report shows improvement conclusion."""
        md = generator.generate_comparison_report(
            baseline_results, experiment_results, "Baseline", "Experiment"
        )

        assert "## Conclusion" in md
        assert "improvement" in md

    def test_comparison_report_degradation_comparison(
        self,
        generator: SDDReportGenerator,
        baseline_results: list[SDDBenchResult],
        experiment_results: list[SDDBenchResult],
    ):
        """Test comparison report includes degradation level comparison."""
        md = generator.generate_comparison_report(
            baseline_results, experiment_results, "Baseline", "Experiment"
        )

        assert "## Comparison by Degradation Level" in md
        assert "full" in md

    def test_comparison_report_phase_comparison(
        self,
        generator: SDDReportGenerator,
        baseline_results: list[SDDBenchResult],
        experiment_results: list[SDDBenchResult],
    ):
        """Test comparison report includes phase comparison."""
        md = generator.generate_comparison_report(
            baseline_results, experiment_results, "Baseline", "Experiment"
        )

        assert "## Comparison by Phase" in md
        assert "degrade" in md
        assert "implement" in md

    def test_comparison_report_empty_baseline(self, generator: SDDReportGenerator):
        """Test comparison with empty baseline."""
        baseline_results: list[SDDBenchResult] = []
        experiment_results = [
            SDDBenchResult(
                instance_id="django__django-11099",
                degradation_level=SpecDegradationLevel.FULL,
                phase_results=[
                    SDDPhaseResult(phase="degrade", success=True, duration_seconds=0.1),
                    SDDPhaseResult(phase="implement", success=True, duration_seconds=100.0),
                ],
                final_status=EvaluationResult.PASS,
                total_duration_seconds=100.1,
            ),
        ]

        md = generator.generate_comparison_report(
            baseline_results, experiment_results, "Baseline", "Experiment"
        )

        # Report should still be generated
        assert "# SDD-Bench Comparison Report" in md
        assert "## Overall Comparison" in md

        # Baseline should show zeros
        assert "| Total Tasks | 0 | 1 | - |" in md
        assert "| Passed | 0 | 1 | +1" in md

        # Should handle pass rate comparison with zero baseline
        # Baseline: 0%, Experiment: 100%
        assert "0.0%" in md  # Baseline pass rate
        assert "100.0%" in md  # Experiment pass rate

        # Delta should be +100%
        assert "+100.0%" in md

    def test_comparison_report_no_overlap(self, generator: SDDReportGenerator):
        """Test comparison when baseline and experiment have different tasks."""
        baseline_results = [
            SDDBenchResult(
                instance_id="django__django-11099",
                degradation_level=SpecDegradationLevel.FULL,
                phase_results=[
                    SDDPhaseResult(phase="degrade", success=True, duration_seconds=0.1),
                    SDDPhaseResult(phase="implement", success=True, duration_seconds=100.0),
                ],
                final_status=EvaluationResult.PASS,
                total_duration_seconds=100.1,
            ),
            SDDBenchResult(
                instance_id="django__django-11088",
                degradation_level=SpecDegradationLevel.VAGUE,
                phase_results=[
                    SDDPhaseResult(phase="degrade", success=True, duration_seconds=0.1),
                    SDDPhaseResult(phase="implement", success=False, duration_seconds=80.0),
                ],
                final_status=EvaluationResult.FAIL,
                total_duration_seconds=80.1,
            ),
        ]
        experiment_results = [
            SDDBenchResult(
                instance_id="flask__flask-4992",
                degradation_level=SpecDegradationLevel.MINIMAL,
                phase_results=[
                    SDDPhaseResult(phase="degrade", success=True, duration_seconds=0.1),
                    SDDPhaseResult(phase="test", success=True, duration_seconds=50.0),
                    SDDPhaseResult(phase="implement", success=True, duration_seconds=90.0),
                ],
                final_status=EvaluationResult.PASS,
                total_duration_seconds=140.1,
            ),
            SDDBenchResult(
                instance_id="requests__requests-5678",
                degradation_level=SpecDegradationLevel.MINIMAL,
                phase_results=[
                    SDDPhaseResult(phase="degrade", success=True, duration_seconds=0.1),
                    SDDPhaseResult(phase="test", success=True, duration_seconds=40.0),
                    SDDPhaseResult(phase="implement", success=True, duration_seconds=70.0),
                ],
                final_status=EvaluationResult.PASS,
                total_duration_seconds=110.1,
            ),
        ]

        md = generator.generate_comparison_report(
            baseline_results, experiment_results, "Baseline", "Experiment"
        )

        # Report should still be generated
        assert "# SDD-Bench Comparison Report" in md
        assert "## Overall Comparison" in md

        # Both should show their respective counts
        assert "| Total Tasks | 2 | 2 | - |" in md

        # Baseline: 1/2 = 50%, Experiment: 2/2 = 100%
        assert "| Passed | 1 | 2 | +1" in md

        # Degradation level comparison should include all levels from both
        assert "## Comparison by Degradation Level" in md
        assert "full" in md  # From baseline only
        assert "vague" in md  # From baseline only
        assert "minimal" in md  # From experiment only

        # Verify levels that only exist in one set show 0% for the other
        baseline_summary = generator.compute_summary(baseline_results)
        experiment_summary = generator.compute_summary(experiment_results)

        # Baseline has full (100%) and vague (0%), no minimal
        assert "full" in baseline_summary.by_degradation
        assert "vague" in baseline_summary.by_degradation
        assert "minimal" not in baseline_summary.by_degradation

        # Experiment has minimal (100%), no full or vague
        assert "minimal" in experiment_summary.by_degradation
        assert "full" not in experiment_summary.by_degradation
        assert "vague" not in experiment_summary.by_degradation

        # Phase comparison should include all phases from both
        assert "## Comparison by Phase" in md
        assert "degrade" in md
        assert "implement" in md
        assert "test" in md  # Only in experiment


class TestHTMLSecurity:
    """Tests for HTML output security."""

    def test_html_escapes_xss_in_instance_id(
        self, generator: SDDReportGenerator, tmp_output_dir: Path
    ):
        """Test that XSS in instance_id is escaped."""
        results = [
            SDDBenchResult(
                instance_id="<script>alert('xss')</script>",
                degradation_level=SpecDegradationLevel.FULL,
                phase_results=[],
                final_status=EvaluationResult.PASS,
                total_duration_seconds=0.0,
            )
        ]

        output_path = tmp_output_dir / "xss_test.html"
        generator.generate_html_report(results, output_path)

        html_content = output_path.read_text()

        # Script tag should be escaped
        assert "<script>alert" not in html_content
        assert "&lt;script&gt;" in html_content

    def test_html_escapes_xss_in_title(
        self, generator: SDDReportGenerator, tmp_output_dir: Path
    ):
        """Test that XSS in title is escaped."""
        results = []
        output_path = tmp_output_dir / "xss_title.html"

        generator.generate_html_report(
            results, output_path, title="<script>alert('xss')</script>"
        )

        html_content = output_path.read_text()
        assert "<script>alert" not in html_content
        assert "&lt;script&gt;" in html_content


class TestEdgeCases:
    """Tests for edge cases."""

    def test_many_results_truncated_in_markdown(self, generator: SDDReportGenerator):
        """Test that markdown truncates to 50 results."""
        results = [
            SDDBenchResult(
                instance_id=f"task-{i:03d}",
                degradation_level=SpecDegradationLevel.FULL,
                phase_results=[],
                final_status=EvaluationResult.PASS,
                total_duration_seconds=float(i),
            )
            for i in range(100)
        ]

        md = generator.generate_markdown_report(results)

        # Should mention truncation
        assert "...and 50 more results" in md
        # Should have first 50
        assert "task-000" in md
        assert "task-049" in md
        # Should not have task 50+
        assert "task-050" not in md

    def test_single_result(self, generator: SDDReportGenerator):
        """Test with single result."""
        results = [
            SDDBenchResult(
                instance_id="single-task",
                degradation_level=SpecDegradationLevel.FULL,
                phase_results=[],
                final_status=EvaluationResult.PASS,
                total_duration_seconds=100.0,
            )
        ]

        summary = generator.compute_summary(results)
        assert summary.total_tasks == 1
        assert summary.passed == 1
        assert summary.pass_rate == 100.0

        md = generator.generate_markdown_report(results)
        assert "single-task" in md
        assert "100.0%" in md

    def test_all_phases_missing(self, generator: SDDReportGenerator):
        """Test result with no phase results."""
        results = [
            SDDBenchResult(
                instance_id="no-phases",
                degradation_level=SpecDegradationLevel.FULL,
                phase_results=[],
                final_status=EvaluationResult.ERROR,
                total_duration_seconds=0.0,
            )
        ]

        summary = generator.compute_summary(results)
        assert summary.by_phase == {}

    def test_partial_status_counted(self, generator: SDDReportGenerator):
        """Test that partial status is neither pass nor fail."""
        results = [
            SDDBenchResult(
                instance_id="partial-task",
                degradation_level=SpecDegradationLevel.FULL,
                phase_results=[],
                final_status=EvaluationResult.PARTIAL,
                total_duration_seconds=50.0,
            )
        ]

        summary = generator.compute_summary(results)
        assert summary.passed == 0
        assert summary.failed == 0
        assert summary.error == 0
        # Pass rate only counts actual PASS
        assert summary.pass_rate == 0.0
