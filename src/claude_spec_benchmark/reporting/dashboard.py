"""SDD-Bench report generation.

Generates reports in multiple formats for SDD-Bench evaluation results:
- Markdown reports for documentation
- HTML dashboards for interactive viewing
- JSON/CSV exports for data analysis
- Comparison reports for A/B testing frameworks

Example:
    >>> from claude_spec_benchmark.reporting import SDDReportGenerator
    >>> generator = SDDReportGenerator()
    >>> generator.generate_markdown_report(results)
"""

from __future__ import annotations

import csv
import html
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from claude_spec_benchmark.models import SDDBenchResult

# Module-level constant for SDD pipeline phase ordering
SDD_PHASE_ORDER: list[str] = [
    "degrade",
    "elicit",
    "parse",
    "test",
    "implement",
    "refine",
    "validate",
]


@dataclass
class ReportSummary:
    """Summary statistics for a set of results.

    Attributes:
        total_tasks: Total number of tasks evaluated.
        passed: Number of tasks that passed validation.
        failed: Number of tasks that failed.
        error: Number of tasks with errors.
        pass_rate: Percentage of passed tasks.
        avg_duration: Average total duration per task.
        by_degradation: Pass rates broken down by degradation level.
        by_phase: Success rates broken down by phase.
    """

    total_tasks: int
    passed: int
    failed: int
    error: int
    pass_rate: float
    avg_duration: float
    by_degradation: dict[str, float]
    by_phase: dict[str, float]


class SDDReportGenerator:
    """Generate reports from SDD-Bench evaluation results.

    Supports multiple output formats and comparison reports for
    evaluating different frameworks or configurations.

    Example:
        >>> generator = SDDReportGenerator()
        >>> md = generator.generate_markdown_report(results)
        >>> generator.generate_html_report(results, Path("report.html"))
        >>> generator.export_json(results, Path("results.json"))
    """

    def __init__(self) -> None:
        """Initialize the report generator."""
        self._timestamp = datetime.now().isoformat()

    def compute_summary(self, results: list[SDDBenchResult]) -> ReportSummary:
        """Compute summary statistics from results.

        Args:
            results: List of SDDBenchResult objects.

        Returns:
            ReportSummary with aggregated statistics.
        """
        if not results:
            return ReportSummary(
                total_tasks=0,
                passed=0,
                failed=0,
                error=0,
                pass_rate=0.0,
                avg_duration=0.0,
                by_degradation={},
                by_phase={},
            )

        total = len(results)
        passed = sum(1 for r in results if r.final_status.value == "pass")
        failed = sum(1 for r in results if r.final_status.value == "fail")
        error = sum(1 for r in results if r.final_status.value == "error")

        # Pass rate
        pass_rate = (passed / total * 100) if total > 0 else 0.0

        # Average duration
        durations = [r.total_duration_seconds for r in results]
        avg_duration = sum(durations) / len(durations) if durations else 0.0

        # By degradation level
        by_deg: dict[str, list[bool]] = defaultdict(list)
        for r in results:
            by_deg[r.degradation_level.value].append(r.final_status.value == "pass")

        by_degradation = {
            level: (sum(passes) / len(passes) * 100) if passes else 0.0
            for level, passes in by_deg.items()
        }

        # By phase success
        phase_success: dict[str, list[bool]] = defaultdict(list)
        for r in results:
            for phase_result in r.phase_results:
                phase_success[phase_result.phase].append(phase_result.success)

        by_phase = {
            phase: (sum(successes) / len(successes) * 100) if successes else 0.0
            for phase, successes in phase_success.items()
        }

        return ReportSummary(
            total_tasks=total,
            passed=passed,
            failed=failed,
            error=error,
            pass_rate=pass_rate,
            avg_duration=avg_duration,
            by_degradation=by_degradation,
            by_phase=by_phase,
        )

    def generate_markdown_report(
        self,
        results: list[SDDBenchResult],
        title: str = "SDD-Bench Evaluation Report",
    ) -> str:
        """Generate a Markdown report from results.

        Args:
            results: List of SDDBenchResult objects.
            title: Report title.

        Returns:
            Markdown-formatted report string.
        """
        summary = self.compute_summary(results)
        lines = [
            f"# {title}",
            "",
            f"**Generated**: {self._timestamp}",
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Tasks | {summary.total_tasks} |",
            f"| Passed | {summary.passed} |",
            f"| Failed | {summary.failed} |",
            f"| Errors | {summary.error} |",
            f"| **Pass Rate** | **{summary.pass_rate:.1f}%** |",
            f"| Avg Duration | {summary.avg_duration:.2f}s |",
            "",
        ]

        # By degradation level
        if summary.by_degradation:
            lines.extend(
                [
                    "## Results by Degradation Level",
                    "",
                    "| Level | Pass Rate |",
                    "|-------|-----------|",
                ]
            )
            for level, rate in sorted(summary.by_degradation.items()):
                lines.append(f"| {level} | {rate:.1f}% |")
            lines.append("")

        # By phase
        if summary.by_phase:
            lines.extend(
                [
                    "## Phase Success Rates",
                    "",
                    "| Phase | Success Rate |",
                    "|-------|--------------|",
                ]
            )
            for phase in SDD_PHASE_ORDER:
                if phase in summary.by_phase:
                    lines.append(f"| {phase} | {summary.by_phase[phase]:.1f}% |")
            lines.append("")

        # Individual results table
        if results:
            lines.extend(
                [
                    "## Individual Results",
                    "",
                    "| Instance ID | Degradation | Status | Duration |",
                    "|-------------|-------------|--------|----------|",
                ]
            )
            for r in results[:50]:  # Limit to 50 for readability
                status_emoji = "PASS" if r.final_status.value == "pass" else "FAIL"
                lines.append(
                    f"| {r.instance_id} | {r.degradation_level.value} | "
                    f"{status_emoji} | {r.total_duration_seconds:.1f}s |"
                )

            if len(results) > 50:
                lines.append(f"\n*...and {len(results) - 50} more results*")

            lines.append("")

        return "\n".join(lines)

    def generate_html_report(
        self,
        results: list[SDDBenchResult],
        output_path: Path,
        title: str = "SDD-Bench Evaluation Report",
    ) -> None:
        """Generate an interactive HTML dashboard.

        Args:
            results: List of SDDBenchResult objects.
            output_path: Path to write HTML file.
            title: Report title.
        """
        summary = self.compute_summary(results)

        # Build results JSON for JavaScript - escape for safety
        results_data = [
            {
                "instance_id": html.escape(r.instance_id),
                "degradation_level": html.escape(r.degradation_level.value),
                "final_status": html.escape(r.final_status.value),
                "duration": r.total_duration_seconds,
                "phases": [
                    {"phase": html.escape(p.phase), "success": p.success, "duration": p.duration_seconds}
                    for p in r.phase_results
                ],
            }
            for r in results
        ]
        results_json = json.dumps(results_data)

        # Build degradation data for chart
        deg_labels = json.dumps([html.escape(k) for k in summary.by_degradation])
        deg_values = json.dumps(list(summary.by_degradation.values()))

        # Build phase data for chart
        phase_labels = json.dumps([p for p in SDD_PHASE_ORDER if p in summary.by_phase])
        phase_values = json.dumps([summary.by_phase.get(p, 0) for p in SDD_PHASE_ORDER if p in summary.by_phase])

        # Escape title for HTML
        safe_title = html.escape(title)
        safe_timestamp = html.escape(self._timestamp)

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{safe_title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg-color: #f5f5f5;
            --card-bg: #ffffff;
            --text-color: #333;
            --border-color: #ddd;
            --success-color: #4caf50;
            --fail-color: #f44336;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
            padding: 20px;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ margin-bottom: 10px; }}
        .timestamp {{ color: #666; margin-bottom: 20px; }}
        .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .card {{
            background: var(--card-bg);
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .card h3 {{ font-size: 14px; color: #666; margin-bottom: 10px; }}
        .card .value {{ font-size: 32px; font-weight: bold; }}
        .card .value.success {{ color: var(--success-color); }}
        .card .value.fail {{ color: var(--fail-color); }}
        .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px; }}
        .chart-container {{
            background: var(--card-bg);
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .chart-container h3 {{ margin-bottom: 15px; }}
        table {{ width: 100%; border-collapse: collapse; background: var(--card-bg); border-radius: 8px; overflow: hidden; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid var(--border-color); }}
        th {{ background: #f0f0f0; font-weight: 600; }}
        tr:hover {{ background: #f9f9f9; }}
        .status-pass {{ color: var(--success-color); }}
        .status-fail {{ color: var(--fail-color); }}
        .filters {{ margin-bottom: 20px; }}
        .filters select {{ padding: 8px; margin-right: 10px; border-radius: 4px; border: 1px solid var(--border-color); }}
        @media (max-width: 768px) {{
            .charts {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{safe_title}</h1>
        <p class="timestamp">Generated: {safe_timestamp}</p>

        <div class="cards">
            <div class="card">
                <h3>Total Tasks</h3>
                <div class="value">{summary.total_tasks}</div>
            </div>
            <div class="card">
                <h3>Passed</h3>
                <div class="value success">{summary.passed}</div>
            </div>
            <div class="card">
                <h3>Failed</h3>
                <div class="value fail">{summary.failed}</div>
            </div>
            <div class="card">
                <h3>Pass Rate</h3>
                <div class="value">{summary.pass_rate:.1f}%</div>
            </div>
            <div class="card">
                <h3>Avg Duration</h3>
                <div class="value">{summary.avg_duration:.1f}s</div>
            </div>
        </div>

        <div class="charts">
            <div class="chart-container">
                <h3>Pass Rate by Degradation Level</h3>
                <canvas id="degradationChart"></canvas>
            </div>
            <div class="chart-container">
                <h3>Phase Success Rates</h3>
                <canvas id="phaseChart"></canvas>
            </div>
        </div>

        <h2>Results</h2>
        <div class="filters">
            <select id="levelFilter">
                <option value="">All Levels</option>
            </select>
            <select id="statusFilter">
                <option value="">All Statuses</option>
                <option value="pass">Pass</option>
                <option value="fail">Fail</option>
                <option value="error">Error</option>
            </select>
            <select id="phaseFilter">
                <option value="">All Phases</option>
                <option value="degrade">Degrade Failed</option>
                <option value="elicit">Elicit Failed</option>
                <option value="parse">Parse Failed</option>
                <option value="test">Test Failed</option>
                <option value="implement">Implement Failed</option>
                <option value="refine">Refine Failed</option>
                <option value="validate">Validate Failed</option>
            </select>
        </div>
        <table id="resultsTable">
            <thead>
                <tr>
                    <th>Instance ID</th>
                    <th>Degradation</th>
                    <th>Status</th>
                    <th>Duration</th>
                </tr>
            </thead>
            <tbody id="resultsBody"></tbody>
        </table>
    </div>

    <script>
        const results = {results_json};
        const degLabels = {deg_labels};
        const degValues = {deg_values};
        const phaseLabels = {phase_labels};
        const phaseValues = {phase_values};

        // Degradation chart
        new Chart(document.getElementById('degradationChart'), {{
            type: 'bar',
            data: {{
                labels: degLabels,
                datasets: [{{
                    label: 'Pass Rate (%)',
                    data: degValues,
                    backgroundColor: 'rgba(76, 175, 80, 0.7)',
                    borderColor: 'rgba(76, 175, 80, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                scales: {{ y: {{ beginAtZero: true, max: 100 }} }},
                plugins: {{ legend: {{ display: false }} }}
            }}
        }});

        // Phase chart
        new Chart(document.getElementById('phaseChart'), {{
            type: 'bar',
            data: {{
                labels: phaseLabels,
                datasets: [{{
                    label: 'Success Rate (%)',
                    data: phaseValues,
                    backgroundColor: 'rgba(33, 150, 243, 0.7)',
                    borderColor: 'rgba(33, 150, 243, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                scales: {{ y: {{ beginAtZero: true, max: 100 }} }},
                plugins: {{ legend: {{ display: false }} }}
            }}
        }});

        // Populate level filter using safe DOM methods
        const levels = [...new Set(results.map(r => r.degradation_level))];
        const levelSelect = document.getElementById('levelFilter');
        levels.forEach(level => {{
            const opt = document.createElement('option');
            opt.value = level;
            opt.textContent = level;
            levelSelect.appendChild(opt);
        }});

        // Render results table using safe DOM methods (no innerHTML)
        function filterResults() {{
            const levelFilter = document.getElementById('levelFilter').value;
            const statusFilter = document.getElementById('statusFilter').value;
            const phaseFilter = document.getElementById('phaseFilter').value;

            const filtered = results.filter(r => {{
                if (levelFilter && r.degradation_level !== levelFilter) return false;
                if (statusFilter && r.final_status !== statusFilter) return false;
                if (phaseFilter) {{
                    // Check if the specified phase failed
                    const phaseResult = r.phases.find(p => p.phase === phaseFilter);
                    if (!phaseResult || phaseResult.success) return false;
                }}
                return true;
            }});

            const tbody = document.getElementById('resultsBody');
            // Clear existing rows safely
            while (tbody.firstChild) {{
                tbody.removeChild(tbody.firstChild);
            }}

            // Build new rows using safe DOM methods
            filtered.forEach(r => {{
                const tr = document.createElement('tr');

                const tdId = document.createElement('td');
                tdId.textContent = r.instance_id;
                tr.appendChild(tdId);

                const tdLevel = document.createElement('td');
                tdLevel.textContent = r.degradation_level;
                tr.appendChild(tdLevel);

                const tdStatus = document.createElement('td');
                tdStatus.textContent = r.final_status;
                tdStatus.className = 'status-' + r.final_status;
                tr.appendChild(tdStatus);

                const tdDuration = document.createElement('td');
                tdDuration.textContent = r.duration.toFixed(1) + 's';
                tr.appendChild(tdDuration);

                tbody.appendChild(tr);
            }});
        }}

        // Add event listeners
        document.getElementById('levelFilter').addEventListener('change', filterResults);
        document.getElementById('statusFilter').addEventListener('change', filterResults);
        document.getElementById('phaseFilter').addEventListener('change', filterResults);

        // Initial render
        filterResults();
    </script>
</body>
</html>"""

        output_path.write_text(html_content)

    def export_json(
        self,
        results: list[SDDBenchResult],
        output_path: Path,
    ) -> None:
        """Export results to JSON format.

        Args:
            results: List of SDDBenchResult objects.
            output_path: Path to write JSON file.
        """
        summary = self.compute_summary(results)

        data = {
            "metadata": {
                "generated": self._timestamp,
                "total_tasks": summary.total_tasks,
            },
            "summary": {
                "passed": summary.passed,
                "failed": summary.failed,
                "error": summary.error,
                "pass_rate": summary.pass_rate,
                "avg_duration": summary.avg_duration,
                "by_degradation": summary.by_degradation,
                "by_phase": summary.by_phase,
            },
            "results": [
                {
                    "instance_id": r.instance_id,
                    "degradation_level": r.degradation_level.value,
                    "final_status": r.final_status.value,
                    "total_duration_seconds": r.total_duration_seconds,
                    "phase_results": [
                        {
                            "phase": p.phase,
                            "success": p.success,
                            "duration_seconds": p.duration_seconds,
                            "error": p.error,
                        }
                        for p in r.phase_results
                    ],
                }
                for r in results
            ],
        }

        with output_path.open("w") as f:
            json.dump(data, f, indent=2)

    def export_csv(
        self,
        results: list[SDDBenchResult],
        output_path: Path,
    ) -> None:
        """Export results to CSV format.

        Args:
            results: List of SDDBenchResult objects.
            output_path: Path to write CSV file.
        """
        with output_path.open("w", newline="") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(
                [
                    "instance_id",
                    "degradation_level",
                    "final_status",
                    "total_duration",
                    "degrade_success",
                    "elicit_success",
                    "parse_success",
                    "test_success",
                    "implement_success",
                    "refine_success",
                    "validate_success",
                ]
            )

            # Data rows
            for r in results:
                phase_map = {p.phase: p.success for p in r.phase_results}
                writer.writerow(
                    [
                        r.instance_id,
                        r.degradation_level.value,
                        r.final_status.value,
                        f"{r.total_duration_seconds:.2f}",
                        phase_map.get("degrade", ""),
                        phase_map.get("elicit", ""),
                        phase_map.get("parse", ""),
                        phase_map.get("test", ""),
                        phase_map.get("implement", ""),
                        phase_map.get("refine", ""),
                        phase_map.get("validate", ""),
                    ]
                )

    def generate_comparison_report(
        self,
        baseline_results: list[SDDBenchResult],
        experiment_results: list[SDDBenchResult],
        baseline_name: str = "Baseline",
        experiment_name: str = "Experiment",
    ) -> str:
        """Generate a comparison report between two result sets.

        Args:
            baseline_results: Baseline framework results.
            experiment_results: Experiment framework results.
            baseline_name: Name for baseline in report.
            experiment_name: Name for experiment in report.

        Returns:
            Markdown-formatted comparison report.
        """
        baseline_summary = self.compute_summary(baseline_results)
        experiment_summary = self.compute_summary(experiment_results)

        def delta(exp: float, base: float) -> str:
            diff = exp - base
            sign = "+" if diff >= 0 else ""
            return f"{sign}{diff:.1f}"

        lines = [
            "# SDD-Bench Comparison Report",
            "",
            f"**Generated**: {self._timestamp}",
            "",
            "## Overall Comparison",
            "",
            f"| Metric | {baseline_name} | {experiment_name} | Delta |",
            "|--------|---------|------------|-------|",
            f"| Total Tasks | {baseline_summary.total_tasks} | {experiment_summary.total_tasks} | - |",
            f"| Passed | {baseline_summary.passed} | {experiment_summary.passed} | {delta(experiment_summary.passed, baseline_summary.passed)} |",
            f"| Failed | {baseline_summary.failed} | {experiment_summary.failed} | {delta(experiment_summary.failed, baseline_summary.failed)} |",
            f"| **Pass Rate** | **{baseline_summary.pass_rate:.1f}%** | **{experiment_summary.pass_rate:.1f}%** | **{delta(experiment_summary.pass_rate, baseline_summary.pass_rate)}%** |",
            f"| Avg Duration | {baseline_summary.avg_duration:.1f}s | {experiment_summary.avg_duration:.1f}s | {delta(experiment_summary.avg_duration, baseline_summary.avg_duration)}s |",
            "",
        ]

        # By degradation level comparison
        all_levels = set(baseline_summary.by_degradation.keys()) | set(
            experiment_summary.by_degradation.keys()
        )
        if all_levels:
            lines.extend(
                [
                    "## Comparison by Degradation Level",
                    "",
                    f"| Level | {baseline_name} | {experiment_name} | Delta |",
                    "|-------|---------|------------|-------|",
                ]
            )
            for level in sorted(all_levels):
                base_rate = baseline_summary.by_degradation.get(level, 0)
                exp_rate = experiment_summary.by_degradation.get(level, 0)
                lines.append(
                    f"| {level} | {base_rate:.1f}% | {exp_rate:.1f}% | {delta(exp_rate, base_rate)}% |"
                )
            lines.append("")

        # By phase comparison
        all_phases = set(baseline_summary.by_phase.keys()) | set(
            experiment_summary.by_phase.keys()
        )
        if all_phases:
            lines.extend(
                [
                    "## Comparison by Phase",
                    "",
                    f"| Phase | {baseline_name} | {experiment_name} | Delta |",
                    "|-------|---------|------------|-------|",
                ]
            )
            for phase in SDD_PHASE_ORDER:
                if phase in all_phases:
                    base_rate = baseline_summary.by_phase.get(phase, 0)
                    exp_rate = experiment_summary.by_phase.get(phase, 0)
                    lines.append(
                        f"| {phase} | {base_rate:.1f}% | {exp_rate:.1f}% | {delta(exp_rate, base_rate)}% |"
                    )
            lines.append("")

        # Summary
        pass_rate_delta = experiment_summary.pass_rate - baseline_summary.pass_rate
        if pass_rate_delta > 0:
            conclusion = f"The {experiment_name} shows a **{pass_rate_delta:.1f}%** improvement in pass rate."
        elif pass_rate_delta < 0:
            conclusion = f"The {experiment_name} shows a **{abs(pass_rate_delta):.1f}%** decrease in pass rate."
        else:
            conclusion = "Both configurations show equivalent performance."

        lines.extend(
            [
                "## Conclusion",
                "",
                conclusion,
                "",
            ]
        )

        return "\n".join(lines)
