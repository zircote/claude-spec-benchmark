"""Reporting and dashboard module for SDD-Bench.

This module provides tools for generating reports and visualizations
of SDD evaluation results.

Example:
    >>> from claude_spec_benchmark.reporting import SDDReportGenerator
    >>> reporter = SDDReportGenerator()
    >>> reporter.generate_html_report(results, Path("report.html"))
"""

from claude_spec_benchmark.reporting.dashboard import (
    ReportSummary,
    SDDReportGenerator,
)

__all__: list[str] = [
    "ReportSummary",
    "SDDReportGenerator",
]
