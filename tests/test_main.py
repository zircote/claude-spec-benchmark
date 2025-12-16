"""Tests for claude_spec_benchmark.main module."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from claude_spec_benchmark import __version__
from claude_spec_benchmark.main import main

if TYPE_CHECKING:
    from click.testing import CliRunner


def test_version() -> None:
    """Test that version is defined."""
    assert __version__ == "0.1.0"


def test_main_help(cli_runner: CliRunner) -> None:
    """Test that main --help works."""
    result = cli_runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "SWE-bench" in result.output


def test_info_command(cli_runner: CliRunner) -> None:
    """Test the info subcommand."""
    result = cli_runner.invoke(main, ["info"])
    assert result.exit_code == 0
    assert "claude-spec-benchmark" in result.output


# SDD-Bench CLI Tests


def test_sdd_help(cli_runner: CliRunner) -> None:
    """Test sdd command group help."""
    result = cli_runner.invoke(main, ["sdd", "--help"])
    assert result.exit_code == 0
    assert "SDD-Bench" in result.output
    assert "degrade" in result.output


def test_sdd_degrade_help(cli_runner: CliRunner) -> None:
    """Test sdd degrade command help."""
    result = cli_runner.invoke(main, ["sdd", "degrade", "--help"])
    assert result.exit_code == 0
    assert "Degrade a specification" in result.output
    assert "--level" in result.output
    assert "--seed" in result.output


def test_sdd_degrade_command(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test sdd degrade command execution."""
    # Create test issue file
    issue_file = tmp_path / "issue.txt"
    issue_file.write_text(
        "Bug in django/views.py line 42:\n```python\nraise Error()\n```"
    )

    result = cli_runner.invoke(
        main, ["sdd", "degrade", str(issue_file), "--level", "partial", "--seed", "42"]
    )
    assert result.exit_code == 0
    # Code block should be removed in partial mode
    assert "```python" not in result.output


def test_sdd_degrade_json_output(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test sdd degrade with JSON output."""
    import json

    issue_file = tmp_path / "issue.txt"
    issue_file.write_text("Error in auth.py at line 10")

    result = cli_runner.invoke(
        main,
        [
            "sdd",
            "degrade",
            str(issue_file),
            "--level",
            "minimal",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "degraded_text" in data
    assert data["level"] == "minimal"
