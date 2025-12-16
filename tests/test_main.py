"""Tests for claude_spec_benchmark.main module."""

from __future__ import annotations

from typing import TYPE_CHECKING

from claude_spec_benchmark import __version__
from claude_spec_benchmark.main import main

if TYPE_CHECKING:
    from click.testing import CliRunner


def test_version() -> None:
    """Test that version is defined."""
    assert __version__ == "0.1.0"


def test_main_help(cli_runner: "CliRunner") -> None:
    """Test that main --help works."""
    result = cli_runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "SWE-bench" in result.output


def test_info_command(cli_runner: "CliRunner") -> None:
    """Test the info subcommand."""
    result = cli_runner.invoke(main, ["info"])
    assert result.exit_code == 0
    assert "claude-spec-benchmark" in result.output
