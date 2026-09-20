"""Unit tests for GPXMapper TUI command."""

from __future__ import annotations

from unittest.mock import patch

from typer.testing import CliRunner

from gpxmapper.cli import app

runner = CliRunner()


def test_tui_command_in_cli_help() -> None:
    """Test that tui command is listed in main CLI help."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "tui" in result.output


def test_tui_help() -> None:
    """Test that gpxmapper tui --help returns 0."""
    result = runner.invoke(app, ["tui", "--help"])
    assert result.exit_code == 0
    assert "Open Textual TUI" in result.output or "tui" in result.output


def test_tui_invocation_runs_trogon() -> None:
    """Test that invoking tui command calls Trogon.run."""
    with patch("trogon.typer.Trogon.run") as mock_run:
        result = runner.invoke(app, ["tui"])
        assert result.exit_code == 0
        mock_run.assert_called_once()
