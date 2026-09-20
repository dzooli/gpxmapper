"""Unit tests for GPXMapper TUI command."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import typer.main
from trogon.trogon import Trogon
from trogon.widgets.command_info import CommandInfo
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


@pytest.mark.asyncio
async def test_trogon_action_show_command_info_does_not_crash() -> None:
    """Test that action_show_command_info pushes CommandInfo screen without NoMatches error."""
    group = typer.main.get_group(app)
    trogon_app = Trogon(group, app_name="gpxmapper")
    async with trogon_app.run_test() as pilot:
        await pilot.pause()
        trogon_app.action_show_command_info()
        await pilot.pause()
        assert isinstance(trogon_app.screen, CommandInfo)
