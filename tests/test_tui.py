"""Unit tests for GPXMapper TUI command."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import typer.main
from textual.widgets import Checkbox, Select
from trogon.trogon import Trogon
from trogon.widgets.command_info import CommandInfo
from trogon.widgets.command_tree import CommandTree
from trogon.widgets.parameter_controls import ParameterControls
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


@pytest.mark.asyncio
async def test_trogon_renders_checkboxes_and_select_controls() -> None:
    """Test that boolean options render as Checkbox and enum options render as Select dropdowns."""
    group = typer.main.get_group(app)
    trogon_app = Trogon(group, app_name="gpxmapper")
    async with trogon_app.run_test() as pilot:
        await pilot.pause()
        tree = trogon_app.query_one(CommandTree)
        root_group = tree.root.children[0]
        gen_node = [n for n in root_group.children if "generate" in n.label.plain][0]
        await trogon_app.screen._refresh_command_form(gen_node)
        await pilot.pause()

        controls = list(trogon_app.query(ParameterControls))
        controls_by_name = {
            tuple(c.schema.name) if isinstance(c.schema.name, list) else (c.schema.name,): c for c in controls
        }

        # Verify boolean options render as Checkbox
        no_ts_ctrl = [c for names, c in controls_by_name.items() if any("--no-timestamp" in n for n in names)][0]
        assert no_ts_ctrl.query(Checkbox)
        assert not no_ts_ctrl.query("Input")

        geolocate_ctrl = [c for names, c in controls_by_name.items() if any("--geolocate" in n for n in names)][0]
        assert geolocate_ctrl.query(Checkbox)

        # Verify text_align renders as Select dropdown
        text_align_ctrl = [c for names, c in controls_by_name.items() if any("--text-align" in n for n in names)][0]
        selects = list(text_align_ctrl.query(Select))
        assert len(selects) == 1
        assert selects[0].value == "left"
