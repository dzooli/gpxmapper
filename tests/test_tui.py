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

        # Switch to root group to verify --log-level renders as Select dropdown
        await trogon_app.screen._refresh_command_form(root_group)
        await pilot.pause()
        root_controls = list(trogon_app.query(ParameterControls))
        log_level_ctrl = [
            c
            for c in root_controls
            if any("--log-level" in n for n in (c.schema.name if isinstance(c.schema.name, list) else [c.schema.name]))
        ][0]
        log_selects = list(log_level_ctrl.query(Select))
        assert len(log_selects) == 1
        assert log_selects[0].value == "INFO"


@pytest.mark.asyncio
async def test_trogon_renders_range_sliders_for_bounded_numbers() -> None:
    """Test that bounded numerical options render as RangeSlider controls."""
    from gpxmapper.cli.widgets import RangeSlider

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

        # Zoom slider: min 1, max 19, default 15
        zoom_ctrl = [c for names, c in controls_by_name.items() if any("--zoom" in n for n in names)][0]
        zoom_sliders = list(zoom_ctrl.query(RangeSlider))
        assert len(zoom_sliders) == 1
        assert zoom_sliders[0].min_val == 1
        assert zoom_sliders[0].max_val == 19
        assert zoom_sliders[0].value == 15.0
        assert zoom_sliders[0].value_str == "15"

        # FPS slider: min 1, max 60, default 30
        fps_ctrl = [c for names, c in controls_by_name.items() if any("--fps" in n for n in names)][0]
        fps_sliders = list(fps_ctrl.query(RangeSlider))
        assert len(fps_sliders) == 1
        assert fps_sliders[0].min_val == 1
        assert fps_sliders[0].max_val == 60
        assert fps_sliders[0].value == 30.0
        assert fps_sliders[0].value_str == "30"

        # Font scale slider: min 0.1, max 5.0, default 0.7, is_float True
        font_scale_ctrl = [c for names, c in controls_by_name.items() if any("--font-scale" in n for n in names)][0]
        font_sliders = list(font_scale_ctrl.query(RangeSlider))
        assert len(font_sliders) == 1
        assert font_sliders[0].min_val == 0.1
        assert font_sliders[0].max_val == 5.0
        assert font_sliders[0].value == 0.7
        assert font_sliders[0].is_float is True
        assert font_sliders[0].value_str == "0.7"


@pytest.mark.asyncio
async def test_range_slider_widget_interaction() -> None:
    """Test interactive behavior of RangeSlider (inc/dec buttons and key events)."""
    from textual.app import App, ComposeResult
    from textual.widgets import Button

    from gpxmapper.cli.widgets import RangeSlider

    class SliderApp(App[None]):
        def compose(self) -> ComposeResult:
            yield RangeSlider(min_val=1, max_val=10, default_val=5)
            yield RangeSlider(min_val=0.0, max_val=2.0, default_val=1.0, is_float=True)

    test_app = SliderApp()
    async with test_app.run_test() as pilot:
        await pilot.pause()
        sliders = list(test_app.query(RangeSlider))
        int_slider = sliders[0]
        float_slider = sliders[1]

        # Test decrement button on int_slider
        btn_dec = int_slider.query_one(".btn-dec", Button)
        btn_inc = int_slider.query_one(".btn-inc", Button)

        btn_dec.press()
        await pilot.pause()
        assert int_slider.value == 4.0
        assert int_slider.value_str == "4"

        btn_inc.press()
        btn_inc.press()
        await pilot.pause()
        assert int_slider.value == 6.0
        assert int_slider.value_str == "6"

        # Test float slider stepping
        float_dec = float_slider.query_one(".btn-dec", Button)
        float_dec.press()
        await pilot.pause()
        assert float_slider.value == 0.9
        assert float_slider.value_str == "0.9"

        # Test keyboard navigation
        int_slider.focus()
        await pilot.press("left")
        assert int_slider.value == 5.0
        await pilot.press("right")
        assert int_slider.value == 6.0


@pytest.mark.asyncio
async def test_trogon_generate_form_grouped_categories() -> None:
    """Test that the generate command form groups parameters into distinct category sections."""
    from trogon.widgets.form import CommandForm

    group = typer.main.get_group(app)
    trogon_app = Trogon(group, app_name="gpxmapper")
    async with trogon_app.run_test() as pilot:
        await pilot.pause()
        tree = trogon_app.query_one(CommandTree)
        root_group = tree.root.children[0]
        gen_node = [n for n in root_group.children if "generate" in n.label.plain][0]
        await trogon_app.screen._refresh_command_form(gen_node)
        await pilot.pause()

        form = trogon_app.query_one(CommandForm)
        group_headers = [hdr.render().plain for hdr in form.query(".command-form-group-header")]

        assert "📁 Output & File Options" in group_headers
        assert "⏱ Video Dimensions & Timing" in group_headers
        assert "🗺 Map & Marker Styling" in group_headers
        assert "🔤 Typography & Text Overlay" in group_headers
        assert "💬 Captions & Geolocation" in group_headers
