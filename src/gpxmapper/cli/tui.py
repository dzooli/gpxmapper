"""TUI (Terminal User Interface) command powered by Trogon and Textual."""

from __future__ import annotations

from typing import Any

import click
from textual.css.query import NoMatches
from textual.widgets import Select
from trogon.trogon import CommandBuilder, Trogon
from trogon.typer import init_tui
from trogon.widgets.command_info import CommandInfo
from trogon.widgets.parameter_controls import ParameterControls

from . import app


def _patch_trogon_command_info() -> None:
    """Patch upstream Trogon bug where action_show_command_info queries self instead of active screen."""

    def _action_show_command_info(self: Trogon) -> None:
        screen = self.screen
        if isinstance(screen, CommandBuilder):
            schema = getattr(screen, "selected_command_schema", None)
            if schema is None and hasattr(screen, "command_schemas"):
                schema = next(iter(screen.command_schemas.values()), None)
            if schema is not None:
                self.push_screen(CommandInfo(schema))
                return
        try:
            command_builder = self.query_one(CommandBuilder)
            self.push_screen(CommandInfo(command_builder.selected_command_schema))
        except NoMatches:
            pass

    Trogon.action_show_command_info = _action_show_command_info


def _patch_trogon_parameter_controls() -> None:
    """Patch Trogon's ParameterControls to handle boolean checkboxes and Enum/Choice defaults cleanly."""
    orig_compose = ParameterControls.compose
    orig_apply_default = ParameterControls._apply_default_value

    def _patched_compose(self: ParameterControls) -> Any:
        orig_type = self.schema.type
        if isinstance(orig_type, click.types.BoolParamType):
            self.schema.type = click.BOOL
            try:
                yield from orig_compose(self)
            finally:
                self.schema.type = orig_type
        else:
            yield from orig_compose(self)

    @staticmethod
    def _patched_apply_default_value(control_widget: Any, default_value: Any) -> None:
        if isinstance(control_widget, Select):
            val = getattr(default_value, "value", default_value)
            val_str = str(val) if val is not None else Select.BLANK
            control_widget.value = val_str
            control_widget.prompt = f"{val_str} (default)"
        else:
            orig_apply_default(control_widget, default_value)

    ParameterControls.compose = _patched_compose
    ParameterControls._apply_default_value = _patched_apply_default_value


_patch_trogon_command_info()
_patch_trogon_parameter_controls()

# Attach trogon interactive TUI to the Typer app as `gpxmapper tui`
init_tui(app, name="gpxmapper")
