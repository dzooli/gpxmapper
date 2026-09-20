"""TUI (Terminal User Interface) command powered by Trogon and Textual."""

from __future__ import annotations

from textual.css.query import NoMatches
from trogon.trogon import CommandBuilder, Trogon
from trogon.typer import init_tui
from trogon.widgets.command_info import CommandInfo

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


_patch_trogon_command_info()

# Attach trogon interactive TUI to the Typer app as `gpxmapper tui`
init_tui(app, name="gpxmapper")
