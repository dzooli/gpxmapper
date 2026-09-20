"""Custom Textual and Trogon widgets for GPXMapper interactive TUI."""

from __future__ import annotations

from typing import Any, Callable

import click
from rich.text import Text
from textual import events, on
from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Button, Input, Select, Static
from trogon.introspect import ArgumentSchema, MultiValueParamData, OptionSchema
from trogon.widgets.parameter_controls import ControlWidgetType, ParameterControls


class RangeSlider(Widget):
    """Interactive Range Slider widget for bounded numerical parameters in Trogon."""

    can_focus = True

    DEFAULT_CSS = """
    RangeSlider {
        height: 1;
        layout: horizontal;
        padding: 0;
        margin: 0;
    }
    RangeSlider Button {
        min-width: 3;
        width: 3;
        height: 1;
        min-height: 1;
        max-height: 1;
        padding: 0;
        margin: 0;
        border: none !important;
        border-top: none !important;
        border-bottom: none !important;
        border-left: none !important;
        border-right: none !important;
        background: $surface-lighten-2;
        color: $text;
        text-style: bold;
    }
    RangeSlider Button:hover {
        border: none !important;
        border-top: none !important;
        border-bottom: none !important;
        border-left: none !important;
        border-right: none !important;
        background: $accent-darken-1;
        color: $background;
    }
    RangeSlider Button:focus {
        border: none !important;
        border-top: none !important;
        border-bottom: none !important;
        border-left: none !important;
        border-right: none !important;
        background: $accent;
        color: $background;
    }
    .slider-track {
        width: 28;
        height: 1;
        margin: 0 1;
        content-align: center middle;
    }
    .slider-val {
        width: 6;
        height: 1;
        text-align: right;
        color: $accent;
        text-style: bold;
        margin-right: 1;
    }
    .slider-bounds {
        height: 1;
        color: $text-muted;
    }
    """

    value: reactive[float] = reactive(0.0)

    def __init__(
        self,
        min_val: float,
        max_val: float,
        default_val: float | None = None,
        is_float: bool = False,
        classes: str = "",
    ) -> None:
        super().__init__(classes=classes)
        self.min_val = min_val
        self.max_val = max_val
        self.is_float = is_float
        self.step = 0.1 if is_float else 1.0
        self.value = float(default_val) if default_val is not None else float(min_val)

    @property
    def value_str(self) -> str:
        """Formatted string representation for CLI arguments."""
        if self.is_float:
            return f"{self.value:.1f}"
        return f"{int(self.value)}"

    @property
    def bounds_str(self) -> str:
        """Formatted string representation of min/max bounds."""
        min_s = f"{self.min_val:.1f}" if self.is_float else f"{int(self.min_val)}"
        max_s = f"{self.max_val:.1f}" if self.is_float else f"{int(self.max_val)}"
        return f"[{min_s}..{max_s}]"

    def compose(self) -> ComposeResult:
        yield Button("<", classes="slider-btn btn-dec")
        yield Static(self._render_bar(28), classes="slider-track", id="track")
        yield Button(">", classes="slider-btn btn-inc")
        yield Static(self.value_str, classes="slider-val", id="val")
        yield Static(self.bounds_str, classes="slider-bounds", id="bounds")

    def _render_bar(self, width: int = 28) -> Text:
        span = max(1e-9, (self.max_val - self.min_val))
        ratio = (self.value - self.min_val) / span
        ratio = max(0.0, min(1.0, ratio))
        pos = int(ratio * (width - 1))
        t = Text()
        if pos > 0:
            t.append("━" * pos, style="bold cyan")
        t.append("●", style="bold yellow")
        if width - 1 - pos > 0:
            t.append("─" * (width - 1 - pos), style="dim white")
        return t

    def watch_value(self, new_val: float) -> None:
        try:
            track = self.query_one("#track", Static)
            track.update(self._render_bar(28))
            val_lbl = self.query_one("#val", Static)
            val_lbl.update(self.value_str)
        except Exception:
            pass
        # Post Input.Changed message so the Trogon form automatically updates the preview
        fake_input = Input(value=self.value_str)
        self.post_message(Input.Changed(fake_input, self.value_str))

    @on(Button.Pressed, ".btn-dec")
    def dec(self, event: Button.Pressed | None = None) -> None:
        if event is not None:
            event.stop()
        self.value = max(self.min_val, round(self.value - self.step, 2 if self.is_float else 0))

    @on(Button.Pressed, ".btn-inc")
    def inc(self, event: Button.Pressed | None = None) -> None:
        if event is not None:
            event.stop()
        self.value = min(self.max_val, round(self.value + self.step, 2 if self.is_float else 0))

    def on_key(self, event: events.Key) -> None:
        if event.key in ("left", "down", "h", "j"):
            self.value = max(self.min_val, round(self.value - self.step, 2 if self.is_float else 0))
            event.stop()
        elif event.key in ("right", "up", "l", "k"):
            self.value = min(self.max_val, round(self.value + self.step, 2 if self.is_float else 0))
            event.stop()


OPTION_GROUPS: dict[str, dict[str, list[str]]] = {
    "generate": {
        "📁 Output & File Options": ["--output"],
        "⏱ Video Dimensions & Timing": [
            "--duration",
            "--fps",
            "--width",
            "--height",
            "--no-timestamp",
            "--timezone",
        ],
        "🗺 Map & Marker Styling": ["--zoom", "--marker-size", "--marker-color"],
        "🔤 Typography & Text Overlay": [
            "--title",
            "--font",
            "--font-scale",
            "--text-color",
            "--text-align",
        ],
        "💬 Captions & Geolocation": [
            "--captions",
            "--scrolling-text",
            "--scrolling-speed",
            "--geolocate",
        ],
    }
}


def apply_trogon_patches() -> None:
    """Apply monkeypatches to Trogon to support Checkbox, RangeSlider, enum Select dropdowns, and grouped controls."""
    from textual.containers import Vertical, VerticalScroll
    from textual.widgets import Label
    from trogon.widgets.form import CommandForm

    orig_param_compose = ParameterControls.compose
    orig_get_control = ParameterControls.get_control_method
    orig_get_form_val = ParameterControls._get_form_control_value
    orig_apply_default = ParameterControls._apply_default_value

    CommandForm.DEFAULT_CSS += """
    .command-form-group-header {
        margin: 1 0 0 0;
        padding: 0 1;
        color: $accent;
        text-style: bold;
        background: $surface;
        border-left: wide $accent;
    }
    """

    def _patched_param_compose(self: ParameterControls) -> Any:
        orig_type = self.schema.type
        if isinstance(orig_type, click.types.BoolParamType):
            self.schema.type = click.BOOL
            try:
                yield from orig_param_compose(self)
            finally:
                self.schema.type = orig_type
        else:
            yield from orig_param_compose(self)

    def _patched_get_control_method(
        self: ParameterControls, argument_type: Any
    ) -> Callable[
        [MultiValueParamData, Text | None, bool, OptionSchema | ArgumentSchema, str], list[ControlWidgetType]
    ]:
        if (
            isinstance(argument_type, (click.IntRange, click.FloatRange))
            and argument_type.min is not None
            and argument_type.max is not None
        ):
            return lambda default, label, multiple, schema, control_id: [
                RangeSlider(
                    min_val=argument_type.min,
                    max_val=argument_type.max,
                    default_val=default.values[0][0] if default.values else argument_type.min,
                    is_float=isinstance(argument_type, click.FloatRange),
                    classes=f"{control_id} command-form-slider",
                )
            ]
        elif isinstance(argument_type, click.types.BoolParamType) or argument_type == click.BOOL:
            return self.make_checkbox_control
        return orig_get_control(self, argument_type)

    @staticmethod
    def _patched_get_form_control_value(control: Any) -> Any:
        if isinstance(control, RangeSlider):
            return control.value_str
        return orig_get_form_val(control)

    @staticmethod
    def _patched_apply_default_value(control_widget: Any, default_value: Any) -> None:
        if isinstance(control_widget, Select):
            val = getattr(default_value, "value", default_value)
            val_str = str(val) if val is not None else Select.BLANK
            control_widget.value = val_str
            control_widget.prompt = f"{val_str} (default)"
        elif isinstance(control_widget, RangeSlider):
            val = getattr(default_value, "value", default_value)
            if val is not None:
                control_widget.value = float(val)
        else:
            orig_apply_default(control_widget, default_value)

    def _patched_form_compose(self: CommandForm) -> Any:
        path_from_root = iter(reversed(self.command_schema.path_from_root))
        command_node = next(path_from_root)
        with VerticalScroll() as vs:
            vs.can_focus = False

            yield Input(
                placeholder="Search...",
                classes="command-form-filter-input",
                id="search",
            )

            while command_node is not None:
                options = command_node.options
                arguments = command_node.arguments
                if options or arguments:
                    with Vertical(classes="command-form-command-group", id=command_node.key) as v:
                        is_inherited = command_node is not self.command_schema
                        prefix = "↪ " if is_inherited else ""
                        v.border_title = f"{prefix}{command_node.name}"
                        if is_inherited:
                            v.border_title += " [dim not bold](inherited)"
                        if arguments:
                            yield Label("Arguments", classes="command-form-heading")
                            for argument in arguments:
                                controls = ParameterControls(argument, id=argument.key)
                                if self.first_control is None:
                                    self.first_control = controls
                                yield controls

                        if options:
                            groups = OPTION_GROUPS.get(command_node.name)
                            if groups:
                                consumed = set()
                                for grp_name, grp_opts in groups.items():
                                    matched = [
                                        opt
                                        for opt in options
                                        if any(
                                            n in grp_opts
                                            for n in (opt.name if isinstance(opt.name, (list, tuple)) else [opt.name])
                                        )
                                    ]
                                    if matched:
                                        yield Label(grp_name, classes="command-form-group-header")
                                        for option in matched:
                                            consumed.add(option.key)
                                            controls = ParameterControls(option, id=option.key)
                                            if self.first_control is None:
                                                self.first_control = controls
                                            yield controls
                                remaining = [opt for opt in options if opt.key not in consumed]
                                if remaining:
                                    yield Label("⚙ Other Options", classes="command-form-group-header")
                                    for option in remaining:
                                        controls = ParameterControls(option, id=option.key)
                                        if self.first_control is None:
                                            self.first_control = controls
                                        yield controls
                            else:
                                yield Label("Options", classes="command-form-heading")
                                for option in options:
                                    controls = ParameterControls(option, id=option.key)
                                    if self.first_control is None:
                                        self.first_control = controls
                                yield controls

                command_node = next(path_from_root, None)

    ParameterControls.compose = _patched_param_compose
    ParameterControls.get_control_method = _patched_get_control_method
    ParameterControls._get_form_control_value = _patched_get_form_control_value
    ParameterControls._apply_default_value = _patched_apply_default_value
    CommandForm.compose = _patched_form_compose
