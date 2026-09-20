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
        height: auto;
        layout: horizontal;
        padding: 0;
        margin: 0;
    }
    .slider-track {
        width: 1fr;
        height: 1;
        content-align: center middle;
    }
    .slider-val {
        width: 8;
        text-align: right;
        color: $accent;
        text-style: bold;
    }
    .slider-btn {
        min-width: 3;
        width: 3;
        height: 1;
        padding: 0;
        margin: 0 1;
        border: none;
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

    def compose(self) -> ComposeResult:
        yield Button("<", classes="slider-btn btn-dec")
        yield Static(self._render_bar(24), classes="slider-track", id="track")
        yield Button(">", classes="slider-btn btn-inc")
        yield Static(self.value_str, classes="slider-val", id="val")

    def _render_bar(self, width: int = 24) -> Text:
        span = max(1e-9, (self.max_val - self.min_val))
        ratio = (self.value - self.min_val) / span
        ratio = max(0.0, min(1.0, ratio))
        pos = int(ratio * (width - 1))
        chars = []
        for i in range(width):
            if i == pos:
                chars.append("●")
            elif i < pos:
                chars.append("━")
            else:
                chars.append("─")
        return Text("".join(chars), style="bold cyan")

    def watch_value(self, new_val: float) -> None:
        try:
            track = self.query_one("#track", Static)
            track.update(self._render_bar(24))
            val_lbl = self.query_one("#val", Static)
            val_lbl.update(self.value_str)
        except Exception:
            pass
        # Post Input.Changed message so the Trogon form automatically updates the preview
        fake_input = Input(value=self.value_str)
        self.post_message(Input.Changed(fake_input, self.value_str))

    @on(Button.Pressed, ".btn-dec")
    def dec(self, event: Button.Pressed) -> None:
        event.stop()
        self.value = max(self.min_val, round(self.value - self.step, 2 if self.is_float else 0))

    @on(Button.Pressed, ".btn-inc")
    def inc(self, event: Button.Pressed) -> None:
        event.stop()
        self.value = min(self.max_val, round(self.value + self.step, 2 if self.is_float else 0))

    def on_key(self, event: events.Key) -> None:
        if event.key in ("left", "down", "h", "j"):
            self.value = max(self.min_val, round(self.value - self.step, 2 if self.is_float else 0))
            event.stop()
        elif event.key in ("right", "up", "l", "k"):
            self.value = min(self.max_val, round(self.value + self.step, 2 if self.is_float else 0))
            event.stop()


def apply_trogon_patches() -> None:
    """Apply monkeypatches to Trogon to support Checkbox, RangeSlider, and enum Select dropdowns."""
    orig_compose = ParameterControls.compose
    orig_get_control = ParameterControls.get_control_method
    orig_get_form_val = ParameterControls._get_form_control_value
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

    ParameterControls.compose = _patched_compose
    ParameterControls.get_control_method = _patched_get_control_method
    ParameterControls._get_form_control_value = _patched_get_form_control_value
    ParameterControls._apply_default_value = _patched_apply_default_value
