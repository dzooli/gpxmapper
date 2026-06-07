"""Unit tests for ``gpxmapper.cli`` (Typer app, commands, and CLI utilities)."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import typer
from click.testing import Result
from typer.testing import CliRunner

from gpxmapper.cli import app
from gpxmapper.cli.utils import create_text_config, generate_video, parse_color
from gpxmapper.models import MapConfig, VideoConfig

GPX_HEADER = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<gpx version="1.1" creator="test" xmlns="http://www.topografix.com/GPX/1/1">'
)
GPX_FOOTER = "</gpx>"


def _write_gpx(path: Path, body: str) -> Path:
    path.write_text(GPX_HEADER + body + GPX_FOOTER, encoding="utf-8")
    return path


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


def _invoke(runner: CliRunner, args: list[str], *, standalone_mode: bool = False) -> Result:
    """Invoke the Typer app in a way that works with Click's test runner.

    ``gpxmapper.cli`` calls ``basicConfig(StreamHandler(sys.stdout))`` at import
    time. :class:`~click.testing.CliRunner` swaps ``sys.stdout`` for a buffer and
    may close the previous stream; the logging handler still references it, which
    leads to ``ValueError: I/O operation on closed file`` when flushing output.

    Temporarily detach the root handlers, then restore them after the invoke.
    """
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    root.handlers.clear()
    root.addHandler(logging.NullHandler())
    try:
        return runner.invoke(app, args, standalone_mode=standalone_mode)
    finally:
        root.handlers.clear()
        for handler in saved_handlers:
            root.addHandler(handler)


@pytest.fixture
def gpx_with_times(tmp_path: Path) -> Path:
    body = """
  <trk><trkseg>
    <trkpt lat="47.0" lon="19.0"><ele>100</ele><time>2020-01-01T12:00:00Z</time></trkpt>
    <trkpt lat="47.1" lon="19.1"><ele>101</ele><time>2020-01-01T13:00:00Z</time></trkpt>
  </trkseg></trk>
"""
    return _write_gpx(tmp_path / "track.gpx", body)


@pytest.fixture
def gpx_no_times(tmp_path: Path) -> Path:
    body = """
  <trk><trkseg>
    <trkpt lat="1.0" lon="2.0"><ele>10</ele></trkpt>
  </trkseg></trk>
"""
    return _write_gpx(tmp_path / "notime.gpx", body)


@pytest.fixture
def gpx_empty_track(tmp_path: Path) -> Path:
    return _write_gpx(tmp_path / "empty.gpx", "<trk><trkseg></trkseg></trk>")


# --- cli.utils ---


def test_parse_color_valid():
    assert parse_color("0,0,0") == (0, 0, 0)
    assert parse_color("255,128, 0") == (255, 128, 0)


def test_parse_color_out_of_range():
    with pytest.raises(typer.BadParameter):
        parse_color("256,0,0")


def test_parse_color_invalid_format():
    with pytest.raises(typer.BadParameter):
        parse_color("red")


def test_create_text_config_builds_dataclass():
    cfg = create_text_config(
        font_scale=1.2,
        title_text="Hi",
        text_align="center",
        font_file="/tmp/f.ttf",
        no_timestamp=True,
        timezone="Europe/Budapest",
    )
    assert cfg.font_scale == pytest.approx(1.2)
    assert cfg.title_text == "Hi"
    assert cfg.text_align == "center"
    assert cfg.font_file == "/tmp/f.ttf"
    assert cfg.show_timestamp is False
    assert cfg.timezone == "Europe/Budapest"
    assert cfg.geolocate is False


def test_create_text_config_geolocate_true():
    cfg = create_text_config(font_scale=1.0, geolocate=True)
    assert cfg.geolocate is True


def test_create_text_config_rejects_bad_alignment():
    with pytest.raises(typer.BadParameter, match="left, center, right"):
        create_text_config(font_scale=1.0, text_align="justify")


def test_create_text_config_rejects_bad_timestamp_color():
    with pytest.raises(typer.BadParameter, match="Timestamp color"):
        create_text_config(font_scale=1.0, timestamp_color="300,0,0")


def test_generate_video_aborts_when_no_points(gpx_empty_track: Path, tmp_path: Path):
    out = tmp_path / "out.mp4"
    with pytest.raises(typer.Abort):
        generate_video(
            gpx_file=gpx_empty_track,
            output_file=out,
            video_config=VideoConfig(fps=30, width=320, height=240, duration=5),
            map_config=MapConfig(zoom=10, marker_size=5, marker_color=(255, 0, 0)),
            text_config=create_text_config(font_scale=1.0),
        )


def test_generate_video_aborts_when_no_timestamps(
        gpx_no_times: Path, tmp_path: Path
):
    out = tmp_path / "out.mp4"
    with pytest.raises(typer.Abort):
        generate_video(
            gpx_file=gpx_no_times,
            output_file=out,
            video_config=VideoConfig(fps=30, width=320, height=240, duration=5),
            map_config=MapConfig(zoom=10, marker_size=5, marker_color=(255, 0, 0)),
            text_config=create_text_config(font_scale=1.0),
        )


# --- Typer app ---


def test_app_help_lists_commands(cli_runner: CliRunner):
    result = _invoke(cli_runner, ["--help"])
    assert result.exit_code == 0
    assert "generate" in result.stdout
    assert "info" in result.stdout
    assert "clear-cache" in result.stdout
    assert "check-nominatim" in result.stdout


def test_info_success(cli_runner: CliRunner, gpx_with_times: Path):
    result = _invoke(cli_runner, ["info", str(gpx_with_times)])
    assert result.exit_code == 0
    assert "Number of track points: 2" in result.stdout
    assert "Time range:" in result.stdout
    assert "Coordinate bounds:" in result.stdout


def test_info_aborts_when_no_track_points(cli_runner: CliRunner, gpx_empty_track: Path):
    result = _invoke(cli_runner, ["info", str(gpx_empty_track)])
    assert result.exit_code != 0


def test_info_no_time_data_message(cli_runner: CliRunner, gpx_no_times: Path):
    result = _invoke(cli_runner, ["info", str(gpx_no_times)])
    assert result.exit_code == 0
    assert "No time data available" in result.stdout
    assert "Number of track points: 1" in result.stdout


def test_generate_success_mocked_video(
        cli_runner: CliRunner, gpx_with_times: Path, mocker
):
    mock_gen = mocker.patch(
        "gpxmapper.cli.generate.generate_video", return_value=str(gpx_with_times.with_suffix(".mp4"))
    )

    result = _invoke(cli_runner, ["generate", str(gpx_with_times)])

    assert result.exit_code == 0
    mock_gen.assert_called_once()
    assert mock_gen.call_args.kwargs["gpx_file"] == gpx_with_times


def test_generate_aborts_on_bad_marker_color(
        cli_runner: CliRunner, gpx_with_times: Path
):
    result = _invoke(
        cli_runner,
        ["generate", str(gpx_with_times), "--marker-color", "notrgb"],
    )
    assert result.exit_code != 0


def test_generate_geolocate_conflicts_with_scrolling_text(
        cli_runner: CliRunner, gpx_with_times: Path, tmp_path: Path
):
    scroll = tmp_path / "scroll.txt"
    scroll.write_text("hello", encoding="utf-8")
    result = _invoke(
        cli_runner,
        [
            "generate",
            str(gpx_with_times),
            "--geolocate",
            "--scrolling-text",
            str(scroll),
        ],
        standalone_mode=True,
    )
    assert result.exit_code != 0
    combined = (result.stdout or "") + (result.stderr or "")
    assert "--geolocate" in combined
    assert "scrolling" in combined.lower()


def test_generate_aborts_when_generate_video_fails(
        cli_runner: CliRunner, gpx_with_times: Path, mocker
):
    mocker.patch(
        "gpxmapper.cli.generate.generate_video",
        side_effect=RuntimeError("encode failed"),
    )
    result = _invoke(cli_runner, ["generate", str(gpx_with_times)])
    assert result.exit_code != 0


def test_clear_cache_when_directory_missing(cli_runner: CliRunner, tmp_path: Path, mocker):
    missing = tmp_path / "no_cache"
    mocker.patch(
        "gpxmapper.cli.clear_cache.MapRendererBase.resolve_default_cache_directory",
        return_value=str(missing),
    )
    mocker.patch("gpxmapper.cli.clear_cache.os.path.exists", return_value=False)

    result = _invoke(cli_runner, ["clear-cache"])

    assert result.exit_code == 0
    assert "does not exist" in result.stdout


def test_clear_cache_when_already_empty(cli_runner: CliRunner, tmp_path: Path, mocker):
    cache = tmp_path / "cache"
    cache.mkdir()
    mocker.patch(
        "gpxmapper.cli.clear_cache.MapRendererBase.resolve_default_cache_directory",
        return_value=str(cache),
    )
    mocker.patch("gpxmapper.cli.clear_cache.os.path.exists", return_value=True)

    result = _invoke(cli_runner, ["clear-cache"])

    assert result.exit_code == 0
    assert "already empty" in result.stdout


def test_clear_cache_cancelled_when_not_confirmed(
        cli_runner: CliRunner, tmp_path: Path, mocker
):
    cache = tmp_path / "cache"
    cache.mkdir()
    stale = cache / "tile.png"
    stale.write_bytes(b"x")
    mocker.patch(
        "gpxmapper.cli.clear_cache.MapRendererBase.resolve_default_cache_directory",
        return_value=str(cache),
    )
    mocker.patch("gpxmapper.cli.clear_cache.os.path.exists", return_value=True)
    mocker.patch("gpxmapper.cli.clear_cache.typer.confirm", return_value=False)

    result = _invoke(cli_runner, ["clear-cache"])

    assert result.exit_code == 0
    assert "cancelled" in result.stdout
    assert stale.is_file()


def test_clear_cache_deletes_files_when_confirmed(
        cli_runner: CliRunner, tmp_path: Path, mocker
):
    cache = tmp_path / "cache"
    cache.mkdir()
    stale = cache / "tile.png"
    stale.write_bytes(b"x")
    mocker.patch(
        "gpxmapper.cli.clear_cache.MapRendererBase.resolve_default_cache_directory",
        return_value=str(cache),
    )
    mocker.patch("gpxmapper.cli.clear_cache.os.path.exists", return_value=True)
    mocker.patch("gpxmapper.cli.clear_cache.typer.confirm", return_value=True)

    result = _invoke(cli_runner, ["clear-cache"])

    assert result.exit_code == 0
    assert "Successfully cleared" in result.stdout
    assert not stale.exists()
