"""Unit tests for the gpxmapper.api programmatic interface."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import ANY

import pytest

from gpxmapper.api import (
    check_nominatim_status,
    clear_geolocation_cache,
    clear_tile_cache,
    create_text_config,
    generate_video,
    get_geolocation_cache_info,
    get_gpx_info,
    get_tile_cache_info,
    parse_color,
    resolve_configs,
    resolve_map_config,
    resolve_text_config,
    resolve_video_config,
)
from gpxmapper.exceptions import (
    ConfigurationError,
    GPXEmptyError,
    GPXMissingTimeError,
    GPXParseError,
    VideoGenerationError,
)
from gpxmapper.models import MapConfig, VideoConfig

GPX_HEADER = (
    '<?xml version="1.0" encoding="UTF-8"?><gpx version="1.1" creator="test" xmlns="http://www.topografix.com/GPX/1/1">'
)
GPX_FOOTER = "</gpx>"


def _write_gpx(path: Path, body: str) -> Path:
    path.write_text(GPX_HEADER + body + GPX_FOOTER, encoding="utf-8")
    return path


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


# --- Config & Color Parsing ---


def test_parse_color_valid():
    assert parse_color("0,0,0") == (0, 0, 0)
    assert parse_color("255, 128, 0") == (255, 128, 0)


def test_parse_color_invalid():
    with pytest.raises(ConfigurationError):
        parse_color("invalid")
    with pytest.raises(ConfigurationError):
        parse_color("256,0,0")
    with pytest.raises(ConfigurationError):
        parse_color("-1,0,0")


def test_create_text_config_valid():
    cfg = create_text_config(
        font_scale=1.5,
        title_text="Track Title",
        text_align="center",
        timestamp_color="10,20,30",
        timezone="UTC",
    )
    assert cfg.font_scale == 1.5
    assert cfg.title_text == "Track Title"
    assert cfg.text_align == "center"
    assert cfg.timestamp_color == (10, 20, 30)
    assert cfg.timezone == "UTC"


def test_create_text_config_tuple_color():
    cfg = create_text_config(timestamp_color=(100, 150, 200))
    assert cfg.timestamp_color == (100, 150, 200)


def test_create_text_config_invalid_color():
    with pytest.raises(ConfigurationError):
        create_text_config(timestamp_color="not_a_color")
    with pytest.raises(ConfigurationError):
        create_text_config(timestamp_color=(300, 0, 0))  # type: ignore


def test_create_text_config_invalid_alignment():
    with pytest.raises(ConfigurationError, match="Text alignment must be one of"):
        create_text_config(text_align="justify")


def test_create_text_config_geolocate_conflict():
    with pytest.raises(ConfigurationError, match="geolocate cannot be enabled together with"):
        create_text_config(geolocate=True, scrolling_text_file="some_file.txt")


def test_resolve_configs_direct_helpers():
    v = resolve_video_config(options={"fps": 60, "duration": 120})
    assert v.fps == 60
    assert v.duration == 120
    assert v.width == 320

    m = resolve_map_config(options={"zoom": 17, "marker_color": "0,255,0"})
    assert m.zoom == 17
    assert m.marker_color == (0, 255, 0)

    t = resolve_text_config(options={"title": "My Track", "text_color": "1,2,3", "show_timestamp": True})
    assert t.title_text == "My Track"
    assert t.timestamp_color == (1, 2, 3)
    assert t.show_timestamp is True

    vc, mc, tc = resolve_configs(options={"fps": 24, "zoom": 10, "title": "Trip"})
    assert vc.fps == 24
    assert mc.zoom == 10
    assert tc.title_text == "Trip"


# --- GPX Info ---


def test_get_gpx_info_success(gpx_with_times: Path):
    info = get_gpx_info(gpx_with_times)
    assert info.file_path == gpx_with_times
    assert info.point_count == 2
    assert info.start_time is not None
    assert info.end_time is not None
    assert info.duration is not None
    assert info.duration.total_seconds() == 3600
    assert info.coordinate_bounds == (47.0, 19.0, 47.1, 19.1)


def test_get_gpx_info_no_times(gpx_no_times: Path):
    info = get_gpx_info(gpx_no_times)
    assert info.point_count == 1
    assert info.start_time is None
    assert info.duration is None


def test_get_gpx_info_empty_raises(gpx_empty_track: Path):
    with pytest.raises(GPXEmptyError):
        get_gpx_info(gpx_empty_track)


def test_get_gpx_info_invalid_file_raises(tmp_path: Path):
    bad_file = tmp_path / "bad.gpx"
    bad_file.write_text("not gpx content", encoding="utf-8")
    with pytest.raises(GPXParseError):
        get_gpx_info(bad_file)


# --- Video Generation ---


def test_generate_video_empty_gpx_raises(gpx_empty_track: Path, tmp_path: Path):
    out = tmp_path / "out.mp4"
    with pytest.raises(GPXEmptyError):
        generate_video(gpx_file=gpx_empty_track, output_file=out)


def test_generate_video_no_timestamps_raises(gpx_no_times: Path, tmp_path: Path):
    out = tmp_path / "out.mp4"
    with pytest.raises(GPXMissingTimeError):
        generate_video(gpx_file=gpx_no_times, output_file=out)


@pytest.fixture
def mock_video_generator(mocker, tmp_path: Path):
    """Fixture providing a mocked VideoGenerator and output path."""
    out = tmp_path / "output.mp4"
    mock_cls = mocker.patch("gpxmapper.api.video.VideoGenerator")
    instance = mock_cls.return_value
    instance.generate_video.return_value = str(out)
    return out, mock_cls, instance


def test_generate_video_success_mocked(gpx_with_times: Path, mock_video_generator):
    out, _, mock_instance = mock_video_generator

    result = generate_video(
        gpx_file=gpx_with_times,
        output_file=out,
        video_config=VideoConfig(fps=30, width=320, height=240, duration=10),
        map_config=MapConfig(zoom=12, marker_size=8, marker_color=(0, 255, 0)),
        text_config=create_text_config(title_text="Trip"),
    )

    assert result == str(out)
    mock_instance.generate_video.assert_called_once()


def test_generate_video_failure_wraps_exception(gpx_with_times: Path, tmp_path: Path, mocker):
    out = tmp_path / "out.mp4"
    mock_gen_instance = mocker.MagicMock()
    mock_gen_instance.generate_video.side_effect = RuntimeError("OpenCV encoder crash")
    mocker.patch("gpxmapper.api.video.VideoGenerator", return_value=mock_gen_instance)

    with pytest.raises(VideoGenerationError, match="OpenCV encoder crash"):
        generate_video(gpx_file=gpx_with_times, output_file=out)


def test_generate_video_convenience_kwargs_example(gpx_with_times: Path, mock_video_generator):
    """Test the exact documented example from README.md with convenience kwargs."""
    out, mock_cls, mock_instance = mock_video_generator

    result = generate_video(
        gpx_path=gpx_with_times,
        output_path=out,
        duration=60,
        fps=30,
        width=1280,
        height=720,
        zoom=15,
        marker_color=(255, 0, 0),
        title="Morning Ride",
        text_color=(255, 255, 255),
    )

    assert result == str(out)
    mock_cls.assert_called_once()
    _, kwargs = mock_cls.call_args
    assert kwargs["output_path"] == str(out)
    assert kwargs["fps"] == 30
    assert kwargs["resolution"] == (1280, 720)
    assert kwargs["zoom_level"] == 15
    assert kwargs["marker_color"] == (255, 0, 0)
    assert kwargs["marker_size"] == 10
    assert kwargs["text_config"].title_text == "Morning Ride"
    assert kwargs["text_config"].timestamp_color == (255, 255, 255)
    mock_instance.generate_video.assert_called_once_with(ANY, 60)


def test_generate_video_missing_gpx_path_raises():
    with pytest.raises(ValueError, match="A GPX file path must be provided"):
        generate_video()


def test_generate_video_convenience_string_colors_and_options(gpx_with_times: Path, mock_video_generator):
    out, mock_cls, _ = mock_video_generator

    result = generate_video(
        gpx_file=gpx_with_times,
        output_file=out,
        marker_color="0,128,255",
        marker_size=12,
        title_text="Custom Title",
        text_color="10,20,30",
        font_scale=1.2,
        text_align="center",
        no_timestamp=True,
    )

    assert result == str(out)
    _, kwargs = mock_cls.call_args
    assert kwargs["marker_color"] == (0, 128, 255)
    assert kwargs["marker_size"] == 12
    assert kwargs["text_config"].title_text == "Custom Title"
    assert kwargs["text_config"].timestamp_color == (10, 20, 30)
    assert kwargs["text_config"].font_scale == 1.2
    assert kwargs["text_config"].text_align == "center"
    assert kwargs["text_config"].show_timestamp is False


def test_generate_video_config_with_keyword_overrides(gpx_with_times: Path, mock_video_generator):
    out, mock_cls, mock_instance = mock_video_generator

    result = generate_video(
        gpx_file=gpx_with_times,
        output_file=out,
        video_config=VideoConfig(fps=24, width=640, height=480, duration=10),
        map_config=MapConfig(zoom=10, marker_size=5, marker_color=(0, 0, 0)),
        text_config=create_text_config(title_text="Base Title", font_scale=0.8),
        duration=45,
        zoom=14,
        title="Overridden Title",
    )

    assert result == str(out)
    _, kwargs = mock_cls.call_args
    assert kwargs["fps"] == 24
    assert kwargs["resolution"] == (640, 480)
    assert kwargs["zoom_level"] == 14
    assert kwargs["text_config"].title_text == "Overridden Title"
    assert kwargs["text_config"].font_scale == 0.8
    mock_instance.generate_video.assert_called_once_with(ANY, 45)


# --- Cache Services ---


def test_tile_cache_info_and_clear(tmp_path: Path):
    cache_dir = tmp_path / "tile_cache"
    info = get_tile_cache_info(cache_dir)
    assert not info.exists
    assert info.file_count == 0

    cache_dir.mkdir()
    (cache_dir / "tile1.png").write_bytes(b"tile1")
    (cache_dir / "tile2.png").write_bytes(b"tile2")

    info = get_tile_cache_info(cache_dir)
    assert info.exists
    assert info.file_count == 2

    res = clear_tile_cache(cache_dir)
    assert res.success
    assert res.files_deleted == 2
    assert get_tile_cache_info(cache_dir).file_count == 0


def test_geolocation_cache_info_and_clear(tmp_path: Path):
    db_file = tmp_path / "cache.sqlite"
    info = get_geolocation_cache_info(db_file)
    assert not info.exists

    db_file.write_bytes(b"sqlite content")
    info = get_geolocation_cache_info(db_file)
    assert info.exists
    assert info.file_count == 1

    res = clear_geolocation_cache(db_file)
    assert res.success
    assert res.files_deleted == 1
    assert not db_file.exists()


# --- Nominatim Status ---


def test_check_nominatim_status_success(mocker):
    mocker.patch("gpxmapper.api.nominatim.probe_nominatim_status_sync", return_value=(True, None))
    mocker.patch("gpxmapper.api.nominatim.get_nominatim_base_url", return_value="http://localhost:8080")

    ok, url, err = check_nominatim_status()
    assert ok is True
    assert url == "http://localhost:8080"
    assert err is None


def test_check_nominatim_status_failure(mocker):
    mocker.patch(
        "gpxmapper.api.nominatim.probe_nominatim_status_sync",
        return_value=(False, "Connection refused"),
    )
    mocker.patch("gpxmapper.api.nominatim.get_nominatim_base_url", return_value="http://localhost:8080")

    ok, url, err = check_nominatim_status()
    assert ok is False
    assert url == "http://localhost:8080"
    assert err == "Connection refused"
