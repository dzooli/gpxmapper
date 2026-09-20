"""Unit tests for the gpxmapper.api programmatic interface."""

from __future__ import annotations

from pathlib import Path

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


def test_generate_video_success_mocked(gpx_with_times: Path, tmp_path: Path, mocker):
    out = tmp_path / "out.mp4"
    mock_gen_instance = mocker.MagicMock()
    mock_gen_instance.generate_video.return_value = str(out)
    mocker.patch("gpxmapper.api.video.VideoGenerator", return_value=mock_gen_instance)

    result = generate_video(
        gpx_file=gpx_with_times,
        output_file=out,
        video_config=VideoConfig(fps=30, width=320, height=240, duration=10),
        map_config=MapConfig(zoom=12, marker_size=8, marker_color=(0, 255, 0)),
        text_config=create_text_config(title_text="Trip"),
    )

    assert result == str(out)
    mock_gen_instance.generate_video.assert_called_once()


def test_generate_video_failure_wraps_exception(gpx_with_times: Path, tmp_path: Path, mocker):
    out = tmp_path / "out.mp4"
    mock_gen_instance = mocker.MagicMock()
    mock_gen_instance.generate_video.side_effect = RuntimeError("OpenCV encoder crash")
    mocker.patch("gpxmapper.api.video.VideoGenerator", return_value=mock_gen_instance)

    with pytest.raises(VideoGenerationError, match="OpenCV encoder crash"):
        generate_video(gpx_file=gpx_with_times, output_file=out)


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
