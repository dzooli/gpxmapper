from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

import numpy as np
import pytest
from PIL import Image

from gpxmapper.models import GPXTrackPoint
from gpxmapper.video_generator import VideoGenerator


class _DummyCaptioner:
    def __init__(self, *args, **kwargs):  # noqa: ANN002
        self.start_time = None
        self.duration = None

    def add_timestamp_to_frame(self, frame, timestamp):  # noqa: ANN001
        return frame

    def add_title_to_frame(self, frame):  # noqa: ANN001
        return frame

    def add_caption_to_frame(self, frame, seconds_since_start):  # noqa: ANN001
        return frame

    def add_scrolling_text_to_frame(self, frame, frame_idx):  # noqa: ANN001
        return frame

    def add_geolocation_text_to_frame(self, frame, frame_idx):  # noqa: ANN001
        return frame

    def set_video_start_time(self, start_time):  # noqa: ANN001
        self.start_time = start_time

    def set_video_duration(self, duration_seconds, fps):  # noqa: ANN001
        self.duration = (duration_seconds, fps)


def _point(lat: float, lon: float, ts: datetime | None) -> GPXTrackPoint:
    return GPXTrackPoint(lat, lon, None, ts, None)


@pytest.fixture
def vg():
    renderer = Mock()
    renderer.render_from_composite.return_value = Image.new("RGB", (64, 48), (10, 20, 30))
    renderer.create_composite_map.return_value = None
    renderer.composite_map_info = {"width": 64, "height": 48}
    with patch("gpxmapper.video_generator.MapRendererFactory.create", return_value=renderer), patch(
            "gpxmapper.video_generator.VideoCaptioner", _DummyCaptioner
    ):
        yield VideoGenerator(output_path="out.mp4", fps=2, resolution=(64, 48))


def test_prepare_track_points_filters_and_sorts(vg: VideoGenerator) -> None:
    t0 = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
    points = [
        _point(1.0, 1.0, t0 + timedelta(seconds=10)),
        _point(0.0, 0.0, None),
        _point(2.0, 2.0, t0),
    ]
    prepared = vg._prepare_track_points(points)
    assert [p.time for p in prepared] == [t0, t0 + timedelta(seconds=10)]


def test_prepare_track_points_raises_without_timestamps(vg: VideoGenerator) -> None:
    with pytest.raises(ValueError, match="don't have time data"):
        vg._prepare_track_points([_point(1.0, 1.0, None)])


def test_interpolate_position_interpolates_and_caches(vg: VideoGenerator) -> None:
    t0 = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(seconds=10)
    points = [_point(0.0, 0.0, t0), _point(10.0, 20.0, t1)]

    pos_mid = vg._interpolate_position(points, t0 + timedelta(seconds=5))
    assert pos_mid == pytest.approx((5.0, 10.0))

    cached = vg._interpolate_position(points, t0 + timedelta(seconds=5))
    assert cached == pytest.approx(pos_mid)
    assert len(vg._position_cache) == 1


def test_interpolate_position_out_of_range_raises(vg: VideoGenerator) -> None:
    t0 = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
    points = [_point(0.0, 0.0, t0), _point(1.0, 1.0, t0 + timedelta(seconds=10))]
    with pytest.raises(ValueError, match="outside the track time range"):
        vg._interpolate_position(points, t0 - timedelta(seconds=1))


def test_generate_frame_uses_blank_frame_when_map_missing(vg: VideoGenerator) -> None:
    vg.map_renderer.render_from_composite.return_value = None
    t0 = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
    points = [_point(0.0, 0.0, t0), _point(1.0, 1.0, t0 + timedelta(seconds=10))]
    frame = vg._generate_frame(0, t0, 0.0, points)
    assert frame.shape == (48, 64, 3)
    assert np.count_nonzero(frame) == 0


def test_generate_frame_converts_rendered_image(vg: VideoGenerator) -> None:
    t0 = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
    points = [_point(0.0, 0.0, t0), _point(1.0, 1.0, t0 + timedelta(seconds=10))]
    frame = vg._generate_frame(0, t0, 0.0, points)
    assert frame.shape == (48, 64, 3)
    assert np.count_nonzero(frame) > 0


def test_write_video_frames_writes_expected_frame_count(vg: VideoGenerator) -> None:
    t0 = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
    points = [_point(0.0, 0.0, t0), _point(1.0, 1.0, t0 + timedelta(seconds=4))]
    writer = Mock()
    vg._write_video_frames(writer, points, duration_seconds=2, start_time=t0, total_track_seconds=4.0)
    # fps=2 * duration=2
    assert writer.write.call_count == 4


def test_generate_video_raises_when_writer_not_opened(vg: VideoGenerator) -> None:
    t0 = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
    points = [_point(0.0, 0.0, t0), _point(1.0, 1.0, t0 + timedelta(seconds=5))]
    writer = Mock()
    writer.isOpened.return_value = False
    with patch("gpxmapper.video_generator.cv2.VideoWriter", return_value=writer):
        with pytest.raises(ValueError, match="Failed to open video writer"):
            vg.generate_video(points, duration_seconds=2)
