"""Unit tests for ``FontManager``."""

from __future__ import annotations

from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from gpxmapper.font_manager import FontManager


def test_init_default_uses_cv2_font():
    fm = FontManager()
    assert fm.font_scale == 1.0
    assert fm.pil_font is None
    assert fm.cv2_font == cv2.FONT_HERSHEY_SIMPLEX


def test_init_respects_font_scale():
    fm = FontManager(font_scale=1.5)
    assert fm.font_scale == pytest.approx(1.5)


def test_missing_font_file_yields_no_pil_font(tmp_path):
    missing = tmp_path / "nope.ttf"
    fm = FontManager(font_file=str(missing))
    assert fm.pil_font is None


def test_get_text_size_cv2_returns_non_negative_dimensions():
    fm = FontManager()
    (w, h), baseline = fm.get_text_size("Hello", thickness=2)
    assert w >= 0 and h >= 0
    assert baseline >= 0


def test_render_text_cv2_path_mutates_frame_preserves_shape():
    fm = FontManager()
    frame = np.zeros((50, 120, 3), dtype=np.uint8)
    before = frame.copy()
    out = fm.render_text(frame, "Hi", (5, 25), (255, 255, 255), thickness=1)
    assert out is frame
    assert out.shape == before.shape
    assert not np.array_equal(out, before)


def test_render_text_pil_path_returns_new_array_same_shape(mocker):
    mocker.patch("gpxmapper.font_manager.os.path.exists", return_value=True)
    mock_font = MagicMock()
    mock_font.getbbox.return_value = (0, 0, 80, 24)
    mocker.patch("gpxmapper.font_manager.ImageFont.truetype", return_value=mock_font)
    mock_draw = MagicMock()
    mocker.patch("gpxmapper.font_manager.ImageDraw.Draw", return_value=mock_draw)

    fm = FontManager(font_file="/fake/path.ttf", font_scale=1.0)
    assert fm.pil_font is mock_font

    frame = np.zeros((40, 100, 3), dtype=np.uint8)
    out = fm.render_text(frame, "ABC", (10, 10), (255, 0, 0), thickness=1)

    assert out is not frame
    assert out.shape == frame.shape
    assert out.dtype == frame.dtype
    mock_draw.text.assert_called_once()


def test_load_custom_font_passes_scaled_size_to_truetype(mocker):
    mocker.patch("gpxmapper.font_manager.os.path.exists", return_value=True)
    mock_font = MagicMock()
    mock_truetype = mocker.patch(
        "gpxmapper.font_manager.ImageFont.truetype", return_value=mock_font
    )

    FontManager(font_file="/fake/font.ttf", font_scale=2.0)

    mock_truetype.assert_called_once()
    _path, size = mock_truetype.call_args[0]
    assert size == int(24 * 2.0)


def test_load_custom_font_truetype_failure_returns_none(mocker):
    mocker.patch("gpxmapper.font_manager.os.path.exists", return_value=True)
    mocker.patch(
        "gpxmapper.font_manager.ImageFont.truetype",
        side_effect=OSError("bad font"),
    )

    fm = FontManager(font_file="/fake/broken.ttf")
    assert fm.pil_font is None


def test_get_text_size_pil_uses_getbbox_when_available(mocker):
    mocker.patch("gpxmapper.font_manager.os.path.exists", return_value=True)
    mock_font = MagicMock()
    mock_font.getbbox.return_value = (2, 4, 42, 28)
    mocker.patch("gpxmapper.font_manager.ImageFont.truetype", return_value=mock_font)

    fm = FontManager(font_file="/fake/font.ttf")
    (w, h), baseline = fm.get_text_size("x")

    assert (w, h) == (40, 24)
    assert baseline == 24 // 4  # int(size[1] // 4) per FontManager.get_text_size
    mock_font.getbbox.assert_called_once_with("x")


def test_get_text_size_pil_falls_back_to_getlength_when_no_getbbox(mocker):
    mocker.patch("gpxmapper.font_manager.os.path.exists", return_value=True)
    mock_font = MagicMock(spec=["getlength", "size"])
    del mock_font.getbbox  # ensure AttributeError on getbbox branch
    mock_font.getlength.return_value = 99.5
    mock_font.size = 20
    mocker.patch("gpxmapper.font_manager.ImageFont.truetype", return_value=mock_font)

    fm = FontManager(font_file="/fake/font.ttf")
    (w, h), baseline = fm.get_text_size("abcde")

    assert (w, h) == (99, 20)
    assert baseline == 20 // 4


def test_get_text_size_pil_approximation_when_no_getbbox_or_getlength(mocker):
    class _LegacyFont:
        size = 24

    mocker.patch("gpxmapper.font_manager.os.path.exists", return_value=True)
    mocker.patch("gpxmapper.font_manager.ImageFont.truetype", return_value=_LegacyFont())

    fm = FontManager(font_file="/fake/font.ttf")
    text = "abcd"
    (w, h), baseline = fm.get_text_size(text)

    assert h == 24
    assert w == int(len(text) * int(24 * 0.6))
    assert baseline == int(h // 4)
