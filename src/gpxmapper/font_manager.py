"""Font management module for text rendering in videos.

This module provides the FontManager class which handles font loading and text rendering
for video frames using either PIL or OpenCV.
"""

import logging
import os
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


class FontManager:
    """Manages font loading and text rendering for video frames."""

    def __init__(self, font_file: Optional[str] = None, font_scale: float = 1.0):
        """Initialize the font manager."""
        self.font_scale = font_scale
        self.pil_font = self._load_custom_font(font_file) if font_file else None
        self.cv2_font = cv2.FONT_HERSHEY_SIMPLEX

    def _load_custom_font(self, font_file: str) -> Optional[ImageFont.FreeTypeFont]:
        """Load a custom TrueType font."""
        if not os.path.exists(font_file):
            return None

        try:
            font_size = int(24 * self.font_scale)
            font = ImageFont.truetype(font_file, font_size)
            logger.info(f"Loaded custom font from {font_file} with size {font_size}")
            return font
        except Exception as e:
            logger.error(f"Failed to load custom font {font_file}: {e}")
            return None

    def render_text(self, frame: np.ndarray, text: str, position: Tuple[int, int],
                    color: Tuple[int, int, int], thickness: int = 2) -> np.ndarray:
        """Render text on a frame using either PIL or OpenCV."""
        if not self.pil_font:
            cv2.putText(frame, text, position, self.cv2_font, self.font_scale,
                        color, thickness)
            return frame

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ImageDraw.Draw(pil_img).text(position, text, fill=color, font=self.pil_font)
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def get_text_size(self, text: str, thickness: int = 2) -> Tuple[Tuple[int, int], int]:
        """Get the size of text when rendered."""
        if not self.pil_font:
            size, baseline = cv2.getTextSize(text, self.cv2_font, self.font_scale, thickness)
            return (size[0], size[1]), baseline

        try:
            bbox = self.pil_font.getbbox(text)
            size = (int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1]))
        except AttributeError:
            try:
                # getlength is available in newer Pillow versions
                width = self.pil_font.getlength(text)
                # Approximate height based on font size
                height = self.pil_font.size
                size = (int(width), int(height))
            except AttributeError:
                # Fallback to a very basic approximation
                size = (int(len(text) * int(self.pil_font.size * 0.6)), int(self.pil_font.size))

        return size, int(size[1] // 4)
