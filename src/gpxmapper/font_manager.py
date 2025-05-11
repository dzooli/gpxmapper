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
        """Initialize the font manager.

        Args:
            font_file: Optional path to a TrueType font file (.ttf)
            font_scale: Font scale for OpenCV rendering (used if custom font is not provided)
        """
        self.font_file = font_file
        self.font_scale = font_scale
        self.pil_font = None
        self.cv2_font = cv2.FONT_HERSHEY_SIMPLEX

        # Load custom font if provided
        if font_file and os.path.exists(font_file):
            try:
                # Calculate font size based on font_scale (approximation)
                # OpenCV font_scale 1.0 is roughly equivalent to a 24pt font
                font_size = int(24 * font_scale)
                self.pil_font = ImageFont.truetype(font_file, font_size)
                logger.info(f"Loaded custom font from {font_file} with size {font_size}")
            except Exception as e:
                logger.error(f"Failed to load custom font {font_file}: {e}")
                self.pil_font = None

    def render_text(self, frame: np.ndarray, text: str, position: Tuple[int, int], 
                   color: Tuple[int, int, int], thickness: int = 2) -> np.ndarray:
        """Render text on a frame using either PIL (if custom font is loaded) or OpenCV.

        Args:
            frame: The frame to render text on
            text: The text to render
            position: (x, y) position for the text
            color: RGB color tuple for the text
            thickness: Line thickness for OpenCV rendering

        Returns:
            The frame with rendered text
        """
        if self.pil_font is None:
            # Use OpenCV for rendering if no custom font is loaded
            cv2.putText(
                frame, text, position,
                self.cv2_font, self.font_scale, color, thickness
            )
            return frame
        else:
            # Convert OpenCV BGR frame to PIL RGB Image
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)

            # Draw text with custom font
            draw.text(position, text, fill=color, font=self.pil_font)

            # Convert back to OpenCV BGR format
            return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def get_text_size(self, text: str, thickness: int = 2) -> Tuple[Tuple[int, int], int]:
        """Get the size of text when rendered.

        Args:
            text: The text to measure
            thickness: Line thickness for OpenCV rendering

        Returns:
            ((width, height), baseline) tuple
        """
        if self.pil_font is None:
            # Use OpenCV for measuring text size
            size, baseline = cv2.getTextSize(text, self.cv2_font, self.font_scale, thickness)
            # Ensure the size is a tuple[int, int] rather than a Sequence[int]
            return (size[0], size[1]), baseline
        else:
            # Use PIL for measuring text size
            # Handle different Pillow versions (getsize is deprecated in newer versions)
            try:
                # Try newer Pillow version method first
                bbox = self.pil_font.getbbox(text)
                width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            except AttributeError:
                try:
                    # Fall back to older Pillow version method
                    width, height = self.pil_font.getsize(text)
                except AttributeError:
                    # If all else fails, use a rough approximation
                    width = len(text) * int(self.pil_font.size * 0.6)
                    height = self.pil_font.size

            # Approximate baseline as 1/4 of height
            baseline = height // 4
            return (width, height), baseline