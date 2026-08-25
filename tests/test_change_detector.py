"""Tests for the pluggable change-detector boundary."""
from __future__ import annotations

from app.tools.change_detector import PixelDifferenceDetector, get_change_detector


def test_default_detector_is_local_and_deterministic():
    detector = get_change_detector()
    assert detector.name == "pixel_difference"


def test_pixel_detector_produces_change_map():
    from PIL import Image

    result = PixelDifferenceDetector().detect(
        Image.new("RGB", (8, 8), "black"),
        Image.new("RGB", (8, 8), "white"),
        0.1,
    )
    assert result.changed_fraction == 1.0