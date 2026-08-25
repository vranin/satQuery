"""Change-detector interface with a deterministic local fallback."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from PIL import Image

from app.tools.difference import ChangeMapResult, compute_change_map


class ChangeDetector(Protocol):
    """Common interface for pixel difference and model-based detectors."""

    name: str

    def detect(self, image_t1: Image.Image, image_t2: Image.Image, threshold: float) -> ChangeMapResult:
        ...


@dataclass(frozen=True)
class PixelDifferenceDetector:
    """CPU-only fallback used unless a ChangeFormer adapter is configured."""

    name: str = "pixel_difference"

    def detect(self, image_t1: Image.Image, image_t2: Image.Image, threshold: float) -> ChangeMapResult:
        return compute_change_map(image_t1, image_t2, threshold=threshold)


def get_change_detector() -> ChangeDetector:
    """Return the configured detector without downloading model weights."""
    # ChangeFormer can be added behind this interface when its runtime and
    # checkpoint are provisioned on a remote GPU.
    return PixelDifferenceDetector()