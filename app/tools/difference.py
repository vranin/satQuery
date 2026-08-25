"""
app/tools/difference.py — Pixel-difference and change-map computation (Stage 9).

Deterministic image processing only — no VLM involved here.
The VLM receives the resulting heatmap/overlay, not raw pixel arrays.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class ChangeMapResult:
    """Result of the pixel-difference computation."""
    difference_array: np.ndarray    # (h, w) float32, values in [0, 1]
    heatmap: Image.Image            # PIL RGB heatmap for VLM display
    overlay: Image.Image            # Difference overlaid on T2
    mean_change: float              # Global mean absolute difference
    max_change: float               # Maximum pixel difference
    changed_fraction: float         # Fraction of pixels above threshold
    threshold_used: float


def _resize_to_match(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    If arrays differ in spatial size, resize *b* to match *a*.
    Both must be (h, w) arrays.
    """
    if a.shape == b.shape:
        return a, b
    h, w = a.shape
    img_b = Image.fromarray((b * 255).astype(np.uint8))
    img_b_resized = img_b.resize((w, h), Image.LANCZOS)
    b_resized = np.array(img_b_resized).astype(np.float32) / 255.0
    logger.warning(
        "Images have different sizes (%s vs %s); T2 resized to match T1.",
        b.shape,
        a.shape,
    )
    return a, b_resized


def _normalize_band(band: np.ndarray) -> np.ndarray:
    """Normalise a 2-D band to [0, 1] using 2–98 percentile clipping."""
    lo = float(np.nanpercentile(band, 2))
    hi = float(np.nanpercentile(band, 98))
    if hi == lo:
        return np.zeros_like(band, dtype=np.float32)
    return np.clip((band.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)


def _array_to_heatmap(diff: np.ndarray) -> Image.Image:
    """
    Convert a (h, w) float32 difference map [0,1] to a false-colour heatmap.

    Colour scheme (intuitive for change detection):
      Low change   → dark blue
      Medium change → yellow / orange
      High change  → bright red
    """
    # Simple jet-like palette via numpy
    r = np.clip(1.5 - np.abs(4 * diff - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * diff - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * diff - 1), 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")


def compute_change_map(
    image_t1: Image.Image,
    image_t2: Image.Image,
    threshold: float = 0.1,
) -> ChangeMapResult:
    """
    Compute a pixel-level difference map between two images.

    This function performs ONLY deterministic image processing.
    The resulting heatmap is then passed to the VLM for interpretation.

    Parameters
    ----------
    image_t1:
        Older image (PIL RGB).
    image_t2:
        Newer image (PIL RGB).
    threshold:
        Pixels with normalised diff above this are counted as "changed".

    Returns
    -------
    ChangeMapResult
    """
    # Convert to float32 grayscale [0,1] for comparison
    arr1 = np.array(image_t1.convert("L")).astype(np.float32) / 255.0
    arr2 = np.array(image_t2.convert("L")).astype(np.float32) / 255.0

    # Align spatial dimensions if needed
    arr1, arr2 = _resize_to_match(arr1, arr2)

    # Absolute difference
    diff = np.abs(arr1 - arr2)  # (h, w) in [0, 1]

    mean_change = float(np.mean(diff))
    max_change = float(np.max(diff))
    changed_fraction = float(np.mean(diff > threshold))

    logger.info(
        "Change map computed: mean=%.4f  max=%.4f  changed_fraction=%.4f  threshold=%.2f",
        mean_change,
        max_change,
        changed_fraction,
        threshold,
    )

    heatmap = _array_to_heatmap(diff)

    # Overlay: blend T2 with heatmap (T2 at 60%, heatmap at 40%)
    t2_resized = image_t2.resize(heatmap.size, Image.LANCZOS).convert("RGB")
    overlay = Image.blend(t2_resized, heatmap, alpha=0.4)

    return ChangeMapResult(
        difference_array=diff,
        heatmap=heatmap,
        overlay=overlay,
        mean_change=mean_change,
        max_change=max_change,
        changed_fraction=changed_fraction,
        threshold_used=threshold,
    )


def save_change_map(result: ChangeMapResult, output_dir: str | Path, stem: str = "change") -> dict[str, str]:
    """
    Save the heatmap and overlay to *output_dir*.

    Returns a dict of {name: relative_path} for inclusion in the API response.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    heatmap_path = out / f"{stem}_heatmap.png"
    overlay_path = out / f"{stem}_overlay.png"

    result.heatmap.save(heatmap_path)
    result.overlay.save(overlay_path)

    logger.info("Saved change map to %s", out)
    return {
        "heatmap": str(heatmap_path),
        "overlay": str(overlay_path),
    }
