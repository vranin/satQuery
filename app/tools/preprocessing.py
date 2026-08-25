"""
app/tools/preprocessing.py — Image preprocessing utilities (Stage 4).

Wraps raster.py operations for use by task modules.
Handles: resizing, format conversion, PIL → bytes, validation.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path

from PIL import Image

from app.tools.raster import (
    RasterValidationError,
    load_s2_rgb_from_patch_dir,
    prepare_s1,
    prepare_s2,
    validate_raster,
)

logger = logging.getLogger(__name__)

# Maximum dimension we send to the VLM (pixels).
# Keeps inference time reasonable on CPU.
VLM_MAX_DIM = 1024


def resize_for_vlm(image: Image.Image, max_dim: int = VLM_MAX_DIM) -> Image.Image:
    """
    Resize *image* so its longest side is <= *max_dim*, preserving aspect ratio.
    Returns the original image if it is already small enough.
    """
    w, h = image.size
    if max(w, h) <= max_dim:
        return image
    scale = max_dim / max(w, h)
    new_size = (int(w * scale), int(h * scale))
    return image.resize(new_size, Image.LANCZOS)


def image_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    """Encode a PIL Image to bytes (default PNG)."""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


def load_image_for_vlm(path: str | Path) -> Image.Image:
    """
    Load any supported satellite image file and return a VLM-ready PIL Image.

    Decision logic:
    - Ends with .tif / .tiff  → detect S1 vs S2 by band count / filename
    - Regular image (png/jpg) → open directly
    - Directory               → treat as BigEarthNet S2 patch directory

    Parameters
    ----------
    path:
        Path to a GeoTIFF, PNG/JPG, or BigEarthNet patch directory.

    Returns
    -------
    PIL.Image.Image
        RGB image resized to VLM_MAX_DIM.
    """
    p = Path(path)

    if p.is_dir():
        logger.info("Loading BigEarthNet patch directory: %s", p)
        img = load_s2_rgb_from_patch_dir(p)
        return resize_for_vlm(img)

    suffix = p.suffix.lower()

    if suffix in (".tif", ".tiff"):
        validate_raster(p)
        # Heuristic: S1 files often contain "S1" or have 2 bands (VV, VH).
        name_upper = p.stem.upper()
        if "S1" in name_upper or "_VV" in name_upper or "_VH" in name_upper:
            logger.info("Detected Sentinel-1, using SAR preprocessing: %s", p.name)
            img = prepare_s1(p)
        else:
            logger.info("Treating as Sentinel-2 / generic GeoTIFF: %s", p.name)
            img = prepare_s2(p)
        return resize_for_vlm(img)

    # Fallback: ordinary image formats
    if suffix in (".png", ".jpg", ".jpeg", ".webp"):
        img = Image.open(p).convert("RGB")
        return resize_for_vlm(img)

    raise RasterValidationError(
        f"Unsupported file type '{suffix}'. "
        "Supported: .tif, .tiff, .png, .jpg, .jpeg, .webp, or a patch directory."
    )
