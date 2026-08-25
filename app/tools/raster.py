"""
app/tools/raster.py — GeoTIFF I/O and satellite image preprocessing (Stage 4).

Provides:
  read_raster(path)   — open a GeoTIFF and return metadata + array
  prepare_s2(path)    — Sentinel-2 multispectral → RGB PIL Image
  prepare_s1(path)    — Sentinel-1 SAR → grayscale/pseudo-color PIL Image
  validate_raster(path) — raises descriptive errors on bad files
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image
from rasterio.errors import RasterioIOError

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Constants
# ------------------------------------------------------------------ #

# Sentinel-2 band order when all 13 bands are present (1-indexed names).
# We only need B04=Red, B03=Green, B02=Blue for a natural-colour RGB.
S2_BAND_NAMES = [
    "B01", "B02", "B03", "B04", "B05",
    "B06", "B07", "B08", "B8A", "B09",
    "B10", "B11", "B12",
]
# 0-indexed positions of B04, B03, B02 in a 13-band stack
S2_RGB_BANDS = (3, 2, 1)  # indices into band axis

# BigEarthNet per-band TIF files are named like: {patch}_B04.tif
S2_RGB_SUFFIXES = ("B04", "B03", "B02")  # Red, Green, Blue

# Percentile clipping for display normalisation
DISPLAY_P_LOW = 2
DISPLAY_P_HIGH = 98


# ------------------------------------------------------------------ #
# Data class
# ------------------------------------------------------------------ #

@dataclass
class RasterData:
    """Holds a raster array and its metadata."""
    path: Path
    array: np.ndarray          # shape: (bands, height, width)
    meta: dict[str, Any]
    crs: Any
    transform: Any
    band_count: int
    width: int
    height: int
    tags: dict[str, str] = field(default_factory=dict)


# ------------------------------------------------------------------ #
# Validation
# ------------------------------------------------------------------ #

class RasterValidationError(ValueError):
    """Raised when a raster file fails validation."""


def validate_raster(path: str | Path) -> Path:
    """
    Validate that *path* points to a readable GeoTIFF.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    RasterValidationError
        If the file cannot be opened or is corrupt.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Raster file not found: {p}")
    if not p.is_file():
        raise RasterValidationError(f"Path is not a file: {p}")

    try:
        with rasterio.open(p) as src:
            if src.count == 0:
                raise RasterValidationError(f"Raster has 0 bands: {p}")
            # Try reading a 1-pixel window to detect corruption
            src.read(1, window=rasterio.windows.Window(0, 0, 1, 1))
    except RasterioIOError as exc:
        raise RasterValidationError(f"Cannot open raster (corrupt or unsupported): {p} — {exc}") from exc

    return p


# ------------------------------------------------------------------ #
# Core reader
# ------------------------------------------------------------------ #

def read_raster(path: str | Path) -> RasterData:
    """
    Open a GeoTIFF and return a RasterData object.

    Parameters
    ----------
    path:
        Path to the GeoTIFF file.

    Returns
    -------
    RasterData
        Structured container with the full array and metadata.
    """
    p = validate_raster(path)

    with rasterio.open(p) as src:
        array = src.read().astype(np.float32)  # (bands, h, w)
        return RasterData(
            path=p,
            array=array,
            meta=src.meta.copy(),
            crs=src.crs,
            transform=src.transform,
            band_count=src.count,
            width=src.width,
            height=src.height,
            tags=dict(src.tags()),
        )


# ------------------------------------------------------------------ #
# Normalisation helpers
# ------------------------------------------------------------------ #

def _percentile_normalize(band: np.ndarray) -> np.ndarray:
    """
    Stretch a 2-D band to [0, 255] using percentile clipping.
    Handles constant bands (all NaN or flat) gracefully.
    """
    lo = float(np.nanpercentile(band, DISPLAY_P_LOW))
    hi = float(np.nanpercentile(band, DISPLAY_P_HIGH))
    if hi == lo:
        return np.zeros_like(band, dtype=np.uint8)
    clipped = np.clip(band, lo, hi)
    scaled = (clipped - lo) / (hi - lo) * 255.0
    return scaled.astype(np.uint8)


# ------------------------------------------------------------------ #
# Sentinel-2 preprocessing
# ------------------------------------------------------------------ #

def prepare_s2(path: str | Path) -> Image.Image:
    """
    Convert a Sentinel-2 GeoTIFF to a VLM-ready RGB PIL Image.

    Strategy
    --------
    * If the file has >= 4 bands (stacked), use bands 4, 3, 2 (B04/B03/B02).
    * If the file has exactly 1 band, treat it as a single-band display image
      (e.g. a BigEarthNet individual band file).
    * If the file has 2–3 bands, use whatever is present.

    All bands are percentile-normalised for display.

    Parameters
    ----------
    path:
        Path to the Sentinel-2 GeoTIFF.

    Returns
    -------
    PIL.Image.Image
        RGB image ready for the VLM.
    """
    p = validate_raster(path)
    raster = read_raster(p)
    bands, h, w = raster.array.shape

    logger.debug("prepare_s2: %s  bands=%d  size=%dx%d", p.name, bands, w, h)

    if bands > len(S2_BAND_NAMES):
        raise RasterValidationError(
            f"Unsupported Sentinel-2 band count {bands}; expected 1-{len(S2_BAND_NAMES)}"
        )

    if bands >= 4:
        r = _percentile_normalize(raster.array[S2_RGB_BANDS[0]])
        g = _percentile_normalize(raster.array[S2_RGB_BANDS[1]])
        b = _percentile_normalize(raster.array[S2_RGB_BANDS[2]])
        rgb = np.stack([r, g, b], axis=-1)  # (h, w, 3)
    elif bands == 3:
        r = _percentile_normalize(raster.array[0])
        g = _percentile_normalize(raster.array[1])
        b = _percentile_normalize(raster.array[2])
        rgb = np.stack([r, g, b], axis=-1)
    elif bands == 2:
        ch0 = _percentile_normalize(raster.array[0])
        ch1 = _percentile_normalize(raster.array[1])
        zero = np.zeros_like(ch0)
        rgb = np.stack([ch0, ch1, zero], axis=-1)
    else:  # bands == 1
        gray = _percentile_normalize(raster.array[0])
        rgb = np.stack([gray, gray, gray], axis=-1)

    return Image.fromarray(rgb, mode="RGB")


# ------------------------------------------------------------------ #
# Sentinel-1 preprocessing
# ------------------------------------------------------------------ #

def prepare_s1(path: str | Path) -> Image.Image:
    """
    Convert a Sentinel-1 SAR GeoTIFF to a VLM-ready PIL Image.

    Strategy
    --------
    SAR backscatter is NOT ordinary RGB data.  We apply:

    1. Log-scale conversion (dB) from linear amplitude/power values.
    2. Percentile normalisation.
    3. If 2 bands (VV+VH), create a pseudo-colour composite:
         R = VV, G = VH, B = VV-VH ratio → highlights flooded areas.
    4. If 1 band, produce a grayscale image (replicated to RGB).

    The VLM prompt will explicitly identify this as SAR imagery so
    the model does not misinterpret the visual representation.

    Parameters
    ----------
    path:
        Path to the Sentinel-1 GeoTIFF.

    Returns
    -------
    PIL.Image.Image
        Pseudo-colour or grayscale RGB image for the VLM.
    """
    p = validate_raster(path)
    raster = read_raster(p)
    bands, h, w = raster.array.shape

    logger.debug("prepare_s1: %s  bands=%d  size=%dx%d", p.name, bands, w, h)

    def to_db(arr: np.ndarray) -> np.ndarray:
        """Convert linear amplitude to dB, clamp negatives."""
        arr_clipped = np.clip(arr, 1e-6, None)  # avoid log(0)
        return 10.0 * np.log10(arr_clipped)

    if bands >= 2:
        vv_db = to_db(raster.array[0])
        vh_db = to_db(raster.array[1])
        ratio = vv_db - vh_db  # highlights water / flooded areas

        r = _percentile_normalize(vv_db)
        g = _percentile_normalize(vh_db)
        b = _percentile_normalize(ratio)
        rgb = np.stack([r, g, b], axis=-1)
    else:
        single_db = to_db(raster.array[0])
        gray = _percentile_normalize(single_db)
        rgb = np.stack([gray, gray, gray], axis=-1)

    return Image.fromarray(rgb, mode="RGB")


# ------------------------------------------------------------------ #
# BigEarthNet multi-file helper
# ------------------------------------------------------------------ #

def load_s2_rgb_from_patch_dir(patch_dir: str | Path) -> Image.Image:
    """
    Load an RGB image from a BigEarthNet Sentinel-2 patch directory.

    BigEarthNet stores each band as a separate file, e.g.:
      S2A_MSIL2A_20170613T101031_0_45/
        S2A_MSIL2A_20170613T101031_0_45_B04.tif   ← Red
        S2A_MSIL2A_20170613T101031_0_45_B03.tif   ← Green
        S2A_MSIL2A_20170613T101031_0_45_B02.tif   ← Blue

    Parameters
    ----------
    patch_dir:
        Path to the patch sub-directory.

    Returns
    -------
    PIL.Image.Image
        RGB image.

    Raises
    ------
    FileNotFoundError
        If any of B04, B03, or B02 files are missing.
    """
    d = Path(patch_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"Patch directory not found: {d}")

    channels = []
    for suffix in S2_RGB_SUFFIXES:
        matches = list(d.glob(f"*_{suffix}.tif"))
        if not matches:
            raise FileNotFoundError(
                f"BigEarthNet band file '*_{suffix}.tif' not found in {d}"
            )
        band_path = matches[0]
        raster = read_raster(band_path)
        if raster.band_count != 1:
            raise RasterValidationError(
                f"Expected one band in BigEarthNet file {band_path.name}, "
                f"found {raster.band_count}"
            )
        if channels and raster.array.shape[1:] != channels[0].shape:
            raise RasterValidationError(
                "Sentinel-2 RGB bands have incompatible dimensions in "
                f"{d}: expected {channels[0].shape}, "
                f"found {raster.array.shape[1:]}"
            )
        channels.append(_percentile_normalize(raster.array[0]))

    rgb = np.stack(channels, axis=-1)  # (h, w, 3)
    return Image.fromarray(rgb, mode="RGB")
