"""
tests/conftest.py — Shared pytest fixtures.

Creates synthetic GeoTIFF files in a temp directory so tests
don't require any real satellite data.
"""
from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from PIL import Image


# ------------------------------------------------------------------ #
# Synthetic GeoTIFF factory
# ------------------------------------------------------------------ #

def _make_geotiff(path, bands: int = 4, width: int = 64, height: int = 64) -> None:
    """Write a minimal synthetic GeoTIFF for testing."""
    transform = from_bounds(0, 0, 1, 1, width, height)
    data = np.random.randint(0, 3000, (bands, height, width), dtype=np.uint16)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype="uint16",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)


@pytest.fixture(scope="session")
def tmp_data(tmp_path_factory):
    """Session-scoped temp directory with synthetic test images."""
    base = tmp_path_factory.mktemp("satquery_test_data")
    return base


@pytest.fixture(scope="session")
def s2_tif(tmp_data):
    """Synthetic 4-band Sentinel-2 GeoTIFF."""
    p = tmp_data / "S2_test.tif"
    _make_geotiff(p, bands=4)
    return p


@pytest.fixture(scope="session")
def s2_tif_t2(tmp_data):
    """Second synthetic Sentinel-2 GeoTIFF (slightly different for change detection)."""
    p = tmp_data / "S2_test_t2.tif"
    _make_geotiff(p, bands=4)
    return p


@pytest.fixture(scope="session")
def s1_tif(tmp_data):
    """Synthetic 2-band Sentinel-1 GeoTIFF."""
    p = tmp_data / "S1_VV_VH_test.tif"
    _make_geotiff(p, bands=2)
    return p


@pytest.fixture(scope="session")
def rgb_png(tmp_data):
    """Plain RGB PNG for testing non-GeoTIFF path."""
    p = tmp_data / "test_rgb.png"
    img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    img.save(p)
    return p


@pytest.fixture(scope="session")
def corrupt_file(tmp_data):
    """A file with a .tif extension but corrupt content."""
    p = tmp_data / "corrupt.tif"
    p.write_bytes(b"THIS IS NOT A GEOTIFF")
    return p
