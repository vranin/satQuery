"""tests/test_raster.py — Unit tests for GeoTIFF processing (Stage 15)."""
from __future__ import annotations

import pytest

from app.tools.raster import (
    RasterValidationError,
    prepare_s1,
    prepare_s2,
    read_raster,
    validate_raster,
)


class TestValidateRaster:
    def test_valid_tif(self, s2_tif):
        result = validate_raster(s2_tif)
        assert result == s2_tif

    def test_missing_file_raises(self, tmp_data):
        with pytest.raises(FileNotFoundError):
            validate_raster(tmp_data / "nonexistent.tif")

    def test_corrupt_file_raises(self, corrupt_file):
        with pytest.raises(RasterValidationError):
            validate_raster(corrupt_file)


class TestReadRaster:
    def test_returns_raster_data(self, s2_tif):
        rd = read_raster(s2_tif)
        assert rd.band_count == 4
        assert rd.width == 64
        assert rd.height == 64
        assert rd.array.shape == (4, 64, 64)

    def test_s1_two_bands(self, s1_tif):
        rd = read_raster(s1_tif)
        assert rd.band_count == 2


class TestPrepareS2:
    def test_returns_rgb_image(self, s2_tif):
        img = prepare_s2(s2_tif)
        assert img.mode == "RGB"
        assert img.size == (64, 64)

    def test_single_band_returns_rgb(self, tmp_data):
        import rasterio, numpy as np
        from rasterio.transform import from_bounds
        p = tmp_data / "single_band.tif"
        t = from_bounds(0, 0, 1, 1, 32, 32)
        with rasterio.open(p, "w", driver="GTiff", height=32, width=32,
                           count=1, dtype="uint16", crs="EPSG:4326", transform=t) as dst:
            dst.write(np.ones((1, 32, 32), dtype=np.uint16) * 1000)
        img = prepare_s2(p)
        assert img.mode == "RGB"

    def test_missing_raises(self, tmp_data):
        with pytest.raises(FileNotFoundError):
            prepare_s2(tmp_data / "missing.tif")

    def test_unsupported_band_count_raises(self, tmp_data):
        import numpy as np
        import rasterio
        from rasterio.transform import from_bounds

        p = tmp_data / "too_many_bands.tif"
        transform = from_bounds(0, 0, 1, 1, 8, 8)
        with rasterio.open(p, "w", driver="GTiff", height=8, width=8,
                           count=14, dtype="uint16", transform=transform) as dst:
            dst.write(np.zeros((14, 8, 8), dtype=np.uint16))

        with pytest.raises(RasterValidationError, match="Unsupported Sentinel-2 band count"):
            prepare_s2(p)


class TestPrepareS1:
    def test_returns_rgb_image(self, s1_tif):
        img = prepare_s1(s1_tif)
        assert img.mode == "RGB"
        assert img.size == (64, 64)
