"""Tests for deterministic change detection and VLM interpretation."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from app.models.base import VLMError
from app.tasks.change_detection import detect_changes
from app.tools.difference import compute_change_map


def _mock_vlm(response: str = "A localized change is visible.") -> MagicMock:
    vlm = MagicMock()
    vlm.model_name = "mock-model"
    vlm.generate.return_value = response
    return vlm


def test_identical_images_have_no_change():
    image = Image.new("RGB", (32, 32), "green")
    result = compute_change_map(image, image)

    assert result.mean_change == 0.0
    assert result.max_change == 0.0
    assert result.changed_fraction == 0.0


def test_localized_change_is_detected():
    first = Image.new("RGB", (32, 32), "black")
    second_array = np.zeros((32, 32, 3), dtype=np.uint8)
    second_array[8:16, 8:16] = 255
    second = Image.fromarray(second_array, mode="RGB")

    result = compute_change_map(first, second, threshold=0.1)

    assert result.max_change == 1.0
    assert 0.05 < result.changed_fraction < 0.1


def test_different_sizes_are_aligned():
    result = compute_change_map(Image.new("RGB", (32, 32)), Image.new("RGB", (16, 16)))

    assert result.difference_array.shape == (32, 32)
    assert result.heatmap.size == (32, 32)


def test_detect_changes_returns_evidence_and_saves_maps(tmp_path: Path):
    result = detect_changes(
        Image.new("RGB", (32, 32), "black"),
        Image.new("RGB", (32, 32), "white"),
        _mock_vlm(),
        output_dir=tmp_path,
    )

    assert result["description"] == "A localized change is visible."
    assert result["stats"]["changed_fraction"] == 1.0
    assert Path(result["change_map"]["heatmap"]).exists()
    assert Path(result["change_map"]["overlay"]).exists()


def test_detect_changes_vlm_error_becomes_runtime_error(tmp_path: Path):
    vlm = _mock_vlm()
    vlm.generate.side_effect = VLMError("unavailable")

    with pytest.raises(RuntimeError, match="VLM error during change interpretation"):
        detect_changes(Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8)), vlm, tmp_path)