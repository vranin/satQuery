"""Tests for Sentinel-1/Sentinel-2 prompt-level fusion."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from PIL import Image

from app.models.base import VLMError
from app.tasks.fusion import analyze_fusion


def _mock_vlm(response: str = "Both sensors show the same landscape.") -> MagicMock:
    vlm = MagicMock()
    vlm.model_name = "mock-model"
    vlm.supports_multiple_images.return_value = True
    vlm.generate.return_value = response
    return vlm


def test_fusion_returns_structured_result():
    result = analyze_fusion(
        Image.new("RGB", (16, 16)),
        Image.new("RGB", (16, 16)),
        "What features are visible?",
        _mock_vlm(),
    )

    assert result["answer"] == "Both sensors show the same landscape."
    assert result["sensors"] == ["sentinel-1-sar", "sentinel-2-optical"]


def test_fusion_prompt_identifies_both_sensors():
    vlm = _mock_vlm()
    analyze_fusion(Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8)), "Is there water?", vlm)

    prompt = vlm.generate.call_args.kwargs["prompt"]
    assert "Sentinel-1 SAR" in prompt
    assert "Sentinel-2 Optical" in prompt
    assert "BOTH images" in prompt


def test_fusion_rejects_single_image_model():
    vlm = _mock_vlm()
    vlm.supports_multiple_images.return_value = False

    with pytest.raises(RuntimeError, match="does not support multiple images"):
        analyze_fusion(Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8)), "Question?", vlm)


def test_fusion_vlm_error_becomes_runtime_error():
    vlm = _mock_vlm()
    vlm.generate.side_effect = VLMError("unavailable")

    with pytest.raises(RuntimeError, match="VLM error during fusion analysis"):
        analyze_fusion(Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8)), "Question?", vlm)