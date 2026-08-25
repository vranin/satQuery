"""Tests for the satellite image captioning task."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.models.base import VLMError
from app.tasks.caption import generate_caption
from app.tools.preprocessing import load_image_for_vlm


def _mock_vlm(response: str = "A forested area with visible field boundaries.") -> MagicMock:
    vlm = MagicMock()
    vlm.model_name = "mock-model"
    vlm.generate.return_value = response
    return vlm


def test_caption_from_raster_returns_structured_result(s2_tif):
    result = generate_caption(s2_tif, _mock_vlm())

    assert result == {
        "caption": "A forested area with visible field boundaries.",
        "model": "mock-model",
    }


def test_caption_accepts_pil_image(s2_tif):
    vlm = _mock_vlm("Urban settlement near vegetation.")
    result = generate_caption(load_image_for_vlm(s2_tif), vlm)

    assert result["caption"] == "Urban settlement near vegetation."
    vlm.generate.assert_called_once()


def test_caption_prompt_requests_remote_sensing_details(s2_tif):
    vlm = _mock_vlm()
    generate_caption(s2_tif, vlm)

    prompt = vlm.generate.call_args.kwargs["prompt"]
    assert "Land use / land cover" in prompt
    assert "Spatial relationships" in prompt
    assert "Only describe what is visually observable" in prompt


def test_caption_vlm_error_becomes_runtime_error(s2_tif):
    vlm = _mock_vlm()
    vlm.generate.side_effect = VLMError("Connection refused")

    with pytest.raises(RuntimeError, match="VLM error during captioning"):
        generate_caption(s2_tif, vlm)