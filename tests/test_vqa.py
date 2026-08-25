"""tests/test_vqa.py — Unit tests for the VQA task (VLM mocked)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from PIL import Image

from app.tasks.vqa import answer_vqa


def _mock_vlm(response: str = "Yes, there is a water body.") -> MagicMock:
    vlm = MagicMock()
    vlm.model_name = "mock-model"
    vlm.generate.return_value = response
    return vlm


class TestAnswerVqa:
    def test_basic_vqa_with_pil_image(self, s2_tif):
        from app.tools.preprocessing import load_image_for_vlm
        img = load_image_for_vlm(s2_tif)
        vlm = _mock_vlm("Yes.")
        result = answer_vqa(img, "Is there water?", vlm)
        assert result["answer"] == "Yes."
        assert result["model"] == "mock-model"
        assert result["question"] == "Is there water?"

    def test_vqa_with_path(self, s2_tif):
        vlm = _mock_vlm("No water visible.")
        result = answer_vqa(s2_tif, "Is there water?", vlm)
        assert "answer" in result

    def test_empty_question_raises(self, s2_tif):
        vlm = _mock_vlm()
        with pytest.raises(ValueError, match="Question must not be empty"):
            answer_vqa(s2_tif, "   ", vlm)

    def test_vlm_called_once(self, s2_tif):
        vlm = _mock_vlm("Maybe.")
        answer_vqa(s2_tif, "What is shown?", vlm)
        vlm.generate.assert_called_once()

    def test_vlm_error_propagates(self, s2_tif):
        from app.models.base import VLMError
        vlm = MagicMock()
        vlm.model_name = "mock-model"
        vlm.generate.side_effect = VLMError("Connection refused")
        with pytest.raises(RuntimeError, match="VLM error"):
            answer_vqa(s2_tif, "What is this?", vlm)

    def test_missing_image_raises(self, tmp_data):
        vlm = _mock_vlm()
        with pytest.raises(FileNotFoundError):
            answer_vqa(tmp_data / "missing.tif", "Question?", vlm)
