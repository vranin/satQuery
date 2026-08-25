"""Tests for the model abstraction and Ollama adapter boundary."""
from __future__ import annotations

import base64
from types import SimpleNamespace

from PIL import Image

from app.models.ollama_client import _pil_to_base64, _response_text


def test_response_text_supports_sdk_object():
    assert _response_text(SimpleNamespace(response="Generated answer")) == "Generated answer"


def test_response_text_supports_mapping():
    assert _response_text({"response": "Generated answer"}) == "Generated answer"


def test_pil_image_is_encoded_as_png():
    encoded = _pil_to_base64(Image.new("RGBA", (2, 2), (255, 0, 0, 255)))
    assert base64.b64decode(encoded).startswith(b"\x89PNG\r\n\x1a\n")