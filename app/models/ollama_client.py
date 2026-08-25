"""
app/models/ollama_client.py — Ollama VLM adapter (Stage 5).

Implements VisionLanguageModel using the ollama Python SDK.
Only this file is allowed to import ollama.
"""
from __future__ import annotations

import base64
import io
import logging
from typing import Any

import ollama as _ollama
from PIL import Image

from app.config import get_settings
from app.models.base import VisionLanguageModel, VLMError

logger = logging.getLogger(__name__)


def _pil_to_base64(image: Image.Image) -> str:
    """Encode a PIL Image as base64 PNG string for Ollama."""
    buf = io.BytesIO()
    # Always convert to RGB before encoding — Ollama cannot handle RGBA/L modes.
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _response_text(response: Any) -> str:
    """Extract generated text from dict and SDK response-object variants."""
    if isinstance(response, dict):
        text = response.get("response")
    else:
        text = getattr(response, "response", None)
    if not isinstance(text, str):
        raise VLMError("Ollama returned a response without text.")
    return text


class OllamaVLM(VisionLanguageModel):
    """
    Ollama-backed Vision-Language Model.

    Uses the Ollama Python SDK to communicate with a locally running
    Ollama server.  Change OLLAMA_BASE_URL / OLLAMA_MODEL in .env to
    point at a different host or model.
    """

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        settings = get_settings()
        self._model = model or settings.ollama_model
        self._base_url = base_url or settings.ollama_base_url
        self._client = _ollama.Client(host=self._base_url)
        logger.info("OllamaVLM initialised: model=%s, host=%s", self._model, self._base_url)

    # ------------------------------------------------------------------ #
    # VisionLanguageModel interface
    # ------------------------------------------------------------------ #

    @property
    def model_name(self) -> str:
        return self._model

    def generate(
        self,
        images: list[Image.Image],
        prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        """
        Send images + prompt to Ollama and return the response text.

        Raises
        ------
        VLMError
            If Ollama is unreachable or returns an error.
        """
        if not images:
            raise VLMError("At least one image is required.")

        encoded = [_pil_to_base64(img) for img in images]

        logger.debug(
            "Calling Ollama model=%s with %d image(s), prompt length=%d",
            self._model,
            len(images),
            len(prompt),
        )

        try:
            response: Any = self._client.generate(
                model=self._model,
                prompt=prompt,
                images=encoded,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            )
            text = _response_text(response)
            logger.debug("Ollama response: %s", text[:200])
            return text.strip()

        except _ollama.ResponseError as exc:
            raise VLMError(f"Ollama model error: {exc}") from exc
        except Exception as exc:
            raise VLMError(f"Ollama unavailable or unexpected error: {exc}") from exc

    def supports_multiple_images(self) -> bool:
        # llava:7b and qwen2.5vl both support multiple images in a single call.
        return True

    # ------------------------------------------------------------------ #
    # Utility
    # ------------------------------------------------------------------ #

    def ping(self) -> bool:
        """
        Return True if the Ollama server is reachable and the model is loaded.
        Use this for health checks.
        """
        try:
            models = self._client.list()
            names = [m["model"] for m in models.get("models", [])]
            return self._model in names
        except Exception:
            return False


def get_vlm() -> VisionLanguageModel:
    """
    Factory function — returns the configured VLM backend.

    To swap the backend, change VLM_BACKEND in .env and add a new
    branch here.  No other code changes are required.
    """
    settings = get_settings()
    backend = settings.vlm_backend.lower()

    if backend == "ollama":
        return OllamaVLM()
    # Future backends:
    # elif backend == "huggingface":
    #     from app.models.hf_client import HuggingFaceVLM
    #     return HuggingFaceVLM()
    # elif backend == "remote":
    #     from app.models.remote_client import RemoteVLM
    #     return RemoteVLM()
    else:
        raise ValueError(f"Unknown VLM backend: '{backend}'. Set VLM_BACKEND in .env.")
