"""
app/models/base.py — Abstract VLM interface (Stage 5).

ALL task modules must call this interface.
Never import ollama (or any concrete client) outside of model adapters.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image


class VisionLanguageModel(ABC):
    """
    Abstract base class for all Vision-Language Model backends.

    Concrete implementations live in sibling modules:
      - ollama_client.py  (Ollama / llava, qwen2.5vl, …)
      - hf_client.py      (HuggingFace Transformers — future)
      - remote_client.py  (Cloud endpoint — future)

    The rest of the application only ever calls .generate().
    """

    @abstractmethod
    def generate(
        self,
        images: list[Image.Image],
        prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        """
        Send *prompt* and *images* to the model and return the response text.

        Parameters
        ----------
        images:
            One or more PIL Images, already converted to RGB / appropriate
            colour space by the preprocessing layer.
        prompt:
            The full instruction / question string.
        temperature:
            Sampling temperature (lower = more deterministic).
        max_tokens:
            Maximum tokens in the response.

        Returns
        -------
        str
            Raw text response from the model.
        """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable name / tag of the underlying model."""

    def supports_multiple_images(self) -> bool:
        """
        Return True if this backend can receive >1 image per call.
        Subclasses should override if they have a different limit.
        """
        return True


class VLMError(Exception):
    """Raised when the VLM backend returns an error or is unavailable."""
