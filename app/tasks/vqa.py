"""
app/tasks/vqa.py — Visual Question Answering task (Stages 6 & 7).

answer_vqa(image, question, vlm) → dict

Supports:
  - Binary questions     ("Is there a lake?")
  - Multiple-choice      ("Which of the following…")
  - Open natural language ("Describe the land use.")
"""
from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

from app.models.base import VisionLanguageModel, VLMError
from app.tools.preprocessing import load_image_for_vlm

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Prompt templates
# ------------------------------------------------------------------ #

_SYSTEM_PREAMBLE = (
    "You are a satellite imagery analyst. "
    "You will be shown one or more satellite images and asked a question. "
    "Base your answer only on what is visually observable in the image(s). "
    "Do not speculate or hallucinate information not visible in the image."
)

_VQA_TEMPLATE = """{preamble}

Question: {question}

Provide a concise, factual answer. If you cannot determine the answer from the image, say so clearly."""


def _build_vqa_prompt(question: str) -> str:
    return _VQA_TEMPLATE.format(preamble=_SYSTEM_PREAMBLE, question=question)


# ------------------------------------------------------------------ #
# Main function
# ------------------------------------------------------------------ #

def answer_vqa(
    image: Image.Image | str | Path,
    question: str,
    vlm: VisionLanguageModel,
) -> dict:
    """
    Answer a natural-language question about a satellite image.

    Parameters
    ----------
    image:
        A PIL Image, or a path to a GeoTIFF / PNG / patch directory.
    question:
        The question to answer.
    vlm:
        A VisionLanguageModel instance (injected — not created here).

    Returns
    -------
    dict with keys:
        answer  (str)
        model   (str)
        question (str)
    """
    if not question or not question.strip():
        raise ValueError("Question must not be empty.")

    # Load image if path given
    if isinstance(image, (str, Path)):
        pil_image = load_image_for_vlm(image)
    else:
        pil_image = image.convert("RGB") if image.mode != "RGB" else image

    prompt = _build_vqa_prompt(question.strip())

    logger.info("VQA: model=%s  question=%r", vlm.model_name, question[:80])

    try:
        answer = vlm.generate(images=[pil_image], prompt=prompt, temperature=0.1)
    except VLMError as exc:
        raise RuntimeError(f"VLM error during VQA: {exc}") from exc

    return {
        "answer": answer,
        "model": vlm.model_name,
        "question": question,
    }
