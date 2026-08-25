"""
app/tasks/fusion.py — Sentinel-1 + Sentinel-2 cross-modal analysis (Stage 10).

analyze_fusion(s1_image, s2_image, question, vlm) → dict

Uses prompt-level fusion: both sensor representations are sent to the VLM
with explicit labels so the model can reason across modalities.
"""
from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

from app.models.base import VisionLanguageModel, VLMError
from app.tools.preprocessing import load_image_for_vlm
from app.tools.raster import prepare_s1, prepare_s2, validate_raster

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Prompt templates
# ------------------------------------------------------------------ #

_FUSION_PREAMBLE = """You are an expert in multi-sensor satellite image analysis.

You are provided with TWO images of the same geographic area:
  Image 1: Sentinel-1 SAR (Synthetic Aperture Radar)
    - Measures microwave backscatter
    - Can penetrate clouds; works day and night
    - Bright areas = high backscatter (rough/dense surfaces, buildings, water)
    - Dark areas = low backscatter (calm water, smooth/flat surfaces)

  Image 2: Sentinel-2 Optical
    - Multispectral visible/near-infrared
    - Intuitive colour representation of the scene
    - Sensitive to vegetation, soil, and urban colours

Use information from BOTH images to answer the question below.
Do not base your answer on just one sensor. Synthesise observations from both.
Base your answer only on what is visually observable."""

_FUSION_PROMPT_TEMPLATE = """{preamble}

Question: {question}

Provide a concise, factual answer. Explicitly mention evidence from both the SAR and optical images where relevant."""


def _build_fusion_prompt(question: str) -> str:
    return _FUSION_PROMPT_TEMPLATE.format(preamble=_FUSION_PREAMBLE, question=question)


# ------------------------------------------------------------------ #
# Main function
# ------------------------------------------------------------------ #

def analyze_fusion(
    s1_image: Image.Image | str | Path,
    s2_image: Image.Image | str | Path,
    question: str,
    vlm: VisionLanguageModel,
) -> dict:
    """
    Answer a question using both Sentinel-1 and Sentinel-2 imagery.

    Parameters
    ----------
    s1_image:
        Sentinel-1 SAR image (PIL or path to GeoTIFF).
    s2_image:
        Sentinel-2 optical image (PIL or path to GeoTIFF).
    question:
        Natural-language question to answer.
    vlm:
        Injected VisionLanguageModel.

    Returns
    -------
    dict with keys:
        answer   (str)
        model    (str)
        question (str)
        sensors  (list[str])
    """
    if not question or not question.strip():
        raise ValueError("Question must not be empty.")

    # -- Load S1 ------------------------------------------------------ #
    if isinstance(s1_image, (str, Path)):
        p1 = Path(s1_image)
        validate_raster(p1)
        img_s1 = prepare_s1(p1)
        logger.info("Fusion: loaded S1 from %s", p1.name)
    else:
        img_s1 = s1_image.convert("RGB")

    # -- Load S2 ------------------------------------------------------ #
    if isinstance(s2_image, (str, Path)):
        p2 = Path(s2_image)
        validate_raster(p2)
        img_s2 = prepare_s2(p2)
        logger.info("Fusion: loaded S2 from %s", p2.name)
    else:
        img_s2 = s2_image.convert("RGB")

    # -- VLM call with both images ------------------------------------ #
    prompt = _build_fusion_prompt(question.strip())
    logger.info("Fusion VLM call: model=%s  question=%r", vlm.model_name, question[:80])

    if not vlm.supports_multiple_images():
        raise RuntimeError(
            f"Model '{vlm.model_name}' does not support multiple images. "
            "Cross-modal fusion requires a multi-image capable VLM."
        )

    try:
        answer = vlm.generate(
            images=[img_s1, img_s2],
            prompt=prompt,
            temperature=0.15,
            max_tokens=512,
        )
    except VLMError as exc:
        raise RuntimeError(f"VLM error during fusion analysis: {exc}") from exc

    return {
        "answer": answer,
        "model": vlm.model_name,
        "question": question,
        "sensors": ["sentinel-1-sar", "sentinel-2-optical"],
    }
