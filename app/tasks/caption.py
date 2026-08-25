"""
app/tasks/caption.py — Image captioning task (Stage 8).

generate_caption(image, vlm) → dict

Requests a structured scene description covering land use, major objects,
vegetation, water, settlement, terrain, and season/climate.
"""
from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

from app.models.base import VisionLanguageModel, VLMError
from app.tools.preprocessing import load_image_for_vlm

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Prompt
# ------------------------------------------------------------------ #

_CAPTION_PROMPT = """You are a satellite imagery analyst with expertise in remote sensing and land cover classification.

Analyse the provided satellite image and produce a structured description. Cover the following aspects if they are visually inferable:

1. Land use / land cover (e.g. agricultural, urban, forest, water, bare soil)
2. Major objects or features (e.g. roads, buildings, water bodies, crop fields)
3. Spatial relationships and layout
4. Vegetation (type, density, condition if visible)
5. Water bodies (presence, approximate extent)
6. Settlement or infrastructure (if present)
7. Terrain / topography (if visible)
8. Season or climate context (if visually determinable)

Important:
- Only describe what is visually observable.
- Do not speculate about things not visible in the image.
- Be factual and concise.
- Structure your response clearly."""


# ------------------------------------------------------------------ #
# Main function
# ------------------------------------------------------------------ #

def generate_caption(
    image: Image.Image | str | Path,
    vlm: VisionLanguageModel,
) -> dict:
    """
    Generate a structured scene caption for a satellite image.

    Parameters
    ----------
    image:
        PIL Image or path to GeoTIFF / PNG / BigEarthNet patch directory.
    vlm:
        Injected VisionLanguageModel instance.

    Returns
    -------
    dict with keys:
        caption (str)
        model   (str)
    """
    if isinstance(image, (str, Path)):
        pil_image = load_image_for_vlm(image)
    else:
        pil_image = image.convert("RGB") if image.mode != "RGB" else image

    logger.info("Caption: model=%s", vlm.model_name)

    try:
        caption = vlm.generate(
            images=[pil_image],
            prompt=_CAPTION_PROMPT,
            temperature=0.2,
            max_tokens=768,
        )
    except VLMError as exc:
        raise RuntimeError(f"VLM error during captioning: {exc}") from exc

    return {
        "caption": caption,
        "model": vlm.model_name,
    }
