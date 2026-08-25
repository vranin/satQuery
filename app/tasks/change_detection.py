"""
app/tasks/change_detection.py — Bi-temporal change detection task (Stage 9).

detect_changes(image_t1, image_t2, vlm, output_dir) → dict

Pipeline:
  1. Load + preprocess both images (deterministic)
  2. Compute pixel-difference map (deterministic — tools/difference.py)
  3. Send change evidence + heatmap to VLM for natural-language interpretation
  4. Return structured result that clearly separates evidence from interpretation
"""
from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

from app.config import get_settings
from app.models.base import VisionLanguageModel, VLMError
from app.tools.difference import ChangeMapResult, compute_change_map, save_change_map
from app.tools.change_detector import ChangeDetector, get_change_detector
from app.tools.preprocessing import load_image_for_vlm

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# VLM prompt for change interpretation
# ------------------------------------------------------------------ #

_CHANGE_PROMPT_TEMPLATE = """You are a satellite imagery analyst specialising in change detection.

You have been provided with:
  Image 1: A heatmap showing PIXEL-LEVEL DIFFERENCES between two satellite images.
           Warm colours (red/orange/yellow) = higher change.
           Cool colours (blue) = little or no change.

  Image 2: The original overlay of the difference on the newer image.

Computed statistics (from deterministic image processing — not from you):
  - Mean change magnitude : {mean_change:.4f}  (scale 0–1)
  - Maximum change        : {max_change:.4f}
  - Changed pixel fraction: {changed_fraction:.1%}  (threshold = {threshold:.2f})

Based on this evidence, describe:
1. Where the most significant changes appear to be located (spatial pattern).
2. What type of change may have occurred (e.g. deforestation, urban growth, flooding, crop harvest).
3. The severity of the change observed.
4. Any caution or uncertainty in your interpretation.

Important:
- You are INTERPRETING computational evidence, not computing changes yourself.
- Do not claim changes exist if the statistics show very low change.
- Be explicit about uncertainty."""


def _build_change_prompt(result: ChangeMapResult) -> str:
    return _CHANGE_PROMPT_TEMPLATE.format(
        mean_change=result.mean_change,
        max_change=result.max_change,
        changed_fraction=result.changed_fraction,
        threshold=result.threshold_used,
    )


# ------------------------------------------------------------------ #
# Main function
# ------------------------------------------------------------------ #

def detect_changes(
    image_t1: Image.Image | str | Path,
    image_t2: Image.Image | str | Path,
    vlm: VisionLanguageModel,
    output_dir: str | Path | None = None,
    threshold: float = 0.1,
    detector: ChangeDetector | None = None,
) -> dict:
    """
    Perform bi-temporal change detection.

    Parameters
    ----------
    image_t1:
        Earlier image (PIL or path).
    image_t2:
        Later image (PIL or path).
    vlm:
        Injected VisionLanguageModel.
    output_dir:
        Where to save heatmap/overlay PNGs.  Defaults to settings.output_path.
    threshold:
        Pixel-difference threshold for "changed" classification.

    Returns
    -------
    dict with keys:
        description      (str)  — VLM interpretation
        change_map       (dict) — paths to saved heatmap and overlay
        stats            (dict) — computed change statistics
        model            (str)
    """
    settings = get_settings()
    out_dir = Path(output_dir) if output_dir else settings.output_path

    # -- Load images -------------------------------------------------- #
    if isinstance(image_t1, (str, Path)):
        img1 = load_image_for_vlm(image_t1)
    else:
        img1 = image_t1.convert("RGB")

    if isinstance(image_t2, (str, Path)):
        img2 = load_image_for_vlm(image_t2)
    else:
        img2 = image_t2.convert("RGB")

    # -- Deterministic change computation ----------------------------- #
    logger.info("Computing pixel-difference change map…")
    selected_detector = detector or get_change_detector()
    change_result = selected_detector.detect(img1, img2, threshold)
    saved_paths = save_change_map(change_result, out_dir)

    # -- VLM interpretation ------------------------------------------- #
    prompt = _build_change_prompt(change_result)
    logger.info("Change detection VLM call: model=%s", vlm.model_name)

    try:
        description = vlm.generate(
            images=[change_result.heatmap, change_result.overlay],
            prompt=prompt,
            temperature=0.15,
            max_tokens=512,
        )
    except VLMError as exc:
        raise RuntimeError(f"VLM error during change interpretation: {exc}") from exc

    return {
        "description": description,
        "change_map": saved_paths,
        "stats": {
            "mean_change": round(change_result.mean_change, 4),
            "max_change": round(change_result.max_change, 4),
            "changed_fraction": round(change_result.changed_fraction, 4),
            "threshold": threshold,
        },
        "model": vlm.model_name,
        "detector": selected_detector.name,
    }
