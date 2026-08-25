"""
app/agent/router.py — Task classifier (Stage 11).

Classifies a (query, image_count, metadata) tuple into one of:
  VQA | CAPTION | CHANGE_DETECTION | CROSS_MODAL

The classification is keyword/heuristic-based for the prototype.
It can later be replaced by a lightweight LLM classifier without
changing any downstream code.
"""
from __future__ import annotations

import re

from app.schemas.requests import TaskType

# ------------------------------------------------------------------ #
# Keyword patterns (lower-cased, regex-ready)
# ------------------------------------------------------------------ #

_CHANGE_PATTERNS = re.compile(
    r"\b(change[ds]?|differ(ence|ent)?|before.and.after|temporal|"
    r"t1.+t2|evolv|transform|modif|what.+(happen|occur)|compar)\b",
    re.IGNORECASE,
)

_FUSION_PATTERNS = re.compile(
    r"\b(sar|radar|sentinel.?1|s1|backscatter|"
    r"both.+(image|sensor)|sensor.+fusion|cross.modal|optical.+sar|sar.+optical)\b",
    re.IGNORECASE,
)

_CAPTION_PATTERNS = re.compile(
    r"\b(descri(be|ption)|caption|explain|summarize|what.+(see|show|there)|"
    r"land.?(use|cover)|scene|overview)\b",
    re.IGNORECASE,
)


def classify_task(
    query: str,
    image_count: int,
    metadata: dict | None = None,
) -> TaskType:
    """
    Determine the appropriate task from the user query and image count.

    Decision tree
    -------------
    1. 2 images + change keywords  → CHANGE_DETECTION
    2. 2 images + SAR/S1 keywords  → CROSS_MODAL
    3. 2 images (no other signal)  → CHANGE_DETECTION (default for 2 images)
    4. 1 image + SAR keywords      → CROSS_MODAL (if metadata says S1+S2)
    5. 1 image + caption keywords  → CAPTION
    6. Otherwise                   → VQA

    Parameters
    ----------
    query:
        Raw natural-language query from the user.
    image_count:
        Number of images uploaded.
    metadata:
        Optional dict, e.g. {"s1": true, "s2": true} from the request.

    Returns
    -------
    TaskType
    """
    q = query.strip().lower()
    meta = metadata or {}

    # -- 2-image path ------------------------------------------------- #
    if image_count >= 2:
        # Explicit SAR/fusion signal → CROSS_MODAL
        if _FUSION_PATTERNS.search(q) or (meta.get("s1") and meta.get("s2")):
            return TaskType.CROSS_MODAL
        # Change keywords or default for 2 images → CHANGE_DETECTION
        return TaskType.CHANGE_DETECTION

    # -- 1-image path ------------------------------------------------- #
    if image_count == 1:
        # SAR/fusion signals without a second image
        if _FUSION_PATTERNS.search(q):
            return TaskType.CROSS_MODAL  # will fail validation downstream
        if _CAPTION_PATTERNS.search(q) and not _is_question(q):
            return TaskType.CAPTION
        return TaskType.VQA

    # -- 0 images ----------------------------------------------------- #
    return TaskType.UNKNOWN


def _is_question(text: str) -> bool:
    """Heuristic: ends with '?' or starts with an interrogative word."""
    if text.rstrip().endswith("?"):
        return True
    interrogatives = ("what", "where", "when", "who", "how", "is", "are", "can", "does")
    return any(text.startswith(w) for w in interrogatives)


def validate_inputs_for_task(task: TaskType, image_count: int) -> str | None:
    """
    Return an error message if the image count is incompatible with the task,
    or None if valid.
    """
    if task == TaskType.UNKNOWN:
        return "No images provided. At least one satellite image is required."

    if task == TaskType.VQA and image_count < 1:
        return "VQA requires at least one image."

    if task == TaskType.CAPTION and image_count < 1:
        return "Captioning requires at least one image."

    if task == TaskType.CHANGE_DETECTION and image_count < 2:
        return (
            "Change detection requires exactly two images (T1 and T2). "
            f"Received {image_count}."
        )

    if task == TaskType.CROSS_MODAL and image_count < 2:
        return (
            "Cross-modal (S1+S2) analysis requires two images: "
            "one Sentinel-1 SAR and one Sentinel-2 optical. "
            f"Received {image_count}."
        )

    return None
