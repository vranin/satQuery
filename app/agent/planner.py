"""
app/agent/planner.py — Pipeline orchestrator (Stage 11).

run_pipeline(query, images, metadata) → AnalysisResponse

Orchestrates: router → task → trace → structured response.
"""
from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

from app.agent.router import classify_task, validate_inputs_for_task
from app.agent.trace import TraceBuilder
from app.models.ollama_client import get_vlm
from app.schemas.requests import (
    AnalysisEvidence,
    AnalysisResponse,
    TaskStatus,
    TaskType,
)
from app.tasks.caption import generate_caption
from app.tasks.change_detection import detect_changes
from app.tasks.fusion import analyze_fusion
from app.tasks.vqa import answer_vqa
from app.tools.preprocessing import load_image_for_vlm

logger = logging.getLogger(__name__)


def run_pipeline(
    query: str,
    image_paths: list[str | Path],
    metadata: dict | None = None,
) -> AnalysisResponse:
    """
    Main entry point for the agent.

    Parameters
    ----------
    query:
        Natural-language query from the user.
    image_paths:
        List of paths to uploaded image files.
    metadata:
        Optional dict with extra context (e.g. {"s1": true, "s2": true}).

    Returns
    -------
    AnalysisResponse
        Fully structured response with answer, evidence, and trace.
    """
    image_count = len(image_paths)
    task = classify_task(query, image_count, metadata)
    trace = TraceBuilder(task=task, inputs=image_count)
    evidence: list[AnalysisEvidence] = []

    logger.info("Pipeline: task=%s  images=%d  query=%r", task, image_count, query[:80])

    # -- Input validation -------------------------------------------- #
    validation_error = validate_inputs_for_task(task, image_count)
    if validation_error:
        trace.set_error(validation_error)
        return AnalysisResponse(
            task=task,
            evidence=evidence,
            trace=trace.build(TaskStatus.FAILURE),
        )

    # -- Get VLM ------------------------------------------------------ #
    try:
        vlm = get_vlm()
        trace.set_model(vlm.model_name)
    except Exception as exc:
        msg = f"Model service unavailable: {exc}"
        logger.error(msg)
        trace.set_error(msg)
        return AnalysisResponse(
            task=task,
            evidence=evidence,
            trace=trace.build(TaskStatus.FAILURE),
        )

    # ------------------------------------------------------------------ #
    # Task dispatch
    # ------------------------------------------------------------------ #
    try:
        if task == TaskType.VQA:
            trace.add_tool("raster_preprocessor")
            img = load_image_for_vlm(image_paths[0])
            result = answer_vqa(img, query, vlm)
            return AnalysisResponse(
                task=task,
                answer=result["answer"],
                evidence=evidence,
                trace=trace.build(TaskStatus.SUCCESS),
            )

        elif task == TaskType.CAPTION:
            trace.add_tool("raster_preprocessor")
            img = load_image_for_vlm(image_paths[0])
            result = generate_caption(img, vlm)
            return AnalysisResponse(
                task=task,
                caption=result["caption"],
                evidence=evidence,
                trace=trace.build(TaskStatus.SUCCESS),
            )

        elif task == TaskType.CHANGE_DETECTION:
            trace.add_tool("raster_preprocessor")
            trace.add_tool("pixel_difference")
            result = detect_changes(image_paths[0], image_paths[1], vlm)
            for name, path in result.get("change_map", {}).items():
                evidence.append(AnalysisEvidence(
                    kind=f"change_map_{name}",
                    description=f"Change detection {name}",
                    path=path,
                    data=result.get("stats", {}),
                ))
            return AnalysisResponse(
                task=task,
                description=result["description"],
                evidence=evidence,
                trace=trace.build(TaskStatus.SUCCESS),
            )

        elif task == TaskType.CROSS_MODAL:
            trace.add_tool("raster_preprocessor_s1")
            trace.add_tool("raster_preprocessor_s2")
            result = analyze_fusion(image_paths[0], image_paths[1], query, vlm)
            return AnalysisResponse(
                task=task,
                answer=result["answer"],
                evidence=evidence,
                trace=trace.build(TaskStatus.SUCCESS),
            )

        else:
            trace.set_error("Unsupported task type.")
            return AnalysisResponse(
                task=task,
                evidence=evidence,
                trace=trace.build(TaskStatus.FAILURE),
            )

    except Exception as exc:
        msg = str(exc)
        logger.exception("Pipeline error: %s", msg)
        trace.set_error(msg)
        return AnalysisResponse(
            task=task,
            evidence=evidence,
            trace=trace.build(TaskStatus.FAILURE),
        )
