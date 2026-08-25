"""
app/main.py — FastAPI application entry point (Stage 13).

Exposes:
  GET  /health          — Liveness + Ollama ping
  POST /analyze         — Main analysis endpoint
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.agent.planner import run_pipeline
from app.config import get_settings
from app.schemas.requests import AnalysisResponse, ErrorResponse, TaskType

# ------------------------------------------------------------------ #
# Logging
# ------------------------------------------------------------------ #

settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# App
# ------------------------------------------------------------------ #

app = FastAPI(
    title="SatQuery",
    description=(
        "Agentic satellite imagery analysis backend. "
        "Accepts natural-language queries + satellite images and routes them "
        "through VQA, captioning, change detection, or cross-modal analysis."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------ #
# Health
# ------------------------------------------------------------------ #

@app.get("/health", tags=["System"])
async def health() -> dict:
    """Liveness check. Pings the Ollama service."""
    from app.models.ollama_client import OllamaVLM

    vlm = OllamaVLM()
    model_ready = vlm.ping()
    return {
        "status": "ok",
        "model": settings.ollama_model,
        "model_ready": model_ready,
        "backend": settings.vlm_backend,
    }


# ------------------------------------------------------------------ #
# Main endpoint
# ------------------------------------------------------------------ #

@app.post(
    "/analyze",
    response_model=AnalysisResponse,
    tags=["Analysis"],
    summary="Analyse satellite image(s) with a natural-language query",
)
async def analyze(
    query: str = Form(..., description="Natural-language query or question"),
    images: list[UploadFile] = File(..., description="One or more satellite images"),
    metadata: str | None = Form(
        default=None,
        description='Optional JSON metadata, e.g. {"s1": true, "s2": true}',
    ),
) -> AnalysisResponse:
    """
    Main analysis endpoint.

    - **query**: e.g. `"Is there a water body in this image?"` or
      `"What changed between these two images?"`
    - **images**: One `.tif`, `.png`, or `.jpg` file (or two for change detection)
    - **metadata**: Optional JSON string for cross-modal hints
    """
    if not images:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one image file is required.",
        )

    if not query or not query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query must not be empty.",
        )

    # Parse optional metadata
    meta_dict: dict | None = None
    if metadata:
        import json
        try:
            meta_dict = json.loads(metadata)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="metadata must be valid JSON.",
            )

    # Save uploads to a temporary directory for processing
    saved_paths: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="satquery_") as tmp:
        for upload in images:
            if not upload.filename:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Each uploaded file must have a filename.",
                )
            dest = Path(tmp) / Path(upload.filename).name
            content = await upload.read()
            dest.write_bytes(content)
            saved_paths.append(dest)

        logger.info(
            "analyze: query=%r  images=%d  files=%s",
            query[:80],
            len(saved_paths),
            [p.name for p in saved_paths],
        )

        response = run_pipeline(
            query=query.strip(),
            image_paths=saved_paths,
            metadata=meta_dict,
        )

    return response


# ------------------------------------------------------------------ #
# Error handlers
# ------------------------------------------------------------------ #

@app.exception_handler(404)
async def not_found_handler(request, exc) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content=ErrorResponse(
            error="Not found",
            detail=str(request.url),
            status_code=404,
        ).model_dump(),
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc) -> JSONResponse:
    logger.exception("Unhandled server error")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail="An unexpected error occurred.",
            status_code=500,
        ).model_dump(),
    )
