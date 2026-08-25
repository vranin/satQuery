"""
app/schemas/requests.py — Pydantic request / response schemas.

Every API boundary uses these types.
No unstructured dicts or raw strings leave the API.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ------------------------------------------------------------------ #
# Enums
# ------------------------------------------------------------------ #

class TaskType(str, Enum):
    VQA = "vqa"
    CAPTION = "caption"
    CHANGE_DETECTION = "change_detection"
    CROSS_MODAL = "cross_modal"
    UNKNOWN = "unknown"


class TaskStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"


# ------------------------------------------------------------------ #
# Sub-models
# ------------------------------------------------------------------ #

class ExecutionTrace(BaseModel):
    """Audit trail for a single analysis request."""
    task: TaskType
    inputs: int = Field(description="Number of images received")
    tools: list[str] = Field(default_factory=list)
    model: str = Field(default="")
    parameters: dict[str, Any] = Field(default_factory=dict)
    status: TaskStatus
    error: str | None = None
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Estimated confidence in [0, 1] if available",
    )


class AnalysisEvidence(BaseModel):
    """A single piece of computational evidence attached to a response."""
    kind: str = Field(description="e.g. 'difference_map', 'band_stats', 'histogram'")
    description: str = ""
    path: str | None = Field(
        default=None,
        description="Relative path to output file if produced",
    )
    data: dict[str, Any] = Field(default_factory=dict)


# ------------------------------------------------------------------ #
# Main response
# ------------------------------------------------------------------ #

class AnalysisResponse(BaseModel):
    """Top-level response returned by POST /analyze."""
    task: TaskType
    answer: str | None = None
    caption: str | None = None
    description: str | None = None
    evidence: list[AnalysisEvidence] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    trace: ExecutionTrace


# ------------------------------------------------------------------ #
# Error response
# ------------------------------------------------------------------ #

class ErrorResponse(BaseModel):
    """Structured error returned instead of crashing."""
    error: str
    detail: str | None = None
    status_code: int
