"""
app/agent/trace.py — Execution audit trace builder (Stage 12).

Every API request generates a structured trace for auditability.
"""
from __future__ import annotations

import time
from typing import Any

from app.schemas.requests import ExecutionTrace, TaskStatus, TaskType


class TraceBuilder:
    """
    Incrementally build an ExecutionTrace for a single request.

    Usage
    -----
    trace = TraceBuilder(task=TaskType.VQA, inputs=1)
    trace.add_tool("raster_preprocessor")
    trace.set_model("llava:7b")
    trace.add_param("temperature", 0.1)
    result = trace.build(status=TaskStatus.SUCCESS)
    """

    def __init__(self, task: TaskType, inputs: int) -> None:
        self._task = task
        self._inputs = inputs
        self._tools: list[str] = []
        self._model: str = ""
        self._params: dict[str, Any] = {}
        self._error: str | None = None
        self._confidence: float | None = None
        self._start = time.monotonic()

    def add_tool(self, name: str) -> "TraceBuilder":
        if name not in self._tools:
            self._tools.append(name)
        return self

    def set_model(self, name: str) -> "TraceBuilder":
        self._model = name
        if name not in self._tools:
            self._tools.append(name)
        return self

    def add_param(self, key: str, value: Any) -> "TraceBuilder":
        self._params[key] = value
        return self

    def set_error(self, error: str) -> "TraceBuilder":
        self._error = error
        return self

    def set_confidence(self, confidence: float) -> "TraceBuilder":
        self._confidence = max(0.0, min(1.0, confidence))
        return self

    def build(self, status: TaskStatus) -> ExecutionTrace:
        elapsed = round(time.monotonic() - self._start, 3)
        params = {**self._params, "elapsed_seconds": elapsed}
        return ExecutionTrace(
            task=self._task,
            inputs=self._inputs,
            tools=self._tools,
            model=self._model,
            parameters=params,
            status=status,
            error=self._error,
            confidence=self._confidence,
        )
