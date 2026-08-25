"""API boundary tests with the model pipeline mocked."""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
from app.schemas.requests import AnalysisResponse, ExecutionTrace, TaskStatus, TaskType


client = TestClient(app)


def _response(task: TaskType = TaskType.VQA) -> AnalysisResponse:
    return AnalysisResponse(
        task=task,
        answer="Yes.",
        trace=ExecutionTrace(task=task, inputs=1, status=TaskStatus.SUCCESS),
    )


def test_analyze_requires_image():
    response = client.post("/analyze", data={"query": "Is there water?"})

    assert response.status_code == 422


def test_analyze_rejects_invalid_metadata():
    response = client.post(
        "/analyze",
        data={"query": "Is there water?", "metadata": "not-json"},
    )

    assert response.status_code == 422


def test_analyze_returns_structured_response(monkeypatch, rgb_png):
    monkeypatch.setattr("app.main.run_pipeline", lambda **kwargs: _response())

    response = client.post(
        "/analyze",
        data={"query": "Is there water?"},
        files={"images": ("image.png", rgb_png.read_bytes(), "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["task"] == "vqa"
    assert body["trace"]["status"] == "success"


def test_upload_filename_cannot_escape_temp_directory(monkeypatch, rgb_png):
    captured = {}

    def fake_pipeline(**kwargs):
        captured["paths"] = kwargs["image_paths"]
        return _response()

    monkeypatch.setattr("app.main.run_pipeline", fake_pipeline)
    response = client.post(
        "/analyze",
        data={"query": "Is there water?"},
        files={"images": ("..\\escaped.png", rgb_png.read_bytes(), "image/png")},
    )

    assert response.status_code == 200
    assert captured["paths"][0].name == "escaped.png"