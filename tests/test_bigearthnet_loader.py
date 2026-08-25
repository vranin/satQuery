"""Tests for the local BigEarthNet sample and annotation loader."""
from __future__ import annotations

import json
from types import SimpleNamespace

from data.bigearthnet import loader


def test_load_sample_matches_external_annotation(tmp_path, monkeypatch):
    sample_id = "sample-001"
    patch_dir = tmp_path / sample_id
    patch_dir.mkdir()
    (patch_dir / "annotations.txt").write_text("forest\n", encoding="utf-8")

    annotation_file = tmp_path / "annotations.json"
    annotation_file.write_text(
        json.dumps([{"patch_id": sample_id, "input": "What is visible?", "output": "Forest."}]),
        encoding="utf-8",
    )

    settings = SimpleNamespace(
        bigearthnet_path=tmp_path,
        bigearthnet_annotations_file=str(annotation_file),
    )
    monkeypatch.setattr(loader, "get_settings", lambda: settings)

    sample = loader.load_bigearthnet_sample(sample_id)

    assert sample.annotations == ["forest"]
    assert sample.annotation_records[0]["output"] == "Forest."