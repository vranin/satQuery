"""
data/bigearthnet/loader.py — BigEarthNet data pipeline (Stage 3).

Provides load_bigearthnet_sample(sample_id) which returns:
  {
    "sample_id": "...",
    "s1_path":   Path | None,
    "s2_path":   Path | None,   (or list of band paths)
    "patch_dir": Path,
    "annotations": [...]
  }

Does NOT download data. Assumes you have placed a patch in:
  data/bigearthnet/<sample_id>/
"""
from __future__ import annotations

import json
import logging
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class BigEarthNetSample:
    sample_id: str
    patch_dir: Path
    s1_path: Optional[Path] = None          # stacked S1 GeoTIFF (if present)
    s2_path: Optional[Path] = None          # stacked S2 GeoTIFF (if present)
    s2_band_paths: dict[str, Path] = field(default_factory=dict)  # per-band files
    annotations: list[str] = field(default_factory=list)
    annotation_records: list[dict[str, Any]] = field(default_factory=list)


def _find_annotations(patch_dir: Path, sample_id: str) -> list[str]:
    """
    Try to load annotations from:
      1. <patch_dir>/labels_metadata.json  (BigEarthNet v2 format)
      2. <patch_dir>/<sample_id>_labels_metadata.json
      3. <patch_dir>/annotations.txt       (simple one-label-per-line)
    Returns an empty list if none found.
    """
    candidates = [
        patch_dir / "labels_metadata.json",
        patch_dir / f"{sample_id}_labels_metadata.json",
        patch_dir / "annotations.txt",
    ]
    for cand in candidates:
        if cand.exists():
            if cand.suffix == ".json":
                try:
                    data = json.loads(cand.read_text(encoding="utf-8"))
                    labels = data.get("labels", data.get("label", []))
                    if isinstance(labels, str):
                        labels = [labels]
                    return labels
                except Exception:
                    pass
            else:
                return [line.strip() for line in cand.read_text().splitlines() if line.strip()]
    return []


def _read_annotation_records(path: Path) -> list[dict[str, Any]]:
    """Read a small local annotation export without adding data dependencies."""
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("data", data.get("annotations", [data]))
        return [item for item in data if isinstance(item, dict)]

    if path.suffix.lower() == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    with path.open("r", encoding="utf-8", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        delimiter = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters="\t,").delimiter
        except csv.Error:
            pass
        return [dict(row) for row in csv.DictReader(handle, delimiter=delimiter)]


def _find_external_annotations(sample_id: str) -> list[dict[str, Any]]:
    """Find records for a sample in the configured annotation export."""
    configured = get_settings().bigearthnet_annotations_file.strip()
    if not configured:
        return []

    path = Path(configured)
    if not path.exists():
        logger.warning("Configured BigEarthNet annotations file not found: %s", path)
        return []
    if path.suffix.lower() == ".parquet":
        raise RuntimeError(
            "Parquet annotations require an optional parquet reader; "
            "export a small CSV/JSON subset for this prototype."
        )

    records = _read_annotation_records(path)
    matches = []
    for record in records:
        identifiers = {
            str(record.get(key, ""))
            for key in ("sample_id", "patch_id", "s1_name", "id", "ID")
        }
        if sample_id in identifiers:
            matches.append(record)
    return matches


def load_bigearthnet_sample(sample_id: str) -> BigEarthNetSample:
    """
    Load a BigEarthNet sample by ID.

    Expects the patch to be placed at:
      <BIGEARTHNET_DATA_DIR>/<sample_id>/

    Supports:
    - Per-band TIF files: *_B01.tif, *_B02.tif, …  (BigEarthNet standard)
    - Stacked S1 GeoTIFF named with "S1" in the filename
    - Stacked S2 GeoTIFF named with "S2" in the filename

    Parameters
    ----------
    sample_id:
        The patch identifier, e.g. "S2A_MSIL2A_20170613T101031_0_45"

    Returns
    -------
    BigEarthNetSample

    Raises
    ------
    FileNotFoundError
        If the patch directory does not exist.
    """
    settings = get_settings()
    patch_dir = settings.bigearthnet_path / sample_id

    if not patch_dir.exists():
        raise FileNotFoundError(
            f"BigEarthNet patch directory not found: {patch_dir}\n"
            f"Place your patch at: {patch_dir}"
        )

    sample = BigEarthNetSample(sample_id=sample_id, patch_dir=patch_dir)

    tif_files = list(patch_dir.glob("*.tif")) + list(patch_dir.glob("*.TIF"))

    for tif in tif_files:
        name_upper = tif.stem.upper()
        # Per-band detection (e.g. _B01, _B02, …, _VV, _VH)
        for band in [
            "B01","B02","B03","B04","B05","B06","B07",
            "B08","B8A","B09","B10","B11","B12",
            "VV","VH",
        ]:
            if name_upper.endswith(f"_{band}"):
                sample.s2_band_paths[band] = tif
                break
        else:
            # Stacked files
            if "S1" in name_upper:
                sample.s1_path = tif
            elif "S2" in name_upper:
                sample.s2_path = tif

    sample.annotations = _find_annotations(patch_dir, sample_id)
    sample.annotation_records = _find_external_annotations(sample_id)
    if sample.annotation_records and not sample.annotations:
        sample.annotations = [
            str(record.get("output") or record.get("caption") or record.get("label"))
            for record in sample.annotation_records
            if record.get("output") or record.get("caption") or record.get("label")
        ]

    logger.info(
        "Loaded BigEarthNet sample: %s  bands=%d  s1=%s  s2=%s  labels=%s",
        sample_id,
        len(sample.s2_band_paths),
        sample.s1_path,
        sample.s2_path,
        sample.annotations,
    )

    return sample


def list_available_samples() -> list[str]:
    """List all sample IDs available in the BigEarthNet data directory."""
    settings = get_settings()
    base = settings.bigearthnet_path
    if not base.exists():
        return []
    return sorted(
        d.name
        for d in base.iterdir()
        if d.is_dir() and not d.name.startswith("__")
    )
