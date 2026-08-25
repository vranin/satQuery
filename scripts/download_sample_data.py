"""
scripts/download_sample_data.py
================================
Downloads a tiny sample of real Sentinel-2 GeoTIFF patches for development.

Strategy
--------
We use the EuroSAT Multispectral dataset (Zenodo record 7711810).
EuroSAT patches are:
  - 64 × 64 pixels
  - 13 Sentinel-2 spectral bands (same sensor as BigEarthNet)
  - Organised by land-cover class
  - Freely available, no registration required

We download just the 'River' class zip (~30 MB) as a real water-body test
case, plus the 'Forest' class for a vegetation test case.

Alternatively, if you have your own GeoTIFFs, place them directly in:
  data/bigearthnet/<sample_id>/

Usage
-----
From the backend/ directory:
    .venv\\Scripts\\python.exe scripts/download_sample_data.py

What it creates
---------------
data/
  bigearthnet/
    eurosat_River_sample/
      River_00001.tif   (13-band S2 GeoTIFF)
      River_00002.tif
      annotations.txt   (label: River)
    eurosat_Forest_sample/
      Forest_00001.tif
      Forest_00002.tif
      annotations.txt   (label: Forest)
    eurosat_Industrial_sample/
      Industrial_00001.tif
      annotations.txt   (label: Industrial)
"""
from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Configuration
# ------------------------------------------------------------------ #

# EuroSAT MS on Zenodo — individual class zip files
# Zenodo record: https://zenodo.org/records/7711810
# Each class zip is ~10-80 MB. We download the smallest ones.
EUROSAT_BASE = "https://zenodo.org/records/7711810/files"

CLASSES_TO_FETCH = [
    ("River",       3),   # water body — great for VQA "Is there a river?"
    ("Forest",      3),   # vegetation — captioning test
    ("Industrial",  2),   # urban/built — change detection T1
    ("Residential", 2),   # urban — change detection T2
]

DATA_DIR = Path("data/bigearthnet")


# ------------------------------------------------------------------ #
# Downloader
# ------------------------------------------------------------------ #

def download_class_sample(class_name: str, n_patches: int) -> None:
    url = f"{EUROSAT_BASE}/{class_name}.zip"
    out_dir = DATA_DIR / f"eurosat_{class_name}_sample"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write annotation file
    (out_dir / "annotations.txt").write_text(class_name, encoding="utf-8")

    log.info("Downloading %s class (%d patches)…", class_name, n_patches)
    log.info("  URL: %s", url)

    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.error("Failed to download %s: %s", class_name, exc)
        log.error("You can download manually from: %s", url)
        return

    total = int(resp.headers.get("content-length", 0))
    log.info("  Size: %.1f MB", total / 1024 / 1024)

    buf = io.BytesIO()
    downloaded = 0
    for chunk in resp.iter_content(chunk_size=1024 * 1024):
        buf.write(chunk)
        downloaded += len(chunk)
        if total:
            pct = downloaded / total * 100
            print(f"\r  Progress: {pct:.1f}%  ({downloaded//1024//1024} MB)", end="", flush=True)
    print()

    buf.seek(0)
    saved = 0
    with zipfile.ZipFile(buf) as zf:
        tif_members = [m for m in zf.namelist() if m.endswith(".tif")]
        for member in tif_members[:n_patches]:
            filename = Path(member).name
            dest = out_dir / filename
            dest.write_bytes(zf.read(member))
            log.info("  Saved: %s", dest)
            saved += 1

    log.info("  Done: %d patches saved to %s", saved, out_dir)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    log.info("=" * 60)
    log.info("SatQuery — Sample Data Downloader")
    log.info("Target directory: %s", DATA_DIR.resolve())
    log.info("=" * 60)

    for class_name, n in CLASSES_TO_FETCH:
        download_class_sample(class_name, n)

    log.info("")
    log.info("Download complete. Available samples:")
    for d in sorted(DATA_DIR.iterdir()):
        if d.is_dir():
            tifs = list(d.glob("*.tif"))
            log.info("  %-40s  %d TIF file(s)", d.name, len(tifs))

    log.info("")
    log.info("Test with:")
    log.info("  .venv\\Scripts\\python.exe scripts/verify_sample_data.py")


if __name__ == "__main__":
    main()
