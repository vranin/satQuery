# SatQuery Backend

A modular, agentic backend for satellite-imagery analysis using Vision-Language Models (VLMs).

## Architecture

```
User Query + Satellite Image(s)
          │
          ▼
    FastAPI /analyze
          │
          ▼
    Agent Router (classifies task)
          │
  ┌───────┼────────────────┐
  │       │                │
  ▼       ▼                ▼
 VQA   Caption     Change Detection
  │       │                │
  │       │         ┌──────┴──────┐
  │       │         ▼             ▼
  │       │     Rasterio         VLM
  │       │         │             │
  └───────┴─────────┴─────────────┘
                    │
                    ▼
               LLaVA via Ollama
               (via VLM abstraction)
                    │
                    ▼
         Structured JSON Response
              + Audit Trace
```

## Supported Tasks

| Task | Input | Description |
|------|-------|-------------|
| `vqa` | 1 image + question | Visual question answering |
| `caption` | 1 image | Land cover / scene description |
| `change_detection` | 2 images (T1, T2) | Bi-temporal change analysis |
| `cross_modal` | S1 + S2 images | Optical + SAR fusion analysis |

## Technology Stack

- **Framework**: FastAPI + Uvicorn
- **VLM**: Ollama (`llava:7b` default; swappable via abstraction)
- **Geospatial**: Rasterio, NumPy, Pillow
- **Validation**: Pydantic v2
- **Testing**: Pytest

## Quick Start

### Prerequisites

1. **Python 3.11+**
2. **Ollama** — install from https://ollama.com/download/windows
3. **LLaVA model**: `ollama pull llava:7b`

### Setup

```powershell
# From the backend/ directory

# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
Copy-Item .env.example .env
# Edit .env as needed

# 4. Verify installation
python -c "import rasterio, numpy, PIL, fastapi; print('All imports OK')"

# 5. Run the server
uvicorn app.main:app --reload
```

### Verify Ollama

```powershell
ollama list          # Should show llava:7b
ollama run llava:7b  # Interactive test
```

### Selected BigEarthNet patch

The prototype uses one official BigEarthNet-S2 patch. It is stored locally
under `data/bigearthnet/` and is excluded from Git. The selective extractor
streams the official archive and writes only requested patch members; it does
not store the full 59 GiB archive:

```powershell
python scripts/download_bigearthnet_subset.py <S2_PATCH_ID>
```

The matching Sentinel-1 archive can be selected with `--archive-url`, but the
archive is a compressed stream and may require substantial network transfer
before reaching a patch. No S1 model or ChangeFormer checkpoint is downloaded
automatically.

### API Usage

```bash
# Single-image VQA
curl -X POST http://localhost:8000/analyze \
  -F "query=Is there a water body in this image?" \
  -F "images=@path/to/image.tif"

# Change detection
curl -X POST http://localhost:8000/analyze \
  -F "query=What changed between these two images?" \
  -F "images=@image_t1.tif" \
  -F "images=@image_t2.tif"
```

### Response Format

```json
{
  "task": "vqa",
  "answer": "Yes, there is a visible water body...",
  "evidence": [],
  "confidence": 0.85,
  "trace": {
    "tools": ["raster_preprocessor", "llava:7b"],
    "model": "llava:7b",
    "status": "success"
  }
}
```

## Project Structure

```
backend/
├── app/
│   ├── main.py              # FastAPI entry point
│   ├── agent/
│   │   ├── router.py        # Task classifier
│   │   ├── planner.py       # Pipeline orchestrator
│   │   └── trace.py         # Audit trace builder
│   ├── models/
│   │   ├── base.py          # VLM abstraction (ABC)
│   │   └── ollama_client.py # Ollama implementation
│   ├── tasks/
│   │   ├── vqa.py           # VQA task
│   │   ├── caption.py       # Captioning task
│   │   ├── change_detection.py
│   │   └── fusion.py        # S1+S2 cross-modal
│   ├── tools/
│   │   ├── raster.py        # GeoTIFF I/O
│   │   ├── preprocessing.py # Image normalization
│   │   └── difference.py    # Change map computation
│   └── schemas/
│       └── requests.py      # Pydantic schemas
├── data/
│   └── bigearthnet/         # Sample patches only
├── outputs/                 # Generated images/maps
├── tests/
│   ├── test_raster.py
│   ├── test_vqa.py
│   ├── test_caption.py
│   ├── test_change.py
│   ├── test_fusion.py
│   └── test_router.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## BigEarthNet Data

This prototype uses a **small subset** of BigEarthNet v2.0.

Download a single patch for testing:
- Source: https://bigearth.net/
- Sample: Any single Sentinel-2 tile (`.SAFE` folder ~60 MB)
- Place extracted bands in: `data/bigearthnet/<sample_id>/`

## Model Abstraction

To swap the VLM, implement `VisionLanguageModel` from `app/models/base.py`:

```python
from app.models.base import VisionLanguageModel

class MyCustomVLM(VisionLanguageModel):
    def generate(self, images: list, prompt: str) -> str:
        # Your implementation
        ...
```

Then update `VLM_BACKEND` in `.env`.

## Development Stages

The backend foundations and local test coverage are implemented through the
router/API stages. Change detection currently uses a deterministic pixel
difference detector behind a pluggable interface; a ChangeFormer adapter can
be provisioned later on a remote GPU without changing the task API. A true
temporal pair still requires a second acquisition of the same footprint.

| Stage | Status | Description |
|-------|--------|-------------|
| 1–8 | ✅ | Environment, structure, data loader, preprocessing, VLM, VQA, captioning |
| 9 | ✅ | Deterministic change detection with pluggable detector boundary |
| 10 | ✅ | Prompt-level S1 + S2 fusion |
| 11–15 | ✅ | Router, trace, FastAPI endpoint, errors, and tests |
