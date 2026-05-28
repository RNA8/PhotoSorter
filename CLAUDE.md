# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

PhotoSorter is a local Python pipeline for curating travel burst photos. It ingests a folder of photos, groups them into "moments" using time proximity and CLIP visual similarity, scores each photo using InsightFace face analysis and OpenCV quality metrics, then serves a FastAPI web UI for reviewing and keeping the best shots. Optimised for travel photos with children (gaze, smile, open eyes).

## Commands

```bash
# Run the pipeline
.venv/bin/python pipeline.py --input /path/to/photos

# Start the review server
.venv/bin/python -m uvicorn photosorter.api:app --port 8080

# Run all tests (43 tests, no GPU needed — AI models are mocked)
.venv/bin/pytest -v

# Run a single test file or test
.venv/bin/pytest tests/test_scorer.py -v
.venv/bin/pytest tests/test_scorer.py::test_name -v

# Install dependencies
pip install -r requirements.txt
```

## Architecture

The pipeline runs as a linear sequence: `ingestor → grouper → scorer → ranker`, orchestrated by `pipeline.py`. Each stage reads from and writes to a single SQLite database (`photosorter.db`). The review server (`api.py`) reads that same database and serves a static JS/HTML UI from `ui/`.

**Key design decisions:**
- SQLite with `check_same_thread=False` — FastAPI worker threads share one connection. All DB access goes through `photosorter/db.py`.
- `INSERT OR IGNORE` on photo path makes re-running the pipeline safe (idempotent ingest).
- Curated photos are hard-linked (not copied) to `output/curated/YYYY-MM-DD/` — zero disk duplication.
- The `outputter` creates hard links at submit time via the API, not as a batch step.
- Sharpness scores are normalized per-moment (relative, not absolute).

**Scoring pipeline:** Each photo gets face scores (gaze, smile, eyes-open via InsightFace landmarks) and quality scores (sharpness via Laplacian+FFT, exposure via histogram). These are combined using configurable weights in `config.yaml` into a single composite score. Photos scoring above `keep_threshold × top_score` in their moment are suggested as keeps.

**Grouping:** Photos are first split into time windows (`time_window_minutes`), then sub-clustered within each window using CLIP embeddings + DBSCAN.

## Testing

Tests mock InsightFace and CLIP so no GPU or model downloads are needed. The `conftest.py` provides `tmp_photo_dir` (generates test JPGs) and `tmp_db` (in-memory SQLite) fixtures. Tests use `httpx.AsyncClient` for API testing via `test_api.py`.

## Configuration

All tunable parameters are in `config.yaml`. The `Config` dataclass in `photosorter/config.py` loads it. Weights for scoring dimensions (gaze, smile, eyes, sharpness, exposure) must sum to 1.0.
