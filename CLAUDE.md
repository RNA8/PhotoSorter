# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

PhotoSorter is a local Python pipeline for curating travel burst photos. It ingests a folder of photos, groups them into "moments" using time proximity and CLIP visual similarity, scores each photo using InsightFace face analysis and OpenCV quality metrics, then serves a FastAPI web UI for reviewing and keeping the best shots. Optimised for travel photos with children (gaze, smile, open eyes).

## Commands

```bash
# Run the pipeline (must be run from the repo root — config.yaml and paths are relative to CWD)
.venv/bin/python pipeline.py --input /path/to/photos

# Start the review server (also must be run from the repo root — it mounts ./ui and loads ./config.yaml)
.venv/bin/python -m uvicorn photosorter.api:app --port 8080

# Run all tests (43 tests, no GPU needed — AI models are mocked)
.venv/bin/pytest -v

# Run a single test file or test
.venv/bin/pytest tests/test_scorer.py -v
.venv/bin/pytest tests/test_scorer.py::test_name -v

# Install dependencies
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended for `pipeline.py` (CLIP + InsightFace inference); it falls back to CPU automatically but is much slower. Not needed for the review server or the test suite.

## Architecture

The pipeline runs as a linear sequence: `ingestor → grouper → scorer → ranker`, orchestrated by `pipeline.py`. Each stage reads from and writes to a single SQLite database (`photosorter.db`), via five tables defined in `photosorter/db.py`: `photos`, `moments`, `moment_photos` (join table with per-moment `rank`), `scores`, and `decisions`. The review server (`api.py`) reads that same database and serves a static JS/HTML UI from `ui/`.

**Key design decisions:**
- SQLite with `check_same_thread=False` — FastAPI worker threads share one connection. All DB access goes through `photosorter/db.py`.
- `INSERT OR IGNORE` on photo path makes re-running the pipeline safe (idempotent ingest); `db.py` checks `cur.rowcount` rather than `cur.lastrowid` to detect a skipped insert (a SQLite quirk: `lastrowid` on an ignored insert returns the *previous* insert's id, not 0).
- Curated photos are hard-linked (not copied) to `output/curated/YYYY-MM-DD/` — zero disk duplication. `outputter.output_photo` falls back to `shutil.copy2` if `os.link` fails (e.g. crossing filesystems).
- The `outputter` creates hard links at submit time via the API, not as a batch step.
- Sharpness scores are normalized per-moment (relative, not absolute).

**Scoring pipeline:** Each photo gets face scores (gaze, smile, eyes-open via InsightFace landmarks) and quality scores (sharpness via Laplacian+FFT, exposure via histogram). These are combined using configurable weights in `config.yaml` into a single composite score (`ranker.py`). Photos scoring above `keep_threshold × top_score` in their moment are suggested as keeps — a threshold relative to the moment's best photo, not an absolute cutoff, so a mediocre moment still yields suggestions.

**Grouping:** Photos are first split into time windows (`time_window_minutes`), then sub-clustered within each window using CLIP embeddings + DBSCAN (cosine distance). DBSCAN noise points (unclustered outliers) are reassigned to their nearest cluster by embedding similarity rather than discarded — see `grouper.cluster_by_visual_similarity`.

For the full rationale behind these choices (and the alternatives considered), see `docs/design-choices.md`.

## Testing

Tests mock InsightFace and CLIP so no GPU or model downloads are needed. The `conftest.py` provides `tmp_photo_dir` (generates test JPGs) and `tmp_db` (in-memory SQLite) fixtures. Tests use `httpx.AsyncClient` for API testing via `test_api.py`.

## Configuration

All tunable parameters are in `config.yaml`, loaded into the `Config`/`Weights` dataclasses in `photosorter/config.py`. `load_config` raises `ValueError` on unknown top-level or `weights` keys (e.g. a typo like `time_windw_minutes`), so config errors surface immediately rather than silently falling back to defaults. Weights for scoring dimensions (gaze, smile, eyes, sharpness, exposure) are expected to sum to 1.0 to keep the composite score in `[0, 1]`, but this isn't validated in code.
