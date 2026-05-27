# PhotoSorter Design Spec
**Date:** 2026-05-26

## Problem

The user takes large bursts of near-duplicate photos and videos while travelling with two young children (ages 4 and 1). Manually curating these is too time-consuming. The goal is an app that automatically scores and ranks photos within each "moment", presents a fast review UI for final human selection, and outputs a curated set for sharing via Google Photos albums — while keeping the full archive locally.

## Scope

This spec covers the initial version:
- **Input:** local folder of photos (exported from Google Takeout or direct iPhone export)
- **Out of scope for now:** Google Photos API pull, scheduled sync, video handling, uploading curated albums back to Google Photos

## Hardware

- GPU: Nvidia GeForce RTX 3080 (12 GB VRAM)
- RAM: 64 GB
- OS: Linux desktop (remote, accessed via SSH)

## Overall Architecture

Five-stage batch pipeline followed by a web review UI:

```
Input folder
  → 1. Ingestor    (EXIF timestamps + GPS)
  → 2. Grouper     (time windows → CLIP embeddings → DBSCAN clusters = "moments")
  → 3. Scorer      (InsightFace face analysis + OpenCV image quality)
  → 4. Ranker      (composite weighted score per photo, keep suggestion per moment)
  → 5. SQLite DB   (all metadata, scores, groups, review decisions)
  → 6. Web UI      (FastAPI, review moments one at a time, submit decisions)
  → Output folder  (curated photos hard-linked, originals untouched)
```

The pipeline (stages 1–5) runs once as a batch job. The web UI (stage 6) is separate and resumable — all state persists in SQLite.

## Stage 1: Ingestor

- Recursively scans input folder for image files (JPEG, HEIC, PNG); HEIC support requires `pillow-heif` registered at startup
- Extracts EXIF timestamp and GPS coordinates per photo
- Falls back to file modification time if EXIF timestamp is absent
- Writes photo records to SQLite

## Stage 2: Grouper

Two-pass algorithm:

**Pass 1 — Time windowing:**
Photos sorted by timestamp. A gap > 3 minutes (configurable) between consecutive photos starts a new candidate group.

**Pass 2 — Visual similarity:**
Within each candidate group, CLIP embeddings are computed on GPU. DBSCAN clustering on the embeddings splits visually distinct sub-scenes within the same time window and produces the final "moments."

Edge cases:
- Single-photo groups are valid moments and appear in review
- DBSCAN noise points are assigned to the nearest cluster (not discarded)
- Very large bursts (50+ photos) are kept as one moment; the review UI paginates them

## Stage 3: Scorer

**Face analysis (InsightFace, GPU):**

All faces in a photo are detected and each produces three signals:
- **Gaze** — facing camera (0–1)
- **Smile / expression** — positive expression (0–1)
- **Eyes open** — binary (1 = open, 0 = closed)

Each signal is averaged independently across all detected faces to produce photo-level `gaze_score`, `smile_score`, and `eyes_score`. Both adult and child faces are evaluated equally.

**Image quality (OpenCV, CPU):**
- **Sharpness** — Laplacian variance, normalised within the moment group
- **Motion blur** — FFT magnitude spectrum analysis
- **Exposure** — histogram-based over/underexposure check

## Stage 4: Ranker

Composite score per photo (0–1) = weighted sum of five photo-level signals, configurable weights with these defaults:

| Signal | Weight |
|---|---|
| Gaze | 30% |
| Smile | 25% |
| Eyes open | 20% |
| Sharpness / motion blur | 15% |
| Exposure | 10% |

Weights are stored in `config.yaml`. The ranker also produces a **keep suggestion** per moment: any photo scoring below 60% of the top photo's composite score in the group is excluded from the suggestion. This threshold is also configurable.

## Stage 5: SQLite Database

Single file (`photosorter.db`). Tables:
- `photos` — file path, timestamp, GPS, EXIF metadata
- `moments` — group ID, photo IDs, time range
- `scores` — per-photo composite and per-signal scores
- `decisions` — per-photo keep/reject, submitted timestamp

Enables full pipeline resumability and review resumability.

## Stage 6: Web Review UI

FastAPI server (default port 8080), accessible from client via SSH tunnel:
```
ssh -L 8080:localhost:8080 remote-host
```

**Review flow:**
- Moments presented chronologically, one at a time
- Photos shown in ranked order (best score first) in a grid
- Each photo displays composite score + signal breakdown (gaze / smile / eyes / sharpness / exposure)
- App suggests how many to keep; user can override individual toggles
- Submit moment → next moment
- Progress bar showing reviewed vs total moments

**Keyboard shortcuts:**
| Key | Action |
|---|---|
| `→` / `Enter` | Submit and advance |
| `1`–`9` | Toggle photo at position |
| `A` | Select all / deselect all |
| `Z` | Undo last submission |

All decisions written to SQLite on submit. Server can be stopped and restarted; reviewed moments are skipped on resume.

## Output

Selected photos are hard-linked into `output/curated/YYYY-MM-DD/` (falls back to copy if cross-filesystem). Originals in the input folder are never modified or deleted.

## Tech Stack

| Purpose | Library |
|---|---|
| Face analysis | `insightface` + `onnxruntime-gpu` |
| Visual embeddings | `open_clip_torch` |
| HEIC support | `pillow-heif` |
| Clustering | `scikit-learn` (DBSCAN) |
| Image quality | `opencv-python` |
| Image loading / EXIF | `Pillow` |
| Web server | `fastapi` + `uvicorn` |
| Database | `sqlite3` (stdlib) |
| Frontend | Vanilla JS + HTML |

## Project Structure

```
PhotoSorting/
├── config.yaml
├── pipeline.py
├── photosorter/
│   ├── ingestor.py
│   ├── grouper.py
│   ├── scorer.py
│   ├── ranker.py
│   ├── db.py
│   └── api.py
├── ui/
│   ├── index.html
│   └── app.js
└── output/
    └── curated/
```

## Typical Workflow

```bash
# Run pipeline on exported folder
python pipeline.py --input /mnt/backup/takeout-2024/

# Start review server
uvicorn photosorter.api:app --port 8080

# Open browser on client machine (after SSH tunnel)
# http://localhost:8080
```

## Future Extensions (Out of Scope)

- Google Photos API pull (manual trigger and scheduled)
- Upload curated albums back to Google Photos
- Video handling (extract best frames)
