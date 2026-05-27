# PhotoSorter

A local Python pipeline that ingests a folder of travel photos, groups them into moments using time proximity and visual similarity, scores each photo with face AI and image quality analysis, and presents a web review UI for fast curation.

Designed for travel burst photography with children — optimises for open eyes, forward gaze, and smiling faces.

---

## Requirements

- Linux or macOS
- Python 3.8+
- NVIDIA GPU with CUDA (recommended; CPU fallback works but is slow)
- [onnxruntime-gpu](https://onnxruntime.ai/) compatible CUDA installation

---

## Installation

```bash
# Clone the repo
git clone https://github.com/RNA8/PhotoSorter
cd PhotoSorter

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

On first run, InsightFace will automatically download the `buffalo_l` model weights (~300 MB) to `~/.insightface/models/`.

---

## Quickstart

### Step 1 — Run the pipeline on your photos folder

```bash
.venv/bin/python pipeline.py --input /path/to/your/photos
```

This will:
1. Scan the folder recursively for JPG, HEIC, HEIF, and PNG files
2. Extract EXIF timestamps and GPS data
3. Group photos into moments (3-minute time windows + visual clustering)
4. Score each photo for face quality and image quality
5. Rank photos within each moment
6. Store everything in `photosorter.db`

Progress is printed to the terminal. Re-running is safe — photos already in the database are skipped.

### Step 2 — Start the review server

```bash
.venv/bin/python -m uvicorn photosorter.api:app --port 8080
```

### Step 3 — Open the review UI

**If you are on the same machine:**

Navigate to [http://localhost:8080](http://localhost:8080)

**If you are connecting via SSH (remote machine):**

On your local machine, open an SSH tunnel before browsing:

```bash
ssh -L 8080:localhost:8080 your-username@your-remote-host
```

Then open [http://localhost:8080](http://localhost:8080) in your local browser.

---

## Review UI

The review interface shows one moment at a time as a grid of photo cards, ranked best-first.

- **Green border** — photo is in the suggested keep set (AI recommendation)
- **Score bars** — gaze, smile, eyes open, sharpness, exposure scores per photo

### Keyboard shortcuts

| Key | Action |
|-----|--------|
| `→` or `Enter` | Submit current moment and advance |
| `1` – `9` | Toggle keep/reject for photo N |
| `A` | Select all / deselect all |
| `Z` | Undo last submitted moment |

Click any photo card to toggle it individually.

### Output

Curated photos are hard-linked (not copied) into:

```
output/curated/YYYY-MM-DD/filename.jpg
```

Hard links mean zero disk space duplication — the curated folder and your original photos share the same underlying file data.

---

## Configuration

All tunable parameters live in `config.yaml`:

```yaml
time_window_minutes: 3        # gap that splits a new moment
dbscan_eps: 0.3               # DBSCAN cosine-distance threshold
dbscan_min_samples: 1         # minimum cluster size
keep_threshold: 0.6           # fraction of top score to include in suggestion

weights:
  gaze: 0.30                  # face looking at camera
  smile: 0.25                 # mouth corners lifted
  eyes: 0.20                  # eyes open (EAR metric)
  sharpness: 0.15             # Laplacian + FFT blur detection
  exposure: 0.10              # histogram midtone coverage

clip_model: "ViT-B-32"
clip_pretrained: "openai"
insightface_model: "buffalo_l"
det_size: [640, 640]
output_dir: "output/curated"
db_path: "photosorter.db"
port: 8080
```

**Tuning tips:**

- Increase `gaze` weight if camera-facing shots matter most to you
- Increase `smile` weight to prioritise happy expressions
- Lower `keep_threshold` (e.g. 0.5) to get more suggestions per moment
- Raise `time_window_minutes` if your bursts span longer shooting sessions

---

## Running tests

```bash
.venv/bin/pytest -v
```

All 43 tests should pass. InsightFace and CLIP are mocked in unit tests so no GPU is needed to run the test suite.

---

## Project structure

```
pipeline.py              # CLI entrypoint
photosorter/
  config.py              # Config dataclass + YAML loader
  db.py                  # SQLite schema + all queries
  ingestor.py            # Folder scan + EXIF extraction
  grouper.py             # Time windowing + CLIP + DBSCAN
  scorer.py              # InsightFace face scoring + OpenCV quality
  ranker.py              # Composite weighted score + keep suggestion
  outputter.py           # Hard-link curated photos to output folder
  api.py                 # FastAPI review server
ui/
  index.html             # Review UI shell
  app.js                 # Review UI logic
tests/                   # pytest test suite
config.yaml              # Tunable parameters
requirements.txt         # Pinned dependencies
```

---

## Architecture overview

```
photos on disk
      │
      ▼
  ingestor ──► SQLite (photos table)
      │
      ▼
  grouper ──► SQLite (moments + moment_photos tables)
  (time windows → CLIP embeddings → DBSCAN)
      │
      ▼
  scorer ──► SQLite (scores table)
  (InsightFace landmarks + OpenCV quality)
      │
      ▼
  ranker ──► composite score, rank per moment
      │
      ▼
  FastAPI ──► browser review UI
      │
      ▼
  outputter ──► output/curated/YYYY-MM-DD/
               SQLite (decisions table)
```
