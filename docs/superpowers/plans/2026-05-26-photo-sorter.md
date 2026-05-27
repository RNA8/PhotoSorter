# PhotoSorter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local Python app that groups travel photos into moments, scores them with face AI + image quality, and presents a web review UI for fast curation.

**Architecture:** A five-stage batch pipeline (ingest → group → score → rank → store) populates a SQLite database, then a separate FastAPI server serves a vanilla-JS review UI where the user toggles keep/reject per moment. Output photos are hard-linked into a dated curated folder.

**Tech Stack:** Python 3.11+, insightface + onnxruntime-gpu, open_clip_torch, scikit-learn, opencv-python, Pillow + pillow-heif, fastapi + uvicorn, sqlite3 (stdlib), vanilla JS

---

## File Map

| File | Responsibility |
|---|---|
| `config.yaml` | All tunable parameters (weights, thresholds, paths, port) |
| `requirements.txt` | Pinned dependencies |
| `pipeline.py` | CLI entrypoint: orchestrates ingest → group → score → rank → store |
| `photosorter/__init__.py` | Empty package marker |
| `photosorter/config.py` | Load `config.yaml` into a `Config` dataclass |
| `photosorter/db.py` | SQLite schema creation + all CRUD queries |
| `photosorter/ingestor.py` | Folder scan, EXIF extraction, write photos to DB |
| `photosorter/grouper.py` | Time windowing + CLIP embeddings + DBSCAN clustering |
| `photosorter/scorer.py` | InsightFace face analysis + OpenCV image quality |
| `photosorter/ranker.py` | Composite weighted score + keep suggestion per moment |
| `photosorter/outputter.py` | Hard-link (or copy) selected photos to output folder |
| `photosorter/api.py` | FastAPI app: progress, next moment, submit, undo, serve photos |
| `ui/index.html` | Review UI shell |
| `ui/app.js` | Review UI logic: grid, toggles, keyboard shortcuts, fetch calls |
| `tests/conftest.py` | Shared fixtures: temp photo dir, temp DB connection |
| `tests/test_config.py` | Config loading tests |
| `tests/test_db.py` | DB CRUD tests |
| `tests/test_ingestor.py` | Ingestor scan + EXIF fallback tests |
| `tests/test_grouper.py` | Time windowing + clustering tests (CLIP mocked) |
| `tests/test_scorer.py` | Quality scoring tests (InsightFace mocked) |
| `tests/test_ranker.py` | Composite score + suggestion tests |
| `tests/test_outputter.py` | Hard-link + fallback-to-copy tests |
| `tests/test_api.py` | FastAPI endpoint tests via TestClient |

---

## Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `config.yaml`
- Create: `photosorter/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `pytest.ini`

- [ ] **Step 1: Create requirements.txt**

```
insightface==0.7.3
onnxruntime-gpu==1.17.1
open_clip_torch==2.24.0
scikit-learn==1.4.2
opencv-python==4.9.0.80
Pillow==10.3.0
pillow-heif==0.16.0
fastapi==0.111.0
uvicorn[standard]==0.29.0
pyyaml==6.0.1
pytest==8.2.0
httpx==0.27.0
```

- [ ] **Step 2: Create config.yaml**

```yaml
time_window_minutes: 3
dbscan_eps: 0.3
dbscan_min_samples: 1
keep_threshold: 0.6
weights:
  gaze: 0.30
  smile: 0.25
  eyes: 0.20
  sharpness: 0.15
  exposure: 0.10
clip_model: "ViT-B-32"
clip_pretrained: "openai"
insightface_model: "buffalo_l"
det_size: [640, 640]
output_dir: "output/curated"
db_path: "photosorter.db"
port: 8080
```

- [ ] **Step 3: Create package markers and pytest config**

`photosorter/__init__.py` — empty file.
`tests/__init__.py` — empty file.

`pytest.ini`:
```ini
[pytest]
testpaths = tests
```

- [ ] **Step 4: Create tests/conftest.py**

```python
import pytest
import numpy as np
from PIL import Image


@pytest.fixture
def tmp_photo_dir(tmp_path):
    for i in range(3):
        arr = np.full((100, 100, 3), i * 80, dtype=np.uint8)
        Image.fromarray(arr).save(str(tmp_path / f"photo_{i}.jpg"))
    return tmp_path


@pytest.fixture
def tmp_db(tmp_path):
    from photosorter.db import init_db
    return init_db(str(tmp_path / "test.db"))
```

- [ ] **Step 5: Install dependencies and verify pytest runs**

```bash
pip install -r requirements.txt
pytest --collect-only
```

Expected: `no tests ran` with 0 errors.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt config.yaml pytest.ini photosorter/__init__.py tests/__init__.py tests/conftest.py
git commit -m "feat: project setup — dependencies, config, test scaffolding"
```

---

## Task 2: Config Module

**Files:**
- Create: `photosorter/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from photosorter.config import load_config, Config, Weights


def test_load_config_defaults(tmp_path):
    cfg_text = """
time_window_minutes: 5
dbscan_eps: 0.4
dbscan_min_samples: 2
keep_threshold: 0.7
weights:
  gaze: 0.40
  smile: 0.20
  eyes: 0.20
  sharpness: 0.10
  exposure: 0.10
clip_model: "ViT-B-32"
clip_pretrained: "openai"
insightface_model: "buffalo_l"
det_size: [640, 640]
output_dir: "out"
db_path: "test.db"
port: 9090
"""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(cfg_text)
    cfg = load_config(str(cfg_file))
    assert isinstance(cfg, Config)
    assert cfg.time_window_minutes == 5
    assert cfg.weights.gaze == pytest.approx(0.40)
    assert cfg.det_size == (640, 640)
    assert cfg.port == 9090
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_config.py -v
```

Expected: `ImportError: cannot import name 'load_config'`

- [ ] **Step 3: Implement photosorter/config.py**

```python
from dataclasses import dataclass, field
import yaml


@dataclass
class Weights:
    gaze: float = 0.30
    smile: float = 0.25
    eyes: float = 0.20
    sharpness: float = 0.15
    exposure: float = 0.10


@dataclass
class Config:
    time_window_minutes: int = 3
    dbscan_eps: float = 0.3
    dbscan_min_samples: int = 1
    keep_threshold: float = 0.6
    weights: Weights = field(default_factory=Weights)
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "openai"
    insightface_model: str = "buffalo_l"
    det_size: tuple = (640, 640)
    output_dir: str = "output/curated"
    db_path: str = "photosorter.db"
    port: int = 8080


def load_config(path: str = "config.yaml") -> Config:
    with open(path) as f:
        data = yaml.safe_load(f)
    weights_data = data.pop("weights", {})
    data["weights"] = Weights(**weights_data)
    if "det_size" in data:
        data["det_size"] = tuple(data["det_size"])
    return Config(**data)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_config.py -v
```

Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add photosorter/config.py tests/test_config.py
git commit -m "feat: config module — load config.yaml into Config dataclass"
```

---

## Task 3: Database Module

**Files:**
- Create: `photosorter/db.py`
- Create: `tests/test_db.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_db.py
import pytest
from photosorter.db import (
    init_db, insert_photo, insert_moment, insert_moment_photos,
    insert_scores, get_next_unreviewed_moment, get_moment_photos,
    submit_moment, undo_last_moment, get_progress,
)


def test_insert_and_fetch_photo(tmp_db):
    pid = insert_photo(tmp_db, "/a/b/c.jpg", 1700000000, 1.23, 4.56, '{}')
    assert pid > 0
    row = tmp_db.execute("SELECT * FROM photos WHERE id=?", (pid,)).fetchone()
    assert row['path'] == "/a/b/c.jpg"
    assert row['timestamp'] == 1700000000


def test_insert_photo_idempotent(tmp_db):
    pid1 = insert_photo(tmp_db, "/x.jpg", 100, None, None, '{}')
    pid2 = insert_photo(tmp_db, "/x.jpg", 100, None, None, '{}')
    assert pid1 == pid2


def test_insert_moment_and_photos(tmp_db):
    pid1 = insert_photo(tmp_db, "/a.jpg", 1000, None, None, '{}')
    pid2 = insert_photo(tmp_db, "/b.jpg", 1001, None, None, '{}')
    mid = insert_moment(tmp_db, 1000, 1001, 2)
    insert_moment_photos(tmp_db, mid, [(pid1, 1), (pid2, 2)])
    rows = get_moment_photos(tmp_db, mid)
    assert len(rows) == 2
    assert rows[0]['rank'] == 1


def test_get_next_unreviewed(tmp_db):
    pid = insert_photo(tmp_db, "/c.jpg", 2000, None, None, '{}')
    mid = insert_moment(tmp_db, 2000, 2000, 1)
    insert_moment_photos(tmp_db, mid, [(pid, 1)])
    moment = get_next_unreviewed_moment(tmp_db)
    assert moment['id'] == mid


def test_submit_and_progress(tmp_db):
    pid = insert_photo(tmp_db, "/d.jpg", 3000, None, None, '{}')
    mid = insert_moment(tmp_db, 3000, 3000, 1)
    insert_moment_photos(tmp_db, mid, [(pid, 1)])
    submit_moment(tmp_db, mid, [pid], {pid: "/out/d.jpg"})
    progress = get_progress(tmp_db)
    assert progress['reviewed'] == 1
    assert progress['total'] == 1


def test_undo_last_moment(tmp_db):
    pid = insert_photo(tmp_db, "/e.jpg", 4000, None, None, '{}')
    mid = insert_moment(tmp_db, 4000, 4000, 1)
    insert_moment_photos(tmp_db, mid, [(pid, 1)])
    submit_moment(tmp_db, mid, [pid], {pid: "/out/e.jpg"})
    undone = undo_last_moment(tmp_db)
    assert undone == mid
    assert get_progress(tmp_db)['reviewed'] == 0
    assert get_next_unreviewed_moment(tmp_db)['id'] == mid
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_db.py -v
```

Expected: `ImportError: cannot import name 'init_db'`

- [ ] **Step 3: Implement photosorter/db.py**

```python
import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS photos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL UNIQUE,
    timestamp INTEGER,
    gps_lat REAL,
    gps_lon REAL,
    exif_json TEXT,
    ingested_at INTEGER DEFAULT (unixepoch())
);
CREATE TABLE IF NOT EXISTS moments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    start_time INTEGER,
    end_time INTEGER,
    photo_count INTEGER,
    reviewed_at INTEGER
);
CREATE TABLE IF NOT EXISTS moment_photos (
    moment_id INTEGER REFERENCES moments(id),
    photo_id INTEGER REFERENCES photos(id),
    rank INTEGER,
    PRIMARY KEY (moment_id, photo_id)
);
CREATE TABLE IF NOT EXISTS scores (
    photo_id INTEGER PRIMARY KEY REFERENCES photos(id),
    gaze_score REAL,
    smile_score REAL,
    eyes_score REAL,
    sharpness_score REAL,
    exposure_score REAL,
    composite_score REAL
);
CREATE TABLE IF NOT EXISTS decisions (
    photo_id INTEGER PRIMARY KEY REFERENCES photos(id),
    keep INTEGER,
    decided_at INTEGER DEFAULT (unixepoch()),
    output_path TEXT
);
"""


def init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def insert_photo(conn, path, timestamp, gps_lat, gps_lon, exif_json) -> int:
    cur = conn.execute(
        "INSERT OR IGNORE INTO photos (path, timestamp, gps_lat, gps_lon, exif_json)"
        " VALUES (?,?,?,?,?)",
        (path, timestamp, gps_lat, gps_lon, exif_json),
    )
    conn.commit()
    if cur.lastrowid:
        return cur.lastrowid
    return conn.execute("SELECT id FROM photos WHERE path=?", (path,)).fetchone()[0]


def insert_moment(conn, start_time, end_time, photo_count) -> int:
    cur = conn.execute(
        "INSERT INTO moments (start_time, end_time, photo_count) VALUES (?,?,?)",
        (start_time, end_time, photo_count),
    )
    conn.commit()
    return cur.lastrowid


def insert_moment_photos(conn, moment_id, photo_ranks: list[tuple[int, int]]) -> None:
    conn.executemany(
        "INSERT OR REPLACE INTO moment_photos (moment_id, photo_id, rank) VALUES (?,?,?)",
        [(moment_id, pid, rank) for pid, rank in photo_ranks],
    )
    conn.commit()


def insert_scores(conn, photo_id, gaze, smile, eyes, sharpness, exposure, composite) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO scores"
        " (photo_id, gaze_score, smile_score, eyes_score, sharpness_score, exposure_score, composite_score)"
        " VALUES (?,?,?,?,?,?,?)",
        (photo_id, gaze, smile, eyes, sharpness, exposure, composite),
    )
    conn.commit()


def get_next_unreviewed_moment(conn):
    return conn.execute(
        "SELECT * FROM moments WHERE reviewed_at IS NULL ORDER BY start_time ASC LIMIT 1"
    ).fetchone()


def get_moment_photos(conn, moment_id) -> list:
    return conn.execute(
        """SELECT p.id as id, p.path as path, p.timestamp as timestamp,
                  s.gaze_score, s.smile_score, s.eyes_score,
                  s.sharpness_score, s.exposure_score, s.composite_score,
                  mp.rank
           FROM moment_photos mp
           JOIN photos p ON p.id = mp.photo_id
           LEFT JOIN scores s ON s.photo_id = p.id
           WHERE mp.moment_id = ?
           ORDER BY mp.rank ASC""",
        (moment_id,),
    ).fetchall()


def submit_moment(conn, moment_id, keep_ids: list[int], output_paths: dict[int, str]) -> None:
    photo_ids = [r[0] for r in conn.execute(
        "SELECT photo_id FROM moment_photos WHERE moment_id=?", (moment_id,)
    ).fetchall()]
    for pid in photo_ids:
        keep = 1 if pid in keep_ids else 0
        conn.execute(
            "INSERT OR REPLACE INTO decisions (photo_id, keep, output_path) VALUES (?,?,?)",
            (pid, keep, output_paths.get(pid)),
        )
    conn.execute("UPDATE moments SET reviewed_at=unixepoch() WHERE id=?", (moment_id,))
    conn.commit()


def undo_last_moment(conn) -> int | None:
    row = conn.execute(
        "SELECT id FROM moments WHERE reviewed_at IS NOT NULL ORDER BY reviewed_at DESC LIMIT 1"
    ).fetchone()
    if not row:
        return None
    moment_id = row[0]
    photo_ids = [r[0] for r in conn.execute(
        "SELECT photo_id FROM moment_photos WHERE moment_id=?", (moment_id,)
    ).fetchall()]
    conn.execute(
        "DELETE FROM decisions WHERE photo_id IN ({})".format(",".join("?" * len(photo_ids))),
        photo_ids,
    )
    conn.execute("UPDATE moments SET reviewed_at=NULL WHERE id=?", (moment_id,))
    conn.commit()
    return moment_id


def get_progress(conn) -> dict:
    total = conn.execute("SELECT COUNT(*) FROM moments").fetchone()[0]
    reviewed = conn.execute(
        "SELECT COUNT(*) FROM moments WHERE reviewed_at IS NOT NULL"
    ).fetchone()[0]
    return {"reviewed": reviewed, "total": total}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_db.py -v
```

Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add photosorter/db.py tests/test_db.py
git commit -m "feat: database module — SQLite schema and all CRUD queries"
```

---

## Task 4: Ingestor

**Files:**
- Create: `photosorter/ingestor.py`
- Create: `tests/test_ingestor.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ingestor.py
import os
import time
from photosorter.ingestor import scan_folder, extract_metadata


def test_scan_finds_jpegs(tmp_photo_dir):
    records = scan_folder(str(tmp_photo_dir))
    assert len(records) == 3
    assert all(r['path'].endswith('.jpg') for r in records)


def test_scan_sorted_by_timestamp(tmp_photo_dir):
    records = scan_folder(str(tmp_photo_dir))
    timestamps = [r['timestamp'] for r in records]
    assert timestamps == sorted(timestamps)


def test_extract_metadata_fallback_to_mtime(tmp_photo_dir):
    path = str(next(tmp_photo_dir.iterdir()))
    record = extract_metadata(path)
    assert record['timestamp'] is not None
    assert abs(record['timestamp'] - int(os.path.getmtime(path))) <= 2


def test_scan_ignores_non_images(tmp_photo_dir):
    (tmp_photo_dir / "notes.txt").write_text("not a photo")
    records = scan_folder(str(tmp_photo_dir))
    assert all(not r['path'].endswith('.txt') for r in records)
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_ingestor.py -v
```

Expected: `ImportError: cannot import name 'scan_folder'`

- [ ] **Step 3: Implement photosorter/ingestor.py**

```python
import os
import json
from pathlib import Path
from datetime import datetime

import pillow_heif
from PIL import Image
from PIL.ExifTags import TAGS

pillow_heif.register_heif_opener()

SUPPORTED = {'.jpg', '.jpeg', '.heic', '.heif', '.png'}

_DATETIME_TAGS = (36867, 306)  # DateTimeOriginal, DateTime


def scan_folder(folder: str) -> list[dict]:
    records = []
    for root, _, files in os.walk(folder):
        for fname in files:
            if Path(fname).suffix.lower() in SUPPORTED:
                path = os.path.join(root, fname)
                records.append(extract_metadata(path))
    return sorted(records, key=lambda r: r['timestamp'])


def extract_metadata(path: str) -> dict:
    exif_data = _read_exif(path)
    timestamp = _extract_timestamp(exif_data, path)
    gps_lat, gps_lon = _extract_gps(exif_data)
    exif_json = json.dumps({
        TAGS.get(k, str(k)): str(v)
        for k, v in exif_data.items()
        if isinstance(v, (str, int, float))
    })
    return {
        'path': path,
        'timestamp': timestamp,
        'gps_lat': gps_lat,
        'gps_lon': gps_lon,
        'exif_json': exif_json,
    }


def _read_exif(path: str) -> dict:
    try:
        img = Image.open(path)
        return img._getexif() or {}
    except Exception:
        return {}


def _extract_timestamp(exif_data: dict, path: str) -> int:
    for tag_id in _DATETIME_TAGS:
        if tag_id in exif_data:
            try:
                dt = datetime.strptime(exif_data[tag_id], "%Y:%m:%d %H:%M:%S")
                return int(dt.timestamp())
            except Exception:
                pass
    return int(os.path.getmtime(path))


def _extract_gps(exif_data: dict) -> tuple[float | None, float | None]:
    GPS_TAG = 34853
    if GPS_TAG not in exif_data:
        return None, None
    gps = exif_data[GPS_TAG]
    try:
        lat = _dms_to_decimal(gps[2], gps[1])
        lon = _dms_to_decimal(gps[4], gps[3])
        return lat, lon
    except Exception:
        return None, None


def _dms_to_decimal(dms, ref: str) -> float:
    d, m, s = float(dms[0]), float(dms[1]), float(dms[2])
    decimal = d + m / 60 + s / 3600
    if ref in ('S', 'W'):
        decimal = -decimal
    return decimal
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_ingestor.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add photosorter/ingestor.py tests/test_ingestor.py
git commit -m "feat: ingestor — folder scan, EXIF extraction, mtime fallback"
```

---

## Task 5: Grouper — Time Windowing

**Files:**
- Create: `photosorter/grouper.py` (time windowing only; CLIP added in Task 6)
- Create: `tests/test_grouper.py` (time windowing tests only)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_grouper.py
import pytest
from photosorter.grouper import time_window_groups


def _photos(timestamps):
    return [{'path': f'/p{i}.jpg', 'timestamp': t, 'id': i}
            for i, t in enumerate(timestamps)]


def test_single_photo_forms_one_group():
    groups = time_window_groups(_photos([1000]), window_minutes=3)
    assert len(groups) == 1
    assert len(groups[0]) == 1


def test_photos_within_window_grouped_together():
    # 0s, 60s, 120s — all within 3-minute window
    groups = time_window_groups(_photos([0, 60, 120]), window_minutes=3)
    assert len(groups) == 1
    assert len(groups[0]) == 3


def test_gap_splits_into_two_groups():
    # 0s and 301s — gap > 3 minutes (180s)
    groups = time_window_groups(_photos([0, 301]), window_minutes=3)
    assert len(groups) == 2


def test_configurable_window():
    groups = time_window_groups(_photos([0, 400]), window_minutes=10)
    assert len(groups) == 1  # 400s < 600s


def test_empty_input():
    assert time_window_groups([], window_minutes=3) == []
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_grouper.py -v
```

Expected: `ImportError: cannot import name 'time_window_groups'`

- [ ] **Step 3: Implement time_window_groups in photosorter/grouper.py**

```python
import numpy as np
from sklearn.cluster import DBSCAN


def time_window_groups(photos: list[dict], window_minutes: int) -> list[list[dict]]:
    if not photos:
        return []
    window_seconds = window_minutes * 60
    groups = [[photos[0]]]
    for photo in photos[1:]:
        if photo['timestamp'] - groups[-1][-1]['timestamp'] > window_seconds:
            groups.append([])
        groups[-1].append(photo)
    return groups
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_grouper.py -v
```

Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add photosorter/grouper.py tests/test_grouper.py
git commit -m "feat: grouper — time windowing (pass 1)"
```

---

## Task 6: Grouper — CLIP Embeddings + DBSCAN

**Files:**
- Modify: `photosorter/grouper.py` (add CLIP + DBSCAN + full `group_photos`)
- Modify: `tests/test_grouper.py` (add clustering tests with mocked CLIP)

- [ ] **Step 1: Add failing tests for clustering**

Append to `tests/test_grouper.py`:

```python
import numpy as np
from unittest.mock import MagicMock, patch
from photosorter.grouper import cluster_by_visual_similarity, group_photos


def test_cluster_single_photo():
    emb = np.array([[1.0, 0.0]])
    labels = cluster_by_visual_similarity(emb, eps=0.3, min_samples=1)
    assert labels.tolist() == [0]


def test_cluster_two_similar_photos():
    emb = np.array([[1.0, 0.0], [0.99, 0.14]])  # very close
    labels = cluster_by_visual_similarity(emb, eps=0.3, min_samples=1)
    assert labels[0] == labels[1]


def test_cluster_two_dissimilar_photos():
    emb = np.array([[1.0, 0.0], [0.0, 1.0]])  # orthogonal
    labels = cluster_by_visual_similarity(emb, eps=0.3, min_samples=1)
    assert labels[0] != labels[1]


def test_noise_points_assigned_to_nearest_cluster():
    # 3 points: two close together, one distant
    emb = np.array([[1.0, 0.0], [0.99, 0.14], [0.0, 1.0]])
    # With min_samples=2, the lone point becomes noise
    labels = cluster_by_visual_similarity(emb, eps=0.3, min_samples=2)
    # noise point should be assigned to the cluster of points 0 and 1
    assert labels[2] == labels[0]


def test_group_photos_splits_visually_distinct():
    photos = _photos([0, 30, 60])  # all in same time window
    # Mock CLIP to return very different embeddings for photos[2]
    mock_model = MagicMock()
    mock_preprocess = MagicMock(side_effect=lambda img: img)

    embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.99, 0.14, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)

    with patch('photosorter.grouper.compute_clip_embeddings', return_value=embeddings):
        moments = group_photos(
            photos, window_minutes=3, eps=0.3, min_samples=1,
            model=mock_model, preprocess=mock_preprocess, device='cpu'
        )
    assert len(moments) == 2
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_grouper.py -v
```

Expected: `ImportError` or `AttributeError` on missing functions.

- [ ] **Step 3: Add CLIP + DBSCAN functions to photosorter/grouper.py**

```python
import torch
from PIL import Image


def compute_clip_embeddings(paths: list[str], model, preprocess, device: str) -> np.ndarray:
    images = [preprocess(Image.open(p).convert('RGB')) for p in paths]
    batch = torch.stack(images).to(device)
    with torch.no_grad():
        feats = model.encode_image(batch).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()


def cluster_by_visual_similarity(embeddings: np.ndarray, eps: float,
                                  min_samples: int) -> np.ndarray:
    if len(embeddings) == 1:
        return np.array([0])
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    labels = db.fit_predict(embeddings)
    noise_mask = labels == -1
    if noise_mask.any():
        non_noise = np.where(~noise_mask)[0]
        if non_noise.size == 0:
            labels[:] = 0
        else:
            for i in np.where(noise_mask)[0]:
                sims = embeddings[i] @ embeddings[non_noise].T
                labels[i] = labels[non_noise[np.argmax(sims)]]
    return labels


def group_photos(photos: list[dict], window_minutes: int, eps: float,
                 min_samples: int, model, preprocess, device: str) -> list[list[dict]]:
    candidate_groups = time_window_groups(photos, window_minutes)
    moments = []
    for group in candidate_groups:
        if len(group) == 1:
            moments.append(group)
            continue
        paths = [p['path'] for p in group]
        embeddings = compute_clip_embeddings(paths, model, preprocess, device)
        labels = cluster_by_visual_similarity(embeddings, eps, min_samples)
        clusters: dict[int, list] = {}
        for photo, label in zip(group, labels):
            clusters.setdefault(int(label), []).append(photo)
        moments.extend(clusters.values())
    return moments
```

- [ ] **Step 4: Run all grouper tests to verify they pass**

```bash
pytest tests/test_grouper.py -v
```

Expected: `11 passed`

- [ ] **Step 5: Commit**

```bash
git add photosorter/grouper.py tests/test_grouper.py
git commit -m "feat: grouper — CLIP embeddings + DBSCAN visual clustering (pass 2)"
```

---

## Task 7: Scorer — Image Quality

**Files:**
- Create: `photosorter/scorer.py` (image quality functions only; face analysis added in Task 8)
- Create: `tests/test_scorer.py` (quality tests only)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_scorer.py
import numpy as np
import pytest
from PIL import Image
from photosorter.scorer import score_image_quality, normalize_sharpness


@pytest.fixture
def sharp_jpg(tmp_path):
    arr = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    path = str(tmp_path / "sharp.jpg")
    Image.fromarray(arr).save(path)
    return path


@pytest.fixture
def blurry_jpg(tmp_path):
    arr = np.full((200, 200, 3), 128, dtype=np.uint8)
    path = str(tmp_path / "blurry.jpg")
    Image.fromarray(arr).save(path)
    return path


def test_score_image_quality_returns_expected_keys(sharp_jpg):
    result = score_image_quality(sharp_jpg)
    assert 'sharpness_raw' in result
    assert 'blur_score' in result
    assert 'exposure' in result


def test_sharp_image_higher_sharpness_raw_than_blurry(sharp_jpg, blurry_jpg):
    sharp_result = score_image_quality(sharp_jpg)
    blurry_result = score_image_quality(blurry_jpg)
    assert sharp_result['sharpness_raw'] > blurry_result['sharpness_raw']


def test_exposure_score_in_range(sharp_jpg):
    result = score_image_quality(sharp_jpg)
    assert 0.0 <= result['exposure'] <= 1.0


def test_normalize_sharpness_max_is_one():
    scores = [
        {'sharpness_raw': 100.0, 'blur_score': 0.5, 'exposure': 0.9},
        {'sharpness_raw': 50.0, 'blur_score': 0.4, 'exposure': 0.8},
    ]
    normalized = normalize_sharpness(scores)
    assert normalized[0]['sharpness'] == pytest.approx(1.0)
    assert normalized[1]['sharpness'] == pytest.approx(0.5, abs=0.01)
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_scorer.py -v
```

Expected: `ImportError: cannot import name 'score_image_quality'`

- [ ] **Step 3: Implement image quality functions in photosorter/scorer.py**

```python
import math
import numpy as np
import cv2
from PIL import Image


def score_image_quality(path: str) -> dict:
    img_bgr = _load_bgr(path)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    sharpness_raw = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    f = np.fft.fft2(gray.astype(np.float32))
    magnitude = np.abs(np.fft.fftshift(f))
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 8
    total = magnitude.sum() + 1e-6
    center = magnitude[cy - r:cy + r, cx - r:cx + r].sum()
    blur_score = float((total - center) / total)

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-6)
    exposure = float(max(0.0, 1.0 - hist[240:].sum() - hist[:16].sum()))

    return {'sharpness_raw': sharpness_raw, 'blur_score': blur_score, 'exposure': exposure}


def normalize_sharpness(scores: list[dict]) -> list[dict]:
    max_raw = max(s['sharpness_raw'] for s in scores) if scores else 1.0
    for s in scores:
        s['sharpness'] = s['sharpness_raw'] / (max_raw + 1e-6)
    return scores


def _load_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is not None:
        return img
    pil = Image.open(path).convert('RGB')
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_scorer.py -v
```

Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add photosorter/scorer.py tests/test_scorer.py
git commit -m "feat: scorer — image quality (sharpness, blur, exposure)"
```

---

## Task 8: Scorer — Face Analysis

**Files:**
- Modify: `photosorter/scorer.py` (add face analysis functions)
- Modify: `tests/test_scorer.py` (add face analysis tests with mocked InsightFace)

- [ ] **Step 1: Add failing tests**

Append to `tests/test_scorer.py`:

```python
import math
from unittest.mock import MagicMock, patch
from photosorter.scorer import score_faces, _eye_aspect_ratio_score, _smile_score


def _make_landmark_68(yaw=0.0, pitch=0.0, eyes_open=True, smiling=True):
    lm = np.zeros((68, 3), dtype=np.float32)
    # Left eye: indices 36-41 (open = tall ellipse)
    eye_h = 0.25 if eyes_open else 0.01
    for j, idx in enumerate([36, 37, 38, 39, 40, 41]):
        lm[idx] = [j * 0.1, eye_h * (1 if j in (1, 2) else 0), 0]
    lm[36] = [0, 0, 0]; lm[39] = [0.6, 0, 0]
    lm[37] = [0.2, eye_h, 0]; lm[38] = [0.4, eye_h, 0]
    lm[40] = [0.4, -eye_h, 0]; lm[41] = [0.2, -eye_h, 0]
    # Right eye: indices 42-47 (mirror)
    for j, idx in enumerate([42, 43, 44, 45, 46, 47]):
        lm[idx] = lm[36 + j] + np.array([1.0, 0, 0])
    # Mouth: 48=left corner, 54=right corner, 51=upper lip, 57=lower lip
    lm[48] = [0.0, 0.5, 0]; lm[54] = [1.0, 0.5, 0]
    if smiling:
        lm[51] = [0.5, 0.7, 0]   # upper lip above corners → smile
    else:
        lm[51] = [0.5, 0.3, 0]   # upper lip below corners → neutral
    lm[57] = [0.5, 0.2, 0]
    return lm


def test_eyes_open_score_open():
    lm = _make_landmark_68(eyes_open=True)
    score = _eye_aspect_ratio_score(lm)
    assert score == 1.0


def test_smile_score_smiling_higher_than_neutral():
    lm_smile = _make_landmark_68(smiling=True)
    lm_neutral = _make_landmark_68(smiling=False)
    assert _smile_score(lm_smile) > _smile_score(lm_neutral)


def test_score_faces_no_faces_returns_midpoint(sharp_jpg):
    mock_analyzer = MagicMock()
    mock_analyzer.get.return_value = []
    result = score_faces(sharp_jpg, mock_analyzer)
    assert result == {'gaze_score': 0.5, 'smile_score': 0.5, 'eyes_score': 0.5}


def test_score_faces_averages_across_faces(sharp_jpg):
    face1 = MagicMock()
    face1.pose = np.array([0.0, 0.0, 0.0])
    face1.landmark_3d_68 = _make_landmark_68(eyes_open=True, smiling=True)

    face2 = MagicMock()
    face2.pose = np.array([45.0, 0.0, 0.0])  # looking away
    face2.landmark_3d_68 = _make_landmark_68(eyes_open=True, smiling=False)

    mock_analyzer = MagicMock()
    mock_analyzer.get.return_value = [face1, face2]

    result = score_faces(sharp_jpg, mock_analyzer)
    # gaze for face1=1.0, face2=0.0 → average=0.5
    assert result['gaze_score'] == pytest.approx(0.5, abs=0.05)
    assert 0.0 <= result['smile_score'] <= 1.0
    assert result['eyes_score'] == 1.0
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_scorer.py -v
```

Expected: `ImportError` on `score_faces`.

- [ ] **Step 3: Add face analysis functions to photosorter/scorer.py**

```python
def score_faces(path: str, face_analyzer) -> dict:
    img_bgr = _load_bgr(path)
    faces = face_analyzer.get(img_bgr)
    if not faces:
        return {'gaze_score': 0.5, 'smile_score': 0.5, 'eyes_score': 0.5}

    gazes, smiles, eyes = [], [], []
    for face in faces:
        pose = getattr(face, 'pose', None)
        if pose is not None:
            yaw, pitch = float(pose[0]), float(pose[1])
            gazes.append(max(0.0, 1.0 - math.sqrt(yaw ** 2 + pitch ** 2) / 45.0))
        else:
            gazes.append(0.5)

        lm = getattr(face, 'landmark_3d_68', None)
        if lm is not None:
            eyes.append(_eye_aspect_ratio_score(lm))
            smiles.append(_smile_score(lm))
        else:
            eyes.append(0.5)
            smiles.append(0.5)

    return {
        'gaze_score': float(np.mean(gazes)),
        'smile_score': float(np.mean(smiles)),
        'eyes_score': float(np.mean(eyes)),
    }


def _eye_aspect_ratio_score(lm: np.ndarray) -> float:
    def ear(indices):
        p = lm[indices]
        v1 = np.linalg.norm(p[1] - p[5])
        v2 = np.linalg.norm(p[2] - p[4])
        h = np.linalg.norm(p[0] - p[3]) + 1e-6
        return (v1 + v2) / (2.0 * h)

    avg = (ear([36, 37, 38, 39, 40, 41]) + ear([42, 43, 44, 45, 46, 47])) / 2.0
    return 1.0 if avg > 0.2 else 0.0


def _smile_score(lm: np.ndarray) -> float:
    left_corner = lm[48]
    right_corner = lm[54]
    upper_lip = lm[51]
    mouth_width = np.linalg.norm(right_corner - left_corner) + 1e-6
    avg_corner_y = (left_corner[1] + right_corner[1]) / 2.0
    uplift = float(upper_lip[1] - avg_corner_y)
    return float(min(1.0, max(0.0, uplift / (mouth_width * 0.3))))
```

Also add `import math` at the top of scorer.py if not already present.

- [ ] **Step 4: Run all scorer tests to verify they pass**

```bash
pytest tests/test_scorer.py -v
```

Expected: `10 passed`

- [ ] **Step 5: Commit**

```bash
git add photosorter/scorer.py tests/test_scorer.py
git commit -m "feat: scorer — face analysis (gaze, smile, eyes) via InsightFace landmarks"
```

---

## Task 9: Ranker

**Files:**
- Create: `photosorter/ranker.py`
- Create: `tests/test_ranker.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ranker.py
import pytest
from photosorter.config import Weights
from photosorter.ranker import rank_moment, suggested_keep_ids, PhotoScore


WEIGHTS = Weights(gaze=0.30, smile=0.25, eyes=0.20, sharpness=0.15, exposure=0.10)


def _make_photo_scores(scores_list):
    keys = ['photo_id', 'gaze_score', 'smile_score', 'eyes_score',
            'sharpness_raw', 'sharpness', 'blur_score', 'exposure']
    return [dict(zip(keys, row)) for row in scores_list]


def test_rank_moment_best_first():
    photos = _make_photo_scores([
        (1, 0.9, 0.8, 1.0, 100, 1.0, 0.8, 0.9),
        (2, 0.1, 0.1, 0.0, 10,  0.1, 0.2, 0.5),
    ])
    ranked = rank_moment(photos, WEIGHTS, keep_threshold=0.6)
    assert ranked[0].photo_id == 1
    assert ranked[0].rank == 1
    assert ranked[1].rank == 2


def test_rank_moment_composite_in_range():
    photos = _make_photo_scores([
        (1, 1.0, 1.0, 1.0, 100, 1.0, 1.0, 1.0),
    ])
    ranked = rank_moment(photos, WEIGHTS, keep_threshold=0.6)
    assert 0.0 <= ranked[0].composite <= 1.0


def test_suggested_keep_ids_filters_below_threshold():
    ranked = [
        PhotoScore(photo_id=1, gaze=0.9, smile=0.9, eyes=1.0,
                   sharpness=0.9, exposure=0.9, composite=0.92, rank=1),
        PhotoScore(photo_id=2, gaze=0.5, smile=0.5, eyes=1.0,
                   sharpness=0.5, exposure=0.5, composite=0.55, rank=2),
        PhotoScore(photo_id=3, gaze=0.1, smile=0.1, eyes=0.0,
                   sharpness=0.1, exposure=0.1, composite=0.10, rank=3),
    ]
    # threshold=0.6: photos above 0.6*0.92=0.552 are suggested
    keep = suggested_keep_ids(ranked, keep_threshold=0.6)
    assert 1 in keep
    assert 2 in keep
    assert 3 not in keep


def test_suggested_keep_single_photo():
    ranked = [
        PhotoScore(photo_id=1, gaze=0.5, smile=0.5, eyes=0.5,
                   sharpness=0.5, exposure=0.5, composite=0.5, rank=1),
    ]
    keep = suggested_keep_ids(ranked, keep_threshold=0.6)
    assert keep == [1]
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_ranker.py -v
```

Expected: `ImportError: cannot import name 'rank_moment'`

- [ ] **Step 3: Implement photosorter/ranker.py**

```python
from dataclasses import dataclass


@dataclass
class PhotoScore:
    photo_id: int
    gaze: float
    smile: float
    eyes: float
    sharpness: float
    exposure: float
    composite: float
    rank: int


def rank_moment(photo_scores: list[dict], weights, keep_threshold: float) -> list[PhotoScore]:
    computed = []
    for ps in photo_scores:
        sharpness = (ps.get('sharpness', 0.0) + ps.get('blur_score', 0.0)) / 2.0
        composite = (
            ps['gaze_score'] * weights.gaze
            + ps['smile_score'] * weights.smile
            + ps['eyes_score'] * weights.eyes
            + sharpness * weights.sharpness
            + ps['exposure'] * weights.exposure
        )
        computed.append({
            'photo_id': ps['photo_id'],
            'gaze': ps['gaze_score'],
            'smile': ps['smile_score'],
            'eyes': ps['eyes_score'],
            'sharpness': sharpness,
            'exposure': ps['exposure'],
            'composite': float(composite),
        })

    computed.sort(key=lambda x: x['composite'], reverse=True)

    return [
        PhotoScore(**c, rank=rank)
        for rank, c in enumerate(computed, start=1)
    ]


def suggested_keep_ids(ranked: list[PhotoScore], keep_threshold: float) -> list[int]:
    if not ranked:
        return []
    threshold = ranked[0].composite * keep_threshold
    return [ps.photo_id for ps in ranked if ps.composite >= threshold]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_ranker.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add photosorter/ranker.py tests/test_ranker.py
git commit -m "feat: ranker — composite weighted scoring and keep suggestion"
```

---

## Task 10: Outputter

**Files:**
- Create: `photosorter/outputter.py`
- Create: `tests/test_outputter.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_outputter.py
import os
from photosorter.outputter import output_photo


def test_output_photo_creates_file(tmp_path, tmp_photo_dir):
    src = str(next(tmp_photo_dir.iterdir()))
    out_dir = str(tmp_path / "curated")
    dest = output_photo(src, 1700000000, out_dir)
    assert os.path.exists(dest)


def test_output_photo_in_dated_subdir(tmp_path, tmp_photo_dir):
    src = str(next(tmp_photo_dir.iterdir()))
    out_dir = str(tmp_path / "curated")
    dest = output_photo(src, 1700000000, out_dir)
    # 1700000000 is 2023-11-14
    assert "2023-11-14" in dest


def test_output_photo_no_duplicates(tmp_path, tmp_photo_dir):
    src = str(next(tmp_photo_dir.iterdir()))
    out_dir = str(tmp_path / "curated")
    dest1 = output_photo(src, 1700000000, out_dir)
    dest2 = output_photo(src, 1700000000, out_dir)
    assert dest1 != dest2
    assert os.path.exists(dest1)
    assert os.path.exists(dest2)
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_outputter.py -v
```

Expected: `ImportError: cannot import name 'output_photo'`

- [ ] **Step 3: Implement photosorter/outputter.py**

```python
import os
import shutil
from datetime import datetime


def output_photo(src_path: str, timestamp: int, output_dir: str) -> str:
    date_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
    dest_dir = os.path.join(output_dir, date_str)
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = _unique_path(dest_dir, os.path.basename(src_path))
    try:
        os.link(src_path, dest_path)
    except OSError:
        shutil.copy2(src_path, dest_path)
    return dest_path


def _unique_path(dest_dir: str, filename: str) -> str:
    base, ext = os.path.splitext(filename)
    candidate = os.path.join(dest_dir, filename)
    counter = 1
    while os.path.exists(candidate):
        candidate = os.path.join(dest_dir, f"{base}_{counter}{ext}")
        counter += 1
    return candidate
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_outputter.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add photosorter/outputter.py tests/test_outputter.py
git commit -m "feat: outputter — hard-link curated photos into dated output folder"
```

---

## Task 11: Pipeline CLI

**Files:**
- Create: `pipeline.py`

No dedicated unit tests for this task — it is integration of all prior modules. Verify manually with a small test folder.

- [ ] **Step 1: Create pipeline.py**

```python
import argparse
import torch
import open_clip
from insightface.app import FaceAnalysis

from photosorter.config import load_config
from photosorter.db import (
    init_db, insert_photo, insert_moment, insert_moment_photos, insert_scores,
)
from photosorter.ingestor import scan_folder
from photosorter.grouper import group_photos
from photosorter.scorer import score_image_quality, normalize_sharpness, score_faces
from photosorter.ranker import rank_moment, suggested_keep_ids


def main():
    parser = argparse.ArgumentParser(description='PhotoSorter pipeline')
    parser.add_argument('--input', required=True, help='Folder of photos to process')
    parser.add_argument('--config', default='config.yaml', help='Path to config.yaml')
    args = parser.parse_args()

    cfg = load_config(args.config)
    conn = init_db(cfg.db_path)

    print("Scanning folder...")
    photos = scan_folder(args.input)
    for p in photos:
        pid = insert_photo(conn, p['path'], p['timestamp'],
                           p['gps_lat'], p['gps_lon'], p['exif_json'])
        p['id'] = pid
    print(f"Found {len(photos)} photos")

    if not photos:
        print("No photos found. Exiting.")
        return

    print("Loading CLIP model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _, preprocess = open_clip.create_model_and_transforms(
        cfg.clip_model, pretrained=cfg.clip_pretrained
    )
    model = model.to(device).eval()

    print("Grouping photos into moments...")
    moments = group_photos(
        photos, cfg.time_window_minutes, cfg.dbscan_eps,
        cfg.dbscan_min_samples, model, preprocess, device,
    )
    print(f"Found {len(moments)} moments")

    print("Loading InsightFace model...")
    face_analyzer = FaceAnalysis(
        name=cfg.insightface_model,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    )
    face_analyzer.prepare(ctx_id=0, det_size=tuple(cfg.det_size))

    print("Scoring and ranking photos...")
    for i, moment_photos in enumerate(moments):
        print(f"  Moment {i + 1}/{len(moments)} ({len(moment_photos)} photos)", end='\r')

        quality_scores = []
        face_score_list = []
        for photo in moment_photos:
            qs = score_image_quality(photo['path'])
            fs = score_faces(photo['path'], face_analyzer)
            quality_scores.append({**qs, 'photo_id': photo['id']})
            face_score_list.append({**fs, 'photo_id': photo['id']})

        quality_scores = normalize_sharpness(quality_scores)

        combined = [{**qs, **fs} for qs, fs in zip(quality_scores, face_score_list)]
        ranked = rank_moment(combined, cfg.weights, cfg.keep_threshold)

        timestamps = [p['timestamp'] for p in moment_photos]
        moment_id = insert_moment(conn, min(timestamps), max(timestamps), len(moment_photos))
        insert_moment_photos(conn, moment_id, [(ps.photo_id, ps.rank) for ps in ranked])
        for ps in ranked:
            insert_scores(conn, ps.photo_id, ps.gaze, ps.smile, ps.eyes,
                          ps.sharpness, ps.exposure, ps.composite)

    print(f"\nDone. {len(moments)} moments stored in {cfg.db_path}")
    print(f"Start the review server: uvicorn photosorter.api:app --port {cfg.port}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run a smoke test on the tmp_photo_dir fixture photos**

Create a temporary folder with a few JPEGs and run the pipeline:

```bash
mkdir -p /tmp/test_photos
python -c "
from PIL import Image
import numpy as np
for i in range(4):
    arr = np.random.randint(0,255,(200,200,3),dtype=np.uint8)
    Image.fromarray(arr).save(f'/tmp/test_photos/photo_{i}.jpg')
"
python pipeline.py --input /tmp/test_photos
```

Expected: pipeline completes, prints "Done. N moments stored in photosorter.db", no Python traceback.

- [ ] **Step 3: Verify DB has data**

```bash
python -c "
import sqlite3
conn = sqlite3.connect('photosorter.db')
print('photos:', conn.execute('SELECT COUNT(*) FROM photos').fetchone()[0])
print('moments:', conn.execute('SELECT COUNT(*) FROM moments').fetchone()[0])
print('scores:', conn.execute('SELECT COUNT(*) FROM scores').fetchone()[0])
"
```

Expected: non-zero counts for photos, moments, and scores.

- [ ] **Step 4: Commit**

```bash
git add pipeline.py
git commit -m "feat: pipeline CLI — orchestrates ingest, group, score, rank, store"
```

---

## Task 12: API

**Files:**
- Create: `photosorter/api.py`
- Create: `tests/test_api.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from photosorter.db import (
    init_db, insert_photo, insert_moment, insert_moment_photos, insert_scores,
)


@pytest.fixture
def seeded_db(tmp_path):
    conn = init_db(str(tmp_path / "test.db"))
    pid1 = insert_photo(conn, "/a.jpg", 1700000000, None, None, '{}')
    pid2 = insert_photo(conn, "/b.jpg", 1700000060, None, None, '{}')
    mid = insert_moment(conn, 1700000000, 1700000060, 2)
    insert_moment_photos(conn, mid, [(pid1, 1), (pid2, 2)])
    insert_scores(conn, pid1, 0.9, 0.8, 1.0, 0.9, 0.9, 0.88)
    insert_scores(conn, pid2, 0.5, 0.4, 1.0, 0.5, 0.8, 0.58)
    return conn, mid, pid1, pid2


@pytest.fixture
def client(seeded_db, tmp_path, monkeypatch):
    import photosorter.api as api_module
    conn, _, _, _ = seeded_db
    monkeypatch.setattr(api_module, 'conn', conn)
    # Use a config with the tmp output dir
    from photosorter.config import Config, Weights
    cfg = Config(output_dir=str(tmp_path / "curated"), db_path=str(tmp_path / "test.db"))
    monkeypatch.setattr(api_module, 'cfg', cfg)
    return TestClient(api_module.app)


def test_progress(client, seeded_db):
    r = client.get("/api/progress")
    assert r.status_code == 200
    data = r.json()
    assert data['total'] == 1
    assert data['reviewed'] == 0


def test_next_moment_returns_photos(client):
    r = client.get("/api/moments/next")
    assert r.status_code == 200
    data = r.json()
    assert 'moment_id' in data
    assert len(data['photos']) == 2
    assert data['photos'][0]['rank'] == 1


def test_submit_moment(client, seeded_db, tmp_path):
    _, mid, pid1, _ = seeded_db
    # Create the actual file so output_photo can link it
    import os
    os.makedirs(str(tmp_path / "photos"), exist_ok=True)
    open(str(tmp_path / "photos" / "a.jpg"), 'w').close()

    # Patch the photo path in DB to point to existing file
    conn, _, _, _ = seeded_db
    conn.execute("UPDATE photos SET path=? WHERE id=?",
                 (str(tmp_path / "photos" / "a.jpg"), pid1))
    conn.commit()

    r = client.post(f"/api/moments/{mid}/submit", json={"keep_ids": [pid1]})
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_undo(client, seeded_db):
    _, mid, pid1, _ = seeded_db
    conn, _, _, _ = seeded_db
    # Submit first (with no actual file hard-link needed — use empty output_paths via mock)
    from photosorter import db as db_module
    db_module.submit_moment(conn, mid, [pid1], {})
    r = client.post("/api/undo")
    assert r.status_code == 200
    assert r.json()["undone_moment_id"] == mid


def test_next_moment_done_when_all_reviewed(client, seeded_db):
    _, mid, pid1, _ = seeded_db
    conn, _, _, _ = seeded_db
    from photosorter import db as db_module
    db_module.submit_moment(conn, mid, [pid1], {})
    r = client.get("/api/moments/next")
    assert r.json().get("done") is True
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_api.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` on `photosorter.api`.

- [ ] **Step 3: Implement photosorter/api.py**

```python
import os
import sqlite3

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import load_config, Config
from .db import (
    init_db, get_next_unreviewed_moment, get_moment_photos,
    get_progress, submit_moment, undo_last_moment,
)
from .outputter import output_photo

app = FastAPI()
cfg: Config = None
conn: sqlite3.Connection = None


@app.on_event("startup")
def startup():
    global conn, cfg
    cfg = load_config()
    conn = init_db(cfg.db_path)


class SubmitBody(BaseModel):
    keep_ids: list[int]


@app.get("/api/progress")
def api_progress():
    return get_progress(conn)


@app.get("/api/moments/next")
def api_next_moment():
    moment = get_next_unreviewed_moment(conn)
    if moment is None:
        return {"done": True}
    photos = get_moment_photos(conn, moment['id'])
    top_score = photos[0]['composite_score'] if photos else 0.0
    progress = get_progress(conn)
    return {
        "moment_id": moment['id'],
        "photos": [
            {
                "photo_id": p['id'],
                "filename": os.path.basename(p['path']),
                "rank": p['rank'],
                "scores": {
                    "gaze": p['gaze_score'],
                    "smile": p['smile_score'],
                    "eyes": p['eyes_score'],
                    "sharpness": p['sharpness_score'],
                    "exposure": p['exposure_score'],
                    "composite": p['composite_score'],
                },
                "suggested_keep": (
                    p['composite_score'] is not None
                    and top_score is not None
                    and p['composite_score'] >= top_score * cfg.keep_threshold
                ),
            }
            for p in photos
        ],
        "reviewed": progress['reviewed'],
        "total": progress['total'],
    }


@app.post("/api/moments/{moment_id}/submit")
def api_submit(moment_id: int, body: SubmitBody):
    photos = get_moment_photos(conn, moment_id)
    photo_map = {p['id']: p for p in photos}
    output_paths = {}
    for pid in body.keep_ids:
        if pid not in photo_map:
            raise HTTPException(status_code=400, detail=f"Photo {pid} not in moment {moment_id}")
        p = photo_map[pid]
        if os.path.exists(p['path']):
            output_paths[pid] = output_photo(p['path'], p['timestamp'], cfg.output_dir)
    submit_moment(conn, moment_id, body.keep_ids, output_paths)
    return {"ok": True}


@app.post("/api/undo")
def api_undo():
    mid = undo_last_moment(conn)
    if mid is None:
        raise HTTPException(status_code=400, detail="Nothing to undo")
    return {"ok": True, "undone_moment_id": mid}


@app.get("/api/photos/{photo_id}")
def api_photo(photo_id: int):
    row = conn.execute("SELECT path FROM photos WHERE id=?", (photo_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404)
    return FileResponse(row['path'])


app.mount("/", StaticFiles(directory="ui", html=True), name="ui")
```

- [ ] **Step 4: Create an empty ui/ directory so StaticFiles doesn't crash during tests**

```bash
mkdir -p ui
touch ui/.gitkeep
```

- [ ] **Step 5: Run all API tests to verify they pass**

```bash
pytest tests/test_api.py -v
```

Expected: `6 passed`

- [ ] **Step 6: Commit**

```bash
git add photosorter/api.py tests/test_api.py ui/.gitkeep
git commit -m "feat: FastAPI review API — progress, next moment, submit, undo, serve photos"
```

---

## Task 13: Web UI

**Files:**
- Create: `ui/index.html`
- Create: `ui/app.js`

No automated tests for this task — verify manually in a browser via SSH tunnel after running the full pipeline.

- [ ] **Step 1: Create ui/index.html**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>PhotoSorter</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: system-ui, sans-serif; background: #111; color: #eee; min-height: 100vh; }
    #progress-bar-wrap { background: #333; height: 6px; }
    #progress-bar { background: #4caf50; height: 6px; width: 0; transition: width .3s; }
    #progress-label { text-align: center; padding: 8px; font-size: 13px; color: #aaa; }
    #grid { display: flex; flex-wrap: wrap; gap: 12px; padding: 16px; justify-content: center; }
    .card {
      position: relative; cursor: pointer; border: 3px solid #555;
      border-radius: 6px; overflow: hidden; width: 220px; transition: border-color .15s;
    }
    .card.keep { border-color: #4caf50; }
    .card img { width: 100%; height: 160px; object-fit: cover; display: block; }
    .card-info { padding: 6px 8px; font-size: 11px; background: #1e1e1e; }
    .card-rank { font-weight: bold; font-size: 13px; margin-bottom: 4px; }
    .bar-row { display: flex; align-items: center; gap: 4px; margin-top: 2px; }
    .bar-label { width: 58px; color: #aaa; font-size: 10px; }
    .bar-bg { flex: 1; background: #333; height: 5px; border-radius: 3px; }
    .bar-fill { height: 5px; border-radius: 3px; background: #4caf50; }
    .bar-val { width: 26px; text-align: right; color: #777; font-size: 10px; }
    .badge {
      position: absolute; top: 6px; left: 6px; background: rgba(0,0,0,.6);
      color: #eee; font-size: 12px; font-weight: bold; padding: 2px 7px; border-radius: 10px;
    }
    #footer {
      position: sticky; bottom: 0; display: flex; justify-content: center;
      align-items: center; gap: 20px; padding: 14px 20px;
      background: #1a1a1a; border-top: 1px solid #333;
    }
    #submit-btn {
      background: #4caf50; color: white; border: none; padding: 10px 36px;
      font-size: 16px; border-radius: 6px; cursor: pointer; font-weight: bold;
    }
    #submit-btn:hover { background: #43a047; }
    #shortcuts { color: #555; font-size: 11px; line-height: 1.8; }
    #done-screen { text-align: center; padding: 100px 20px; font-size: 22px; color: #aaa; }
  </style>
</head>
<body>
  <div id="progress-bar-wrap"><div id="progress-bar"></div></div>
  <div id="progress-label">Loading...</div>
  <div id="grid"></div>
  <div id="footer">
    <button id="submit-btn">Submit &rarr;</button>
    <div id="shortcuts">
      <b>→ / Enter</b> submit &nbsp;|&nbsp; <b>1–9</b> toggle photo &nbsp;|&nbsp;
      <b>A</b> all/none &nbsp;|&nbsp; <b>Z</b> undo
    </div>
  </div>
  <script src="app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Create ui/app.js**

```javascript
let current = null;
let keepIds = new Set();

async function loadNext() {
  const r = await fetch('/api/moments/next');
  const data = await r.json();
  if (data.done) {
    document.getElementById('grid').innerHTML =
      '<div id="done-screen">All moments reviewed! Curated photos are in output/curated/</div>';
    document.getElementById('footer').style.display = 'none';
    document.getElementById('progress-label').textContent = 'Complete';
    return;
  }
  current = data;
  keepIds = new Set(data.photos.filter(p => p.suggested_keep).map(p => p.photo_id));
  setProgress(data.reviewed, data.total);
  renderGrid(data.photos);
}

function setProgress(reviewed, total) {
  const pct = total ? (reviewed / total * 100).toFixed(1) : 0;
  document.getElementById('progress-bar').style.width = pct + '%';
  document.getElementById('progress-label').textContent =
    `Moment ${reviewed + 1} of ${total} — ${pct}% complete`;
}

function renderGrid(photos) {
  const grid = document.getElementById('grid');
  grid.innerHTML = '';
  photos.forEach((p, i) => {
    const card = document.createElement('div');
    card.className = 'card' + (keepIds.has(p.photo_id) ? ' keep' : '');
    card.dataset.id = p.photo_id;
    card.innerHTML = `
      <div class="badge">${i + 1}</div>
      <img src="/api/photos/${p.photo_id}" loading="lazy" alt="${p.filename}">
      <div class="card-info">
        <div class="card-rank">#${p.rank} &mdash; ${pct(p.scores.composite)}%</div>
        ${bar('Gaze', p.scores.gaze)}
        ${bar('Smile', p.scores.smile)}
        ${bar('Eyes', p.scores.eyes)}
        ${bar('Sharpness', p.scores.sharpness)}
        ${bar('Exposure', p.scores.exposure)}
      </div>`;
    card.addEventListener('click', () => toggle(p.photo_id));
    grid.appendChild(card);
  });
}

function pct(v) { return ((v || 0) * 100).toFixed(0); }

function bar(label, value) {
  return `<div class="bar-row">
    <span class="bar-label">${label}</span>
    <div class="bar-bg"><div class="bar-fill" style="width:${pct(value)}%"></div></div>
    <span class="bar-val">${pct(value)}%</span>
  </div>`;
}

function toggle(id) {
  keepIds.has(id) ? keepIds.delete(id) : keepIds.add(id);
  document.querySelectorAll('.card').forEach(c => {
    c.className = 'card' + (keepIds.has(parseInt(c.dataset.id)) ? ' keep' : '');
  });
}

async function submit() {
  if (!current) return;
  await fetch(`/api/moments/${current.moment_id}/submit`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({keep_ids: [...keepIds]}),
  });
  loadNext();
}

async function undo() {
  await fetch('/api/undo', {method: 'POST'});
  loadNext();
}

document.getElementById('submit-btn').addEventListener('click', submit);

document.addEventListener('keydown', e => {
  if (e.key === 'ArrowRight' || e.key === 'Enter') { submit(); return; }
  if (e.key === 'z' || e.key === 'Z') { undo(); return; }
  if (e.key === 'a' || e.key === 'A') {
    const allKept = current?.photos.every(p => keepIds.has(p.photo_id));
    if (allKept) keepIds.clear(); else current?.photos.forEach(p => keepIds.add(p.photo_id));
    renderGrid(current?.photos || []);
    return;
  }
  const n = parseInt(e.key);
  if (n >= 1 && n <= 9 && current?.photos[n - 1]) toggle(current.photos[n - 1].photo_id);
});

loadNext();
```

- [ ] **Step 3: Remove the placeholder .gitkeep from ui/**

```bash
rm ui/.gitkeep
```

- [ ] **Step 4: Run the full stack end-to-end**

```bash
# Ensure pipeline has run (Task 11 smoke test should have populated photosorter.db)
uvicorn photosorter.api:app --port 8080
```

Open `http://localhost:8080` (via SSH tunnel if remote). Verify:
- Photos appear in a grid, best-ranked first
- Green border shows suggested keeps
- Number keys toggle photos
- Enter submits and advances to next moment
- Progress bar advances
- "A" selects/deselects all
- "Z" undoes last submission

- [ ] **Step 5: Run full test suite to confirm nothing regressed**

```bash
pytest -v
```

Expected: all tests pass (no failures).

- [ ] **Step 6: Commit**

```bash
git add ui/index.html ui/app.js
git commit -m "feat: web review UI — photo grid, score bars, keyboard shortcuts"
```

---

## Checkpoint: Full System Verification

- [ ] Run `pytest -v` — all tests green
- [ ] Run pipeline on a real photo folder: `python pipeline.py --input /path/to/photos`
- [ ] Start server: `uvicorn photosorter.api:app --port 8080`
- [ ] Open `http://localhost:8080` in browser (SSH tunnel if remote)
- [ ] Review 3 moments: submit, undo, re-submit
- [ ] Verify `output/curated/YYYY-MM-DD/` contains selected photos
