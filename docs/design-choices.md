# PhotoSorter: Design Choices

This document explains every significant design decision in PhotoSorter — what was chosen, why, what the alternatives were, and what you give up. It is written as an interview-prep guide: by the end you should be able to explain any choice confidently, including its trade-offs.

---

## Table of Contents

1. [Overall Architecture: Batch Pipeline + Separate Server](#1-overall-architecture-batch-pipeline--separate-server)
2. [Storage: SQLite Over a Full Database](#2-storage-sqlite-over-a-full-database)
3. [Photo Grouping: Two-Pass Strategy (Time Windows → Visual Clustering)](#3-photo-grouping-two-pass-strategy-time-windows--visual-clustering)
4. [Visual Similarity: CLIP Embeddings](#4-visual-similarity-clip-embeddings)
5. [Clustering: DBSCAN Over K-Means or Hierarchical](#5-clustering-dbscan-over-k-means-or-hierarchical)
6. [Face Analysis: InsightFace Landmark-Based Scoring](#6-face-analysis-insightface-landmark-based-scoring)
7. [Image Quality: OpenCV Laplacian + FFT + Histogram](#7-image-quality-opencv-laplacian--fft--histogram)
8. [Scoring: Weighted Linear Combination](#8-scoring-weighted-linear-combination)
9. [Keep Suggestion: Threshold Relative to Top Score](#9-keep-suggestion-threshold-relative-to-top-score)
10. [Output: Hard Links Instead of Copies](#10-output-hard-links-instead-of-copies)
11. [API: FastAPI Over Flask or Django](#11-api-fastapi-over-flask-or-django)
12. [Frontend: Vanilla JS Over React or Vue](#12-frontend-vanilla-js-over-react-or-vue)
13. [HEIC Support: pillow-heif Registration at Import Time](#13-heic-support-pillow-heif-registration-at-import-time)
14. [EXIF Reading: Public API First, Private Fallback](#14-exif-reading-public-api-first-private-fallback)
15. [Idempotent Inserts: INSERT OR IGNORE + rowcount Check](#15-idempotent-inserts-insert-or-ignore--rowcount-check)
16. [Thread Safety: check_same_thread=False](#16-thread-safety-check_same_threadfalse)
17. [Configuration: YAML Dataclass With Strict Validation](#17-configuration-yaml-dataclass-with-strict-validation)
18. [GPS Coordinates: DMS to Decimal Conversion](#18-gps-coordinates-dms-to-decimal-conversion)
19. [Noise Handling in DBSCAN: Reassign to Nearest Cluster](#19-noise-handling-in-dbscan-reassign-to-nearest-cluster)

---

## 1. Overall Architecture: Batch Pipeline + Separate Server

### What we did

Separated the app into two independent processes:

1. **`pipeline.py`** — a one-shot CLI that ingests, groups, scores, ranks, and writes to SQLite. Run it once before curation.
2. **`uvicorn photosorter.api:app`** — a FastAPI server that reads from SQLite and serves the review UI.

### Why

**Separation of concerns.** The heavy AI work (CLIP, InsightFace) is slow and GPU-intensive. If you mix it into the web server, every request to load the next moment would potentially trigger model inference. Separating them means:

- The web server is fast and responsive — it only reads from SQLite.
- The pipeline can be re-run, tweaked, or debugged without touching the server.
- You can run the pipeline overnight and do curation the next morning.

**Alternatives considered:**

| Alternative | Why we didn't use it |
|---|---|
| Single process (pipeline embedded in server, run on first request) | Slow first-load, hard to interrupt, mixed concerns |
| Worker queue (Celery + Redis) | Massive overkill for a single-user local tool |
| Streaming pipeline (process → UI live) | Complex; SQLite as intermediary is simpler and recoverable |

### Interview answer

> "I decoupled the batch pipeline from the review server because the AI inference is slow and GPU-intensive — it doesn't belong in the hot path of the web server. The pipeline writes to SQLite once, and the server just reads. This makes the UI fast and lets me re-run the pipeline independently if I change scoring weights."

---

## 2. Storage: SQLite Over a Full Database

### What we did

Used Python's built-in `sqlite3` module with five tables: `photos`, `moments`, `moment_photos`, `scores`, `decisions`.

### Why

**SQLite is the right tool for a local single-user application.** The database stores the results of an analysis run. At most it holds tens of thousands of rows (one per photo). There are no concurrent writers, no remote clients, and no need for connection pooling.

SQLite advantages in this context:
- Zero setup — no server process, no credentials, no port
- The entire database is a single `.db` file — easy to inspect, copy, or delete
- Python ships with `sqlite3` in the standard library — no extra dependencies
- Plenty fast enough: all queries are indexed lookups on primary keys

**Alternatives and why they were rejected:**

| Alternative | Why not |
|---|---|
| PostgreSQL / MySQL | Require a running server, credentials, network config — overkill for a local tool |
| MongoDB | No joins; moment↔photo relationships are naturally relational |
| JSON files on disk | No transactions, no referential integrity, hard to query |
| Pandas DataFrames in memory | Lost on restart; no persistence |

**The one trade-off:** SQLite has weak concurrency. If two processes write simultaneously, the last writer wins and the first may block. This is acceptable here because only `pipeline.py` writes and only the FastAPI server reads (with the exception of the `submit` endpoint, which writes decisions — but it's single-user so there's never true concurrency).

### Interview answer

> "SQLite is the standard recommendation for local, single-user tools. It's zero-config, ships with Python, and the whole database is one file. I have relational data — moments contain multiple photos, and decisions reference photos — so a key-value store or JSON files would make the queries harder. PostgreSQL would have been overkill: no server process needed."

---

## 3. Photo Grouping: Two-Pass Strategy (Time Windows → Visual Clustering)

### What we did

**Pass 1 — Time windows:** Split the photo stream into coarse groups wherever the gap between consecutive photos exceeds 3 minutes.

**Pass 2 — Visual clustering:** Within each time group, embed photos with CLIP and cluster with DBSCAN to separate visually distinct sub-scenes.

### Why

**Time alone is too coarse.** If you photograph a landscape and then immediately turn and photograph your kids, both sets have timestamps within seconds of each other. Time windowing would put them in the same moment. CLIP catches that they look completely different and splits them.

**Visuals alone are too expensive and miss the obvious.** Running DBSCAN across an entire vacation's worth of photos would be slow and would produce clusters that span hours — your child at breakfast and at dinner might get the same CLIP embedding if they're in the same location.

The two-pass combination is **cheap where cheap is enough** (time windows cost nothing) and **precise where precision matters** (CLIP+DBSCAN only runs within each time group, which is usually small).

**Alternatives considered:**

| Alternative | Trade-off |
|---|---|
| Time windows only | Misses scene changes within a burst |
| CLIP+DBSCAN on all photos | Slow; clusters may span large time gaps |
| GPS-based grouping | GPS is often unavailable indoors; still needs visual refinement |
| Hashing for near-duplicates | Detects exact duplicates, not scene grouping |

### Interview answer

> "I used two passes: time windowing first to separate obvious temporal gaps cheaply, then CLIP embeddings and DBSCAN within each time group to split visually distinct sub-scenes. Time alone misses quick scene changes — if you snap a landscape and then turn to photograph your kids in the same minute, they'd be the same moment. The two-pass design keeps the expensive visual step small by limiting it to each already-narrow time window."

---

## 4. Visual Similarity: CLIP Embeddings

### What we did

Used OpenAI's CLIP model (`ViT-B-32`, loaded via `open_clip_torch`) to produce a 512-dimensional embedding for each photo. Photos that look visually similar will have embeddings that are close in cosine distance.

### Why

**CLIP is a strong general visual feature extractor.** It was trained on 400 million image-text pairs and understands high-level visual concepts — scene type, subject, colour palette, composition. Two photos of the same scene will have similar embeddings even if the camera moved slightly. Two photos of completely different subjects will be far apart.

**Why `open_clip_torch` instead of OpenAI's original CLIP:**

The original `clip` Python package uses a pinned version of PyTorch and conflicts with many environments. `open_clip_torch` is the community re-implementation — better maintained, more model options, compatible with modern PyTorch, and the weights are equivalent.

**Why `ViT-B-32` specifically:**

It is the smallest standard CLIP model: fast to load (~350 MB), fast inference, good enough quality for scene grouping. We don't need the precision of a larger model here — we're separating "beach scene" from "restaurant" not performing fine-grained image search.

**Alternatives considered:**

| Alternative | Trade-off |
|---|---|
| ResNet feature extractor (pre-CLIP) | Weaker semantic understanding; needs fine-tuning for best results |
| Hash-based perceptual hashing (pHash) | Only catches near-exact duplicates, not scene changes |
| Pixel-level difference | Sensitive to exposure and focus changes within the same scene |
| CLIP ViT-L/14 | Better quality but ~3× slower and ~4× more VRAM |

### Interview answer

> "CLIP gives me high-level semantic embeddings — it understands 'beach' vs 'restaurant' not just pixel patterns. I used the `ViT-B-32` variant because it's the smallest standard model: fast enough to embed hundreds of photos in seconds and accurate enough for scene grouping. I used `open_clip_torch` instead of OpenAI's original package because it's better maintained and compatible with modern PyTorch."

---

## 5. Clustering: DBSCAN Over K-Means or Hierarchical

### What we did

Used scikit-learn's `DBSCAN` with `metric='cosine'` to cluster CLIP embeddings within each time group.

### Why

**DBSCAN doesn't require you to specify the number of clusters in advance.** Within a 3-minute time window you might have 1 sub-scene or 5 — you don't know ahead of time. K-Means requires you to set `k` before running.

**DBSCAN naturally handles noise points** — photos that don't belong to any cluster get label `-1`. (We then reassign these to the nearest cluster rather than discarding them — see section 19.)

**Cosine distance is the right metric for normalised embeddings.** CLIP embeddings are L2-normalised, so cosine similarity equals the dot product. Cosine distance measures the angle between embedding vectors, which corresponds to semantic dissimilarity — a small angle means the photos look similar.

**Alternatives considered:**

| Alternative | Why not |
|---|---|
| K-Means | Requires knowing K in advance; uses Euclidean distance (wrong for normalised embeddings) |
| Agglomerative (hierarchical) | Good option but slower O(n²) memory; also needs a cut-off distance |
| HDBSCAN | More robust on variable-density clusters, but heavier dependency |

**The key parameter is `dbscan_eps` (default 0.3).** This is the cosine-distance threshold below which two photos are considered neighbours. If it's too small, every photo becomes its own cluster. If it's too large, distinct scenes merge. 0.3 was empirically found to work well for typical travel photography, but it's configurable in `config.yaml`.

### Interview answer

> "I used DBSCAN because I don't know in advance how many sub-scenes are in a time window — K-Means would require guessing that number. DBSCAN finds clusters of arbitrary shape and marks outliers as noise. I used cosine distance because CLIP embeddings are L2-normalised — cosine similarity directly measures semantic closeness for normalised vectors."

---

## 6. Face Analysis: InsightFace Landmark-Based Scoring

### What we did

Used InsightFace's `buffalo_l` model to detect faces and extract 68-point 3D facial landmarks. From these landmarks we derived three scores:

- **Gaze score:** Head pose (yaw, pitch in degrees) → `max(0, 1 − √(yaw² + pitch²) / 45)`. A face looking straight at the camera scores 1.0; a face turned 45° or more scores 0.
- **Eyes score:** Eye Aspect Ratio (EAR) from eyelid landmarks. EAR < 0.2 means closed eyes (score 0); EAR ≥ 0.4 means fully open (score 1).
- **Smile score:** Ratio of lip corner uplift to mouth width. Corners lifted > 30% of mouth width scores 1.0.

If a photo has multiple faces, all scores are averaged across faces.

### Why

**Landmarks are interpretable and direct.** Instead of training a classifier on labelled "smiling/not smiling" examples, we compute geometry from the landmarks InsightFace already provides. This means:

- No training data needed
- Scores are explainable — "gaze is low because head pose yaw=38°"
- Works across ages and ethnicities without fine-tuning

**Why InsightFace over other options:**

| Option | Trade-off |
|---|---|
| MediaPipe Face Mesh | Good landmarks but less accurate face detection in crowded scenes |
| dlib 68-point predictor | Classic and accurate, but CPU-only and slower |
| DeepFace | Higher-level API (emotion classification), less control over raw landmarks |
| OpenCV Haar Cascades | Fast but poor accuracy, especially for profile faces or small faces |

**Why `buffalo_l` specifically:** InsightFace's highest-accuracy bundled model. It combines RetinaFace for detection with ArcFace for recognition and includes 3D landmark estimation. The "l" (large) variant is more accurate than "s" (small), and on an RTX 3080 the difference in speed is negligible.

**The averaging-across-faces design:** When photographing two children, we want a photo where both are looking and smiling, not just one. Averaging penalises photos where one child is looking away, which is exactly what we want.

### Interview answer

> "I used InsightFace's `buffalo_l` model and derived scores geometrically from its 68-point 3D landmarks rather than training classifiers. Gaze comes from head pose angles, eyes open from the Eye Aspect Ratio (ratio of vertical eyelid distance to horizontal eye width), and smile from lip corner geometry. This makes the scoring interpretable — I can point to exactly which measurement caused a low score. When there are multiple faces, I average across them, so a photo where one child looks away scores lower."

---

## 7. Image Quality: OpenCV Laplacian + FFT + Histogram

### What we did

Three quality measurements using OpenCV:

- **Sharpness (Laplacian variance):** Apply a Laplacian filter (edge detector) to the grayscale image. High variance means sharp edges are present — the image is in focus.
- **Blur (FFT high-frequency energy):** Take the Fast Fourier Transform, discard the low-frequency centre, and measure the mean of the remaining high-frequency content. A blurry image has little high-frequency energy.
- **Exposure (histogram midtone coverage):** Build the pixel brightness histogram. Measure what fraction of pixels fall in the middle range (30–220 out of 255). An overexposed or underexposed image has most pixels piled at the extremes.

In the ranker, sharpness and blur scores are averaged into a single `sharpness` component.

### Why

**These three measurements catch the most common technical photo failures:**

- **Out of focus** → low Laplacian variance, low FFT energy
- **Motion blur** → low Laplacian variance, low FFT energy (similar to defocus)
- **Overexposed** → histogram skewed to 255
- **Underexposed** → histogram skewed to 0

**Why two measures of sharpness (Laplacian + FFT)?** They are sensitive to slightly different kinds of blur. Laplacian is sensitive to local edge contrast; FFT measures global frequency content. Averaging them is more robust than either alone.

**Alternatives considered:**

| Alternative | Trade-off |
|---|---|
| BRISQUE (no-reference quality model) | More sophisticated but heavier; requires calibration data |
| NIQE | Similar story — more robust but more complex |
| SSIM / PSNR | These are reference metrics — they compare two images, require a "ground truth" |
| Neural quality model (MUSIQ, etc.) | GPU inference for quality alone is heavy when we already run CLIP + InsightFace |

### Interview answer

> "I used three OpenCV measurements: Laplacian variance for sharpness (sharp edges produce high variance after edge detection), FFT high-frequency energy for blur (blurry images lack high-frequency content), and histogram midtone coverage for exposure (a well-exposed photo has pixels spread across the middle of the brightness range). These catch the main technical failures — focus, motion blur, over/underexposure — without needing a trained model."

---

## 8. Scoring: Weighted Linear Combination

### What we did

Combined five scores into one composite:

```
composite = gaze×0.30 + smile×0.25 + eyes×0.20 + sharpness×0.15 + exposure×0.10
```

The weights are configurable in `config.yaml`.

### Why

**A weighted sum is the simplest model that lets the user express preferences.** The weights encode a priority: for this use case (kids looking at camera and smiling), gaze and smile matter most. Sharpness and exposure matter less because modern phones rarely produce badly exposed shots of moving subjects.

**All input scores are in [0, 1]**, so the composite is also in [0, 1] and the weights are directly interpretable as percentages of importance.

**Alternatives considered:**

| Alternative | Trade-off |
|---|---|
| Unweighted average | Treats gaze and exposure as equally important — wrong for this use case |
| Multiplication | A single 0 score zeros out the entire composite — too harsh |
| Trained ranking model | Needs labelled training data (which photos are better?) — no data available |
| Geometric mean | Similar issue to multiplication with zero scores |
| Rank aggregation (Borda count) | Robust but loses the magnitude of differences between scores |

**The configurable weights are the key design decision.** Different users have different priorities. A wildlife photographer would weight sharpness and exposure highest. A portrait photographer might weight smile and eyes. By externalising weights to `config.yaml` instead of hardcoding them, the same scoring engine serves different use cases without code changes.

### Interview answer

> "I used a configurable weighted linear combination. All five sub-scores are in [0, 1] so the weights directly express relative importance — currently 30% gaze, 25% smile, 20% eyes open, 15% sharpness, 10% exposure. A trained ranking model would be more accurate but requires labelled data I don't have. A linear sum is interpretable: I can explain why any photo scored the way it did by looking at which sub-scores were low."

---

## 9. Keep Suggestion: Threshold Relative to Top Score

### What we did

Suggest keeping all photos whose composite score is at least `keep_threshold × top_score` (default: 60% of the best score in the moment).

```python
threshold = top_score * keep_threshold
suggested = [p for p in ranked if p.composite >= threshold]
```

### Why

**A relative threshold adapts to the quality of the moment.** If your best photo in a moment scores 0.9, the threshold is 0.54. If your best scores 0.5 (low-quality burst), the threshold is 0.3. This means you always get at least one suggestion, even in a mediocre moment.

**A fixed absolute threshold would be brittle.** If you set it at 0.7 and all photos in a moment score 0.65 (decent but not great), you'd get zero suggestions. That forces the user to manually pick with no AI guidance — defeating the purpose.

**Alternatives considered:**

| Alternative | Trade-off |
|---|---|
| Absolute threshold | Fails when whole moment is average quality |
| Top-N suggestions | Doesn't adapt to how spread out the scores are; you might suggest 3 nearly identical photos |
| Top-N% | Similar to relative threshold; less intuitive |
| Neural "keep this photo" classifier | Requires labelled training data |

### Interview answer

> "I used a threshold relative to the top score in each moment — by default 60% of the best composite score. This adapts to the quality of the burst: if the best photo scores 0.9, the threshold is 0.54; if the best scores 0.5, the threshold is 0.3. An absolute threshold would fail on mediocre moments where all photos are decent but none hit a fixed bar. The relative approach always gives the user at least one suggestion and scales gracefully."

---

## 10. Output: Hard Links Instead of Copies

### What we did

```python
try:
    os.link(src_path, dest_path)   # hard link — zero disk space
except OSError:
    shutil.copy2(src_path, dest_path)  # fallback: cross-filesystem or permission issue
```

Photos are hard-linked into `output/curated/YYYY-MM-DD/` rather than copied.

### Why

**A hard link is a second directory entry pointing to the same inode (the actual file data on disk).** It uses zero additional disk space. If you keep 2,000 photos out of 10,000, you don't duplicate 2,000 files — you add 2,000 directory entries.

This matters because travel photo bursts are large. A single vacation might be 50 GB of HEIC files. A curated selection of 10% would still be 5 GB if copied. With hard links, it's a few kilobytes of directory metadata.

**Why the `OSError` fallback to `shutil.copy2`:**

Hard links have two limitations:
1. They cannot cross filesystem boundaries. If your source photos are on an external drive and your output is on the internal drive, `os.link` will raise `OSError`.
2. Some filesystems (FAT32, some network shares) don't support hard links at all.

`shutil.copy2` preserves timestamps and metadata — it's the closest semantic equivalent to a hard link when a link isn't possible.

**What hard links mean for curation workflow:**

If you later delete a hard link (the curated copy), the original is unaffected — the inode persists until its last directory entry is deleted. Conversely, editing the curated copy via an app that modifies in-place (rare) would modify the original. Apps that write by creating a new file (most modern photo apps) are safe.

### Interview answer

> "I used `os.link` to hard-link curated photos into the output folder instead of copying them. A hard link is a second directory entry pointing to the same file data — zero disk cost. For a 50 GB vacation library, copying even 10% would use 5 GB. Hard links keep the curated folder essentially free. I fall back to `shutil.copy2` if the link fails — which happens when source and destination are on different filesystems."

---

## 11. API: FastAPI Over Flask or Django

### What we did

Used FastAPI with Uvicorn to serve the review API and static UI files.

### Why

**FastAPI gives automatic request validation and response serialisation through Python type annotations.** Define a Pydantic model for the request body and FastAPI validates it, returning a 422 with a detailed error message if the client sends bad data. No validation code to write manually.

**It also generates OpenAPI documentation automatically** — browsable at `/docs` — which makes it easy to test endpoints manually.

**Why not Flask:**

Flask is simpler to set up for small projects, but has no built-in validation. You'd add Marshmallow or WTForms for request validation. FastAPI does this out of the box.

**Why not Django:**

Django's strength is large applications with ORM, admin UI, auth, and templating. It's heavyweight for a project that only needs five JSON endpoints and static file serving.

**Why not aiohttp / Starlette directly:**

FastAPI is built on Starlette and adds the Pydantic layer. Using Starlette directly would require writing the validation manually. Using aiohttp would require async throughout the codebase.

**A note on async:** All our route handlers are synchronous (regular `def`, not `async def`). FastAPI runs them in a thread pool automatically. We don't have async database calls or external HTTP calls, so there's no benefit to full async — and it would add complexity.

### Interview answer

> "I chose FastAPI because it gives automatic request validation via Pydantic type annotations and generates OpenAPI docs for free. Flask would have worked for this size of project but has no built-in validation. Django would have been overkill — we only need five endpoints and static file serving, not an ORM, admin interface, and templating engine. I used synchronous route handlers because all our I/O is SQLite reads — no benefit to async here."

---

## 12. Frontend: Vanilla JS Over React or Vue

### What we did

A single `index.html` and `app.js` file — no build step, no npm, no bundler. Plain `fetch` calls to the FastAPI endpoints.

### Why

**The UI has a single screen with a simple interaction model:** display a grid of photos, toggle keep/reject, submit. There is no routing, no shared state across components, no real-time updates, no complex forms.

React and Vue are powerful tools for complex UIs, but they introduce:
- A build step (webpack, Vite, etc.)
- Node.js as a development dependency
- A component model with its own learning curve
- Hundreds of megabytes of `node_modules`

For a personal local tool with one developer and one screen, vanilla JS is the right call. The entire frontend is ~100 lines and loads instantly.

**The trade-off:** As the UI grows — if you added filtering by date/GPS, photo comparison view, batch tagging — vanilla JS becomes harder to maintain. The lack of component abstraction means code duplication, and the lack of a virtual DOM means more manual DOM updates. At that point, a framework would be justified.

**The principle: match the tool to the problem size.** YAGNI (You Aren't Gonna Need It) applies here. We build the UI that solves today's problem efficiently.

### Interview answer

> "I used vanilla JS because the UI is a single screen: show photos, toggle keep/reject, submit. React and Vue would add a build pipeline, Node.js as a dependency, and hundreds of MB of node_modules for a 100-line frontend. The trade-off is maintainability — if the UI grew to multiple screens with complex state, I'd migrate to a framework. But for a personal local tool with one screen, vanilla JS is faster to build, easier to debug, and has zero dependencies."

---

## 13. HEIC Support: pillow-heif Registration at Import Time

### What we did

At the top of `ingestor.py`:

```python
import pillow_heif
pillow_heif.register_heif_opener()
```

### Why

**iPhone photos are saved in HEIC format** (High Efficiency Image Container). Pillow does not support HEIC natively. `pillow_heif` is a plugin that adds HEIC/HEIF decoding support to Pillow by wrapping the `libheif` C library.

`register_heif_opener()` patches Pillow's internal codec registry so that `Image.open("photo.heic")` works transparently — no code changes needed anywhere else.

**Why register at import rather than calling it explicitly per file:**

The registry is process-wide. Registering once at module import ensures HEIC works everywhere Pillow is used — in the ingestor, scorer (which opens images for OpenCV), and CLIP embedding (which uses `Image.open`). Calling it per-file would be repetitive and easy to forget.

**The trade-off:** Side effects at import time are generally discouraged in Python (they make mocking harder in tests). In this case, the alternative — threading the capability through function arguments — would add unnecessary complexity everywhere. The test suite mocks at the function level and is unaffected.

### Interview answer

> "iPhones save photos as HEIC, which Pillow can't open natively. `pillow_heif` provides HEIC/HEIF decoding by wrapping `libheif`. I call `register_heif_opener()` at module import time so it patches Pillow's codec registry once, globally — from then on, `Image.open('photo.heic')` just works everywhere. Registering per-file would be repetitive and easy to miss."

---

## 14. EXIF Reading: Public API First, Private Fallback

### What we did

```python
def _read_exif(path: str) -> dict:
    try:
        img = Image.open(path)
        if hasattr(img, 'getexif'):
            return dict(img.getexif()) or {}
        return img._getexif() or {}
    except Exception:
        return {}
```

### Why

**`getexif()` (no underscore) is the public API** introduced in Pillow 6.0. It returns a clean `Exif` object. **`_getexif()` (underscore prefix) is the old private method** that was the only way to get EXIF data in older Pillow versions, but has been deprecated.

By checking `hasattr(img, 'getexif')` first:
- Modern Pillow (6.0+) uses the stable public API
- Older environments still work via the fallback
- HEIC images opened via `pillow_heif` may not implement `_getexif()` — the public API handles this gracefully

**The GPS ref bytes issue:** Some cameras and phones write EXIF GPS direction references as bytes (`b'N'`) rather than strings (`'N'`). The `_dms_to_decimal` function handles this with `str(ref).upper().strip()` — converting bytes to string (`str(b'N') == "b'N'"`) would fail, but `str(ref)` on a plain string is a no-op. Actually, for bytes we wrap with explicit conversion: `str(ref, 'utf-8')` won't work for all cases, so using `str(ref).upper().strip()` and stripping `b'` is slightly inelegant but robust.

**The blanket `except Exception: return {}`:** EXIF data is optional. If it can't be read — corrupted file, unsupported format, partial download — the photo still gets ingested with a timestamp from the file modification time. Crashing the entire ingestion run over one photo's metadata would be worse than silently degrading.

### Interview answer

> "I try the public `getexif()` method first (added in Pillow 6) and fall back to the private `_getexif()` for older environments. HEIC files opened via `pillow_heif` may not implement `_getexif()`, so preferring the public API is important. If EXIF reading fails entirely, I catch the exception and return an empty dict — I'd rather ingest the photo with a filesystem timestamp than crash the whole pipeline over one photo's missing metadata."

---

## 15. Idempotent Inserts: INSERT OR IGNORE + rowcount Check

### What we did

```python
def insert_photo(conn, path, ...):
    cur = conn.execute(
        "INSERT OR IGNORE INTO photos (path, ...) VALUES (?,?,?,?,?)",
        (path, ...),
    )
    conn.commit()
    if cur.rowcount:          # 1 if inserted, 0 if ignored
        return cur.lastrowid
    return conn.execute("SELECT id FROM photos WHERE path=?", (path,)).fetchone()[0]
```

### Why

**Re-running the pipeline should be safe.** If you add 10 new photos to your folder and re-run `pipeline.py`, the 10 new photos should be inserted and the existing 10,000 should be silently skipped. Without `OR IGNORE`, the second run would raise a UNIQUE constraint violation.

**Why `cur.rowcount` and not `cur.lastrowid`:**

This is a subtle SQLite quirk that causes real bugs. When `INSERT OR IGNORE` silently ignores a duplicate, `cur.lastrowid` does **not** return 0 — it returns the `rowid` from the **previous** successful insert (the last insert in the session). So checking `if cur.lastrowid:` would return the wrong ID for a skipped insert. `cur.rowcount` is reliably 0 when the insert was ignored and 1 when it succeeded.

**The SELECT fallback** fetches the existing row's ID when the insert was ignored, so the caller always gets back a valid `photo_id` regardless of whether this was a new insert or a duplicate.

### Interview answer

> "The `path` column has a UNIQUE constraint, so inserting the same photo twice would normally raise an exception. `INSERT OR IGNORE` silently skips duplicates. The subtle part is checking `cur.rowcount` rather than `cur.lastrowid` to detect whether the insert happened — when SQLite ignores a duplicate, `lastrowid` returns the ID of the previous successful insert, not zero. I then do a SELECT to retrieve the existing row's ID."

---

## 16. Thread Safety: check_same_thread=False

### What we did

```python
conn = sqlite3.connect(db_path, check_same_thread=False)
# FastAPI worker threads share this connection
```

### Why

SQLite's Python driver has a built-in guard: by default it raises `ProgrammingError` if you try to use a connection from a different thread than the one that created it. This is a conservative safety measure.

FastAPI runs synchronous route handlers in a thread pool (Starlette's internal thread pool). The `conn` object is created in the startup thread, then used by worker threads — triggering SQLite's guard.

`check_same_thread=False` disables this check. It is safe in this specific context because:
1. This is a single-user local application — no concurrent requests from different users
2. The synchronous route handlers are short transactions — read or write a few rows, commit, return
3. SQLite serialises write transactions at the database level

**The real risk of `check_same_thread=False`:** If two threads simultaneously use the same connection and one is in the middle of a multi-step transaction, you could get interleaved operations. The safe production solution is one connection per request (FastAPI dependency injection pattern). For a personal tool with one user, the simpler approach is acceptable and documented.

### Interview answer

> "SQLite's Python driver raises an error if you use a connection from a thread other than the one that created it. FastAPI runs synchronous handlers in a thread pool, so the startup thread's connection triggers this guard. `check_same_thread=False` disables the check. It's safe here because this is a single-user local app with short transactions — there's no real concurrency. The production solution would be per-request connections via FastAPI's dependency injection, but that's over-engineering for a personal tool."

---

## 17. Configuration: YAML Dataclass With Strict Validation

### What we did

`config.yaml` is loaded into a `Config` dataclass with explicit validation:

```python
unknown = set(data) - _CONFIG_FIELDS
if unknown:
    raise ValueError(f"{path}: unknown config keys: {sorted(unknown)}")
```

### Why

**Unknown keys in config are always bugs.** If you mistype `time_windw_minutes: 5` in `config.yaml`, Python would silently ignore it and run with the default of 3 minutes. You'd wonder why your grouping changed nothing. Strict validation catches this immediately with a helpful error message: `"config.yaml: unknown config keys: ['time_windw_minutes']"`.

**Why a dataclass instead of a dict:**

Dataclasses give:
- IDE autocompletion for `cfg.time_window_minutes` (vs `cfg['time_window_minutes']`)
- A `__repr__` for debugging
- Type hints for documentation

**Why YAML instead of JSON or TOML:**

YAML supports comments (`# explanation`) — important for a config file that non-technical users might edit. JSON doesn't. TOML is also comment-friendly and arguably better for config files, but PyYAML is the established Python library with no extra installation needed (pip). TOML requires `tomllib` (stdlib in Python 3.11+) or `tomli` (backport).

### Interview answer

> "I used a YAML-backed dataclass with strict validation for unknown keys. YAML allows comments, which matters for a config file meant to be hand-edited. I explicitly validate that no unknown keys are present — a typo like `time_windw_minutes` would otherwise silently use the default, which is a confusing bug to debug. Dataclass fields give IDE autocomplete and type hints instead of string-keyed dict lookups."

---

## 18. GPS Coordinates: DMS to Decimal Conversion

### What we did

GPS coordinates in EXIF are stored in Degrees, Minutes, Seconds (DMS) format with a reference direction (N/S/E/W). We convert to decimal degrees:

```
decimal = degrees + minutes/60 + seconds/3600
if reference in ('S', 'W'):
    decimal = -decimal
```

### Why

**Decimal degrees are the standard format** for GPS libraries, maps, and databases. They store as a single float. DMS is a human-readable legacy format that requires parsing.

**The 'S'/'W' negation:** Latitude is positive in the northern hemisphere (N) and negative in the southern (S). Longitude is positive east of the prime meridian (E) and negative west (W). This is the WGS84 coordinate system convention.

**The `str(ref).upper().strip()` defensive conversion:** EXIF GPS reference values can be stored as strings (`'N'`), bytes (`b'N'`), or bytes with trailing nulls (`b'N\x00'`). Wrapping in `str().upper().strip()` handles all of these without branching on type.

### Interview answer

> "EXIF stores GPS as Degrees/Minutes/Seconds with a direction reference (N/S/E/W). I convert to decimal degrees because that's what every mapping library and database expects — it's just a float. South and West references negate the value per the WGS84 convention. I wrap the reference in `str().upper().strip()` because some cameras write it as bytes and some as strings."

---

## 19. Noise Handling in DBSCAN: Reassign to Nearest Cluster

### What we did

DBSCAN labels outlier points as `-1` (noise). We reassign each noise point to the cluster of its nearest non-noise neighbour:

```python
noise_mask = labels == -1
if noise_mask.any():
    non_noise = np.where(~noise_mask)[0]
    for i in np.where(noise_mask)[0]:
        sims = embeddings[i] @ embeddings[non_noise].T
        labels[i] = labels[non_noise[np.argmax(sims)]]
```

### Why

**Discarding noise points would lose photos.** Every photo in a moment needs to be reviewed. If a photo's embedding is slightly far from its neighbours (unusual composition, unusual lighting), DBSCAN marks it as noise. We can't throw it away — we need to put it in _some_ moment.

**Reassigning to the nearest cluster is the most sensible heuristic.** The noise photo is probably related to the closest cluster even if not close enough to formally belong to it. It's better than creating a singleton moment for every noise point.

**Why dot product for similarity when embeddings are normalised:**

For L2-normalised vectors, cosine similarity = dot product. So `embeddings[i] @ embeddings[non_noise].T` computes all pairwise cosine similarities in one matrix multiplication — fast and correct.

**Edge case:** If all photos in a time group are noise (DBSCAN produced no clusters at all), we assign everyone to cluster 0. This happens when `dbscan_eps` is very small and all photos are far apart — in that case, each photo probably is its own moment, but we keep them together rather than creating N singleton moments.

### Interview answer

> "DBSCAN marks outlier points as noise (label -1). I can't discard those photos — every photo needs to be in some moment. So I reassign each noise point to the cluster of its most similar non-noise neighbour using dot product similarity (equivalent to cosine similarity for L2-normalised CLIP embeddings). It's a simple nearest-neighbour reassignment that keeps all photos while respecting the clustering structure."

---

## Summary Table

| Decision | Choice | Key trade-off |
|---|---|---|
| Architecture | Batch pipeline + separate server | Simplicity over real-time streaming |
| Storage | SQLite | Zero-config local tool; no concurrency |
| Grouping | Time windows → CLIP + DBSCAN | Cheap coarse pass + accurate fine pass |
| Embeddings | CLIP ViT-B-32 | Fast + semantic; larger models exist |
| Clustering | DBSCAN cosine | Unknown K; adapts to cluster count |
| Face scoring | InsightFace landmarks | Geometric, interpretable, no labelled data |
| Quality | Laplacian + FFT + histogram | Covers blur, focus, exposure without a model |
| Scoring model | Weighted linear sum | Configurable; transparent |
| Keep suggestion | Relative threshold | Adapts to moment quality |
| Output | Hard links + copy fallback | Zero disk cost; works cross-filesystem |
| API | FastAPI | Auto-validation; lightweight |
| Frontend | Vanilla JS | Single screen; no build step |
| HEIC | pillow-heif at import | Transparent to all Pillow callers |
| EXIF | Public API + fallback | Handles Pillow version differences |
| Idempotency | INSERT OR IGNORE + rowcount | Safe re-runs; avoids lastrowid bug |
| Threading | check_same_thread=False | Simple for single-user; documented |
| Config | YAML dataclass + strict validation | Catches typos; IDE-friendly |
