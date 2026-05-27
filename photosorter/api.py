import os
import sqlite3
from typing import List

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
    keep_ids: List[int]


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
                    and top_score > 0.0
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
