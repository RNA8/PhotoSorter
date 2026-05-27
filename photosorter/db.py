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
    conn = sqlite3.connect(db_path, check_same_thread=False)  # FastAPI worker threads share this connection
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
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
    if cur.rowcount:
        return cur.lastrowid
    return conn.execute("SELECT id FROM photos WHERE path=?", (path,)).fetchone()[0]


def insert_moment(conn, start_time, end_time, photo_count) -> int:
    cur = conn.execute(
        "INSERT INTO moments (start_time, end_time, photo_count) VALUES (?,?,?)",
        (start_time, end_time, photo_count),
    )
    conn.commit()
    return cur.lastrowid


def insert_moment_photos(conn, moment_id, photo_ranks: list) -> None:
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


def submit_moment(conn, moment_id, keep_ids: list, output_paths: dict) -> None:
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


def undo_last_moment(conn):
    row = conn.execute(
        "SELECT id FROM moments WHERE reviewed_at IS NOT NULL ORDER BY reviewed_at DESC LIMIT 1"
    ).fetchone()
    if not row:
        return None
    moment_id = row[0]
    photo_ids = [r[0] for r in conn.execute(
        "SELECT photo_id FROM moment_photos WHERE moment_id=?", (moment_id,)
    ).fetchall()]
    if photo_ids:
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
