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
    import os
    os.makedirs(str(tmp_path / "photos"), exist_ok=True)
    open(str(tmp_path / "photos" / "a.jpg"), 'w').close()

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
