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
