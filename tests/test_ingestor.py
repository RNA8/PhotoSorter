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
