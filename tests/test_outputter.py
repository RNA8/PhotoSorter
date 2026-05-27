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
