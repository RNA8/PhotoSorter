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
