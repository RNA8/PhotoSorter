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


def scan_folder(folder: str) -> list:
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
        if hasattr(img, 'getexif'):
            return dict(img.getexif()) or {}
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


def _extract_gps(exif_data: dict) -> tuple:
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
    if str(ref).upper().strip() in ('S', 'W'):
        decimal = -decimal
    return decimal
