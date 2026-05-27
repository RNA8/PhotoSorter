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
