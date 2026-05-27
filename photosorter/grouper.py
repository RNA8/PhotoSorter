import numpy as np
from sklearn.cluster import DBSCAN


def time_window_groups(photos: list, window_minutes: int) -> list:
    if not photos:
        return []
    window_seconds = window_minutes * 60
    groups = [[photos[0]]]
    for photo in photos[1:]:
        if photo['timestamp'] - groups[-1][-1]['timestamp'] > window_seconds:
            groups.append([])
        groups[-1].append(photo)
    return groups
