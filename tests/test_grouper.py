import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from photosorter.grouper import time_window_groups, cluster_by_visual_similarity, group_photos


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


def test_cluster_single_photo():
    emb = np.array([[1.0, 0.0]])
    labels = cluster_by_visual_similarity(emb, eps=0.3, min_samples=1)
    assert labels.tolist() == [0]


def test_cluster_two_similar_photos():
    emb = np.array([[1.0, 0.0], [0.99, 0.14]])  # very close in cosine space
    labels = cluster_by_visual_similarity(emb, eps=0.3, min_samples=1)
    assert labels[0] == labels[1]


def test_cluster_two_dissimilar_photos():
    emb = np.array([[1.0, 0.0], [0.0, 1.0]])  # orthogonal
    labels = cluster_by_visual_similarity(emb, eps=0.3, min_samples=1)
    assert labels[0] != labels[1]


def test_noise_points_assigned_to_nearest_cluster():
    # 3 points: two close together, one distant
    # With min_samples=2, the lone point becomes noise and must be assigned
    emb = np.array([[1.0, 0.0], [0.99, 0.14], [0.0, 1.0]])
    labels = cluster_by_visual_similarity(emb, eps=0.3, min_samples=2)
    assert labels[2] == labels[0]


def test_group_photos_splits_visually_distinct():
    photos = _photos([0, 30, 60])  # all in same time window
    embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.99, 0.14, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)

    mock_model = MagicMock()
    mock_preprocess = MagicMock(side_effect=lambda img: img)

    with patch('photosorter.grouper.compute_clip_embeddings', return_value=embeddings):
        moments = group_photos(
            photos, window_minutes=3, eps=0.3, min_samples=1,
            model=mock_model, preprocess=mock_preprocess, device='cpu'
        )
    assert len(moments) == 2
