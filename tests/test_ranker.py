import pytest
from photosorter.config import Weights
from photosorter.ranker import rank_moment, suggested_keep_ids, PhotoScore


WEIGHTS = Weights(gaze=0.30, smile=0.25, eyes=0.20, sharpness=0.15, exposure=0.10)


def _make_photo_scores(scores_list):
    keys = ['photo_id', 'gaze_score', 'smile_score', 'eyes_score',
            'sharpness_raw', 'sharpness', 'blur_score', 'exposure']
    return [dict(zip(keys, row)) for row in scores_list]


def test_rank_moment_best_first():
    photos = _make_photo_scores([
        (1, 0.9, 0.8, 1.0, 100, 1.0, 0.8, 0.9),
        (2, 0.1, 0.1, 0.0, 10,  0.1, 0.2, 0.5),
    ])
    ranked = rank_moment(photos, WEIGHTS, keep_threshold=0.6)
    assert ranked[0].photo_id == 1
    assert ranked[0].rank == 1
    assert ranked[1].rank == 2


def test_rank_moment_composite_in_range():
    photos = _make_photo_scores([
        (1, 1.0, 1.0, 1.0, 100, 1.0, 1.0, 1.0),
    ])
    ranked = rank_moment(photos, WEIGHTS, keep_threshold=0.6)
    assert 0.0 <= ranked[0].composite <= 1.0


def test_suggested_keep_ids_filters_below_threshold():
    ranked = [
        PhotoScore(photo_id=1, gaze=0.9, smile=0.9, eyes=1.0,
                   sharpness=0.9, exposure=0.9, composite=0.92, rank=1),
        PhotoScore(photo_id=2, gaze=0.5, smile=0.5, eyes=1.0,
                   sharpness=0.5, exposure=0.5, composite=0.56, rank=2),
        PhotoScore(photo_id=3, gaze=0.1, smile=0.1, eyes=0.0,
                   sharpness=0.1, exposure=0.1, composite=0.10, rank=3),
    ]
    # threshold=0.6: photos above 0.6*0.92=0.552 are suggested
    keep = suggested_keep_ids(ranked, keep_threshold=0.6)
    assert 1 in keep
    assert 2 in keep
    assert 3 not in keep


def test_suggested_keep_single_photo():
    ranked = [
        PhotoScore(photo_id=1, gaze=0.5, smile=0.5, eyes=0.5,
                   sharpness=0.5, exposure=0.5, composite=0.5, rank=1),
    ]
    keep = suggested_keep_ids(ranked, keep_threshold=0.6)
    assert keep == [1]
