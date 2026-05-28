import numpy as np
import pytest
from PIL import Image
from photosorter.scorer import score_image_quality, normalize_sharpness


@pytest.fixture
def sharp_jpg(tmp_path):
    arr = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    path = str(tmp_path / "sharp.jpg")
    Image.fromarray(arr).save(path)
    return path


@pytest.fixture
def blurry_jpg(tmp_path):
    arr = np.full((200, 200, 3), 128, dtype=np.uint8)
    path = str(tmp_path / "blurry.jpg")
    Image.fromarray(arr).save(path)
    return path


def test_score_image_quality_returns_expected_keys(sharp_jpg):
    result = score_image_quality(sharp_jpg)
    assert 'sharpness_raw' in result
    assert 'blur_score' in result
    assert 'exposure' in result


def test_sharp_image_higher_sharpness_raw_than_blurry(sharp_jpg, blurry_jpg):
    sharp_result = score_image_quality(sharp_jpg)
    blurry_result = score_image_quality(blurry_jpg)
    assert sharp_result['sharpness_raw'] > blurry_result['sharpness_raw']


def test_exposure_score_in_range(sharp_jpg):
    result = score_image_quality(sharp_jpg)
    assert 0.0 <= result['exposure'] <= 1.0


def test_normalize_sharpness_max_is_one():
    scores = [
        {'sharpness_raw': 100.0, 'blur_score': 0.5, 'exposure': 0.9},
        {'sharpness_raw': 50.0, 'blur_score': 0.4, 'exposure': 0.8},
    ]
    normalized = normalize_sharpness(scores)
    assert normalized[0]['sharpness'] == pytest.approx(1.0)
    assert normalized[1]['sharpness'] == pytest.approx(0.5, abs=0.01)


from unittest.mock import MagicMock, patch
from photosorter.scorer import score_faces, _eye_aspect_ratio_score, _smile_score


def _make_landmark_68(eyes_open=True, smiling=True):
    lm = np.zeros((68, 3), dtype=np.float32)
    # Left eye indices 36-41: open = tall ellipse (EAR > 0.2), closed = flat
    eye_h = 0.25 if eyes_open else 0.01
    lm[36] = [0.0, 0.0, 0]; lm[39] = [0.6, 0.0, 0]   # horizontal extent
    lm[37] = [0.2, eye_h, 0]; lm[38] = [0.4, eye_h, 0]  # upper lid
    lm[40] = [0.4, -eye_h, 0]; lm[41] = [0.2, -eye_h, 0] # lower lid
    # Right eye indices 42-47: offset by 1.0 in x
    for j in range(6):
        lm[42 + j] = lm[36 + j] + np.array([1.0, 0, 0])
    # Mouth: 48=left corner, 54=right corner, 51=upper lip center
    lm[48] = [0.0, 0.5, 0]; lm[54] = [1.0, 0.5, 0]
    # Smile: upper lip above corner avg → positive uplift
    lm[51] = [0.5, 0.7, 0] if smiling else [0.5, 0.3, 0]
    lm[57] = [0.5, 0.2, 0]  # lower lip center
    return lm


def test_eyes_open_score_open():
    lm = _make_landmark_68(eyes_open=True)
    assert _eye_aspect_ratio_score(lm) == 1.0


def test_eyes_open_score_closed():
    lm = _make_landmark_68(eyes_open=False)
    assert _eye_aspect_ratio_score(lm) == 0.0


def test_smile_score_smiling_higher_than_neutral():
    lm_smile = _make_landmark_68(smiling=True)
    lm_neutral = _make_landmark_68(smiling=False)
    assert _smile_score(lm_smile) > _smile_score(lm_neutral)


def test_score_faces_no_faces_returns_midpoint(sharp_jpg):
    mock_analyzer = MagicMock()
    mock_analyzer.get.return_value = []
    result = score_faces(sharp_jpg, mock_analyzer)
    assert result == {'gaze_score': 0.5, 'smile_score': 0.5, 'eyes_score': 0.5}


def test_score_faces_averages_across_faces(sharp_jpg):
    face1 = MagicMock()
    face1.pose = np.array([0.0, 0.0, 0.0])   # looking at camera
    face1.landmark_3d_68 = _make_landmark_68(eyes_open=True, smiling=True)

    face2 = MagicMock()
    face2.pose = np.array([45.0, 0.0, 0.0])  # looking away
    face2.landmark_3d_68 = _make_landmark_68(eyes_open=True, smiling=False)

    mock_analyzer = MagicMock()
    mock_analyzer.get.return_value = [face1, face2]

    result = score_faces(sharp_jpg, mock_analyzer)
    # face1 gaze=1.0, face2 gaze=0.0 → average ≈ 0.5
    assert result['gaze_score'] == pytest.approx(0.5, abs=0.05)
    assert 0.0 <= result['smile_score'] <= 1.0
    assert result['eyes_score'] == 1.0  # both faces have eyes open
