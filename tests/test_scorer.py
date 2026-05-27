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
