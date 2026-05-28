import pytest
import numpy as np
from PIL import Image


@pytest.fixture
def tmp_photo_dir(tmp_path):
    for i in range(3):
        arr = np.full((100, 100, 3), i * 80, dtype=np.uint8)
        Image.fromarray(arr).save(str(tmp_path / f"photo_{i}.jpg"))
    return tmp_path


@pytest.fixture
def tmp_db(tmp_path):
    from photosorter.db import init_db
    return init_db(str(tmp_path / "test.db"))
