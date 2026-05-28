import pytest
from photosorter.config import load_config, Config, Weights


def test_load_config_defaults(tmp_path):
    cfg_text = """
time_window_minutes: 5
dbscan_eps: 0.4
dbscan_min_samples: 2
keep_threshold: 0.7
weights:
  gaze: 0.40
  smile: 0.20
  eyes: 0.20
  sharpness: 0.10
  exposure: 0.10
clip_model: "ViT-B-32"
clip_pretrained: "openai"
insightface_model: "buffalo_l"
det_size: [640, 640]
output_dir: "out"
db_path: "test.db"
port: 9090
"""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(cfg_text)
    cfg = load_config(str(cfg_file))
    assert isinstance(cfg, Config)
    assert cfg.time_window_minutes == 5
    assert cfg.weights.gaze == pytest.approx(0.40)
    assert cfg.det_size == (640, 640)
    assert cfg.port == 9090
