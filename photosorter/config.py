from dataclasses import dataclass, field
import yaml


@dataclass
class Weights:
    gaze: float = 0.30
    smile: float = 0.25
    eyes: float = 0.20
    sharpness: float = 0.15
    exposure: float = 0.10


@dataclass
class Config:
    time_window_minutes: int = 3
    dbscan_eps: float = 0.3
    dbscan_min_samples: int = 1
    keep_threshold: float = 0.6
    weights: Weights = field(default_factory=Weights)
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "openai"
    insightface_model: str = "buffalo_l"
    det_size: tuple = (640, 640)
    output_dir: str = "output/curated"
    db_path: str = "photosorter.db"
    port: int = 8080


def load_config(path: str = "config.yaml") -> Config:
    with open(path) as f:
        data = yaml.safe_load(f)
    weights_data = data.pop("weights", {})
    data["weights"] = Weights(**weights_data)
    if "det_size" in data:
        data["det_size"] = tuple(data["det_size"])
    return Config(**data)
