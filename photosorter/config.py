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


_CONFIG_FIELDS = {f.name for f in Config.__dataclass_fields__.values()}
_WEIGHT_FIELDS = {f.name for f in Weights.__dataclass_fields__.values()}


def load_config(path: str = "config.yaml") -> Config:
    with open(path) as f:
        data = yaml.safe_load(f) or {}

    weights_data = data.pop("weights", {})
    unknown_weights = set(weights_data) - _WEIGHT_FIELDS
    if unknown_weights:
        raise ValueError(f"{path}: unknown weights keys: {sorted(unknown_weights)}")
    data["weights"] = Weights(**weights_data)

    if "det_size" in data:
        v = data["det_size"]
        if not (isinstance(v, (list, tuple)) and len(v) == 2):
            raise ValueError(f"{path}: det_size must be a list of two ints, got {v!r}")
        data["det_size"] = tuple(v)

    unknown = set(data) - _CONFIG_FIELDS
    if unknown:
        raise ValueError(f"{path}: unknown config keys: {sorted(unknown)}")

    return Config(**data)
