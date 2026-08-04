"""MPSI configuration models."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass
class MPSIConfig:
    panorama_dir: str
    nodes_file: str
    links_file: str
    route_data: str
    score_file: str
    rule_image_dir: str
    rule_labels: str
    shape_dir: str
    segformer_model: str
    flux_transformer: str
    flux_fill_model: str
    flux_redux_model: str
    insert_anything_lora: str
    clip_model: str
    seed: int = 777
    diffusion_seed: int = 666
    group_count: int = 4
    quality_probability: float = 0.3
    road_label: int = 1
    shape_label: int = 7
    use_background_segmentation: bool = False
    inspyrenet_config: str | None = None
    inspyrenet_checkpoint: str | None = None


def load_config(path):
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    base = config_path.parent
    known = {field.name for field in fields(MPSIConfig)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown MPSI configuration keys: {sorted(unknown)}")
    for key, value in list(values.items()):
        if not isinstance(value, str):
            continue
        if key.endswith(("_dir", "_file", "_labels", "_data", "_lora", "_config", "_checkpoint")):
            candidate = Path(value).expanduser()
            if not candidate.is_absolute() and not value.startswith(("hf://", "http://", "https://")):
                values[key] = str((base / candidate).resolve())
    return MPSIConfig(**values)
