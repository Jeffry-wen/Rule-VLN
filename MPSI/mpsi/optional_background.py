"""Optional InSPyReNet rule-background segmentation."""

from __future__ import annotations

import sys
from pathlib import Path


def segment_rule(rule_path, config):
    if not config.inspyrenet_config or not config.inspyrenet_checkpoint:
        raise ValueError(
            "InSPyReNet config and checkpoint are required when background segmentation is enabled"
        )
    third_party = Path(__file__).resolve().parents[1] / "third_party/inspyrenet"
    sys.path.insert(0, str(third_party))
    from PIL import Image
    from run.Inference import segment_image

    mask = segment_image(
        config_file=config.inspyrenet_config,
        source=str(rule_path),
        type="mask",
        gpu=True,
        checkpoint=config.inspyrenet_checkpoint,
    )
    return Image.fromarray(mask).convert("L")
