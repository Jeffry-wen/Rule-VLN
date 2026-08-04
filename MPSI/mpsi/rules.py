"""Select CityNav-Rules examples compatible with a node action space."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from PIL import Image


class RuleCatalog:
    def __init__(self, image_root, labels_path):
        self.image_root = Path(image_root)
        with Path(labels_path).open("r", encoding="utf-8") as handle:
            self.entries = json.load(handle)
        self.by_label = defaultdict(list)
        self.actions = defaultdict(set)
        self.by_file = {}
        for entry in self.entries:
            self.by_label[entry["label_id"]].append(entry)
            self.actions[entry["label_id"]].update(entry["action_node"])
            self.by_file[entry["file"].replace("\\", "/")] = entry

    def select(self, allowed_actions, rng):
        allowed = set(allowed_actions)
        labels = sorted(
            label
            for label, actions in self.actions.items()
            if set(actions).issubset(allowed)
        )
        if not labels:
            raise LookupError(
                f"No CityNav-Rules label supports actions {sorted(allowed)}"
            )
        label = rng.choice(labels)
        candidates = []
        for entry in self.by_label[label]:
            path = self.image_root / entry["file"]
            if not path.exists() and path.with_suffix(".jpg").exists():
                path = path.with_suffix(".jpg")
            if not path.exists():
                continue
            try:
                with Image.open(path) as image:
                    resolution = image.width * image.height
            except OSError:
                continue
            candidates.append((resolution, entry["file"], path, entry))
        if not candidates:
            raise FileNotFoundError(f"No readable images found for rule label {label}")
        _, _, path, entry = max(candidates)
        return path, entry
