#!/usr/bin/env python3
"""Download Rule-VLN assets and normalize the historical remote folder name."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


MAPPINGS = {
    "Rule-VLN-Envirement/flame_clip_touchdown_120_crop_lmdb": (
        "data/features/flame_clip_touchdown_120_crop_lmdb"
    ),
    "Rule-VLN-Envirement/flame_clip_touchdown_insert_120_crop_lmdb": (
        "data/features/flame_clip_touchdown_insert_120_crop_lmdb"
    ),
    "Rule-VLN-Envirement/insert_touchdown": "data/touchdown/insert_touchdown",
    "Finetune-ckpt/Flame_checkpoint-2500": "checkpoints/Flame_checkpoint-2500",
    "Rule-VLN-Environment/touchdown/data": "data/touchdown/data",
    "Rule-VLN-Environment/touchdown/ft_data": "data/touchdown/ft_data",
    "City_Navigation_Rules_Dataset": "data/citynav_rules",
}


def move_tree(source, destination):
    if not source.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        shutil.rmtree(destination)
    os.replace(source, destination)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="jeffry77/Rule-VLN")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output-root", type=Path, default=Path("."))
    parser.add_argument("--token", default=None)
    args = parser.parse_args()

    staging = args.output_root / ".asset_download"
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        token=args.token,
        local_dir=staging,
        allow_patterns=[f"{source}/**" for source in MAPPINGS],
    )
    moved = []
    for source, destination in MAPPINGS.items():
        if move_tree(staging / source, args.output_root / destination):
            moved.append(destination)
    shutil.rmtree(staging, ignore_errors=True)
    if not moved:
        raise RuntimeError("No recognized Rule-VLN assets were downloaded.")
    print("Downloaded assets:")
    for path in moved:
        print(f"  {path}")


if __name__ == "__main__":
    main()
