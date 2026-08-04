#!/usr/bin/env python3
"""Validate graph, task manifests, and LMDB key coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lmdb


def lmdb_keys(path):
    env = lmdb.open(
        str(path), readonly=True, lock=False, readahead=False, meminit=False
    )
    try:
        with env.begin() as txn:
            return {key.decode("utf-8") for key in txn.cursor().iternext(values=False)}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--level", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--check-raw", action="store_true")
    args = parser.parse_args()
    graph = args.data_root / "touchdown/graph"
    node_count = sum(1 for _ in (graph / "nodes.txt").open())
    link_count = sum(1 for _ in (graph / "links.txt").open())
    if node_count < 1 or link_count < 1:
        raise ValueError("Touchdown graph is empty")
    manifest_path = (
        args.data_root
        / f"touchdown/insert_touchdown/insert_task{args.level}_dataset.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature_keys = lmdb_keys(
        args.data_root
        / "features/flame_clip_touchdown_insert_120_crop_lmdb"
    )
    missing = [
        node
        for node in manifest
        if not any(key == node or key.startswith(node + "_") for key in feature_keys)
    ]
    if missing:
        raise ValueError(
            f"{len(missing)} task nodes have no inserted feature, for example {missing[0]}"
        )
    if args.check_raw:
        raw_keys = lmdb_keys(args.data_root / "images/touchdown_insert_crop_120_lmdb")
        raw_missing = [
            node
            for node in manifest
            if not any(key == node or key.startswith(node + "_") for key in raw_keys)
        ]
        if raw_missing:
            raise ValueError(
                f"{len(raw_missing)} task nodes have no inserted raw view"
            )
    print(
        f"Validated {node_count} nodes, {link_count} links, "
        f"{len(manifest)} task nodes, and inserted feature coverage."
    )


if __name__ == "__main__":
    main()
