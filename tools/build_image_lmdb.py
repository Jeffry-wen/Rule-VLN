#!/usr/bin/env python3
"""Pack projected JPEG/PNG views into an LMDB without changing their pixels."""

from __future__ import annotations

import argparse
from pathlib import Path

import lmdb
from tqdm import tqdm


def build_image_lmdb(image_dir, output_dir, map_size=1_000_000_000_000):
    image_dir, output_dir = Path(image_dir), Path(output_dir)
    files = sorted(
        path
        for path in image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not files:
        raise FileNotFoundError(f"No images found in {image_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(output_dir), map_size=int(map_size), subdir=True)
    try:
        with env.begin(write=True) as txn:
            for path in tqdm(files, desc="Packing image LMDB"):
                txn.put(path.stem.encode("utf-8"), path.read_bytes())
        env.sync()
    finally:
        env.close()
    return len(files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--map-size", type=int, default=1_000_000_000_000)
    args = parser.parse_args()
    count = build_image_lmdb(args.image_dir, args.output_dir, args.map_size)
    print(f"Packed {count} images into {args.output_dir}")


if __name__ == "__main__":
    main()
