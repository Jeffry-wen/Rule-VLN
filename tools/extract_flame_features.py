#!/usr/bin/env python3
"""Extract FLAME visual features from an image LMDB into a tensor LMDB."""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import cv2
import lmdb
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPImageProcessor

from llm_nav.model.modeling_flamingo import FlamingoForConditionalGeneration


def serialize_tensor(tensor):
    stream = io.BytesIO()
    torch.save(tensor.cpu(), stream)
    return stream.getvalue()


def flush_batch(model, keys, images, transaction):
    if not keys:
        return
    pixel_values = torch.stack(images).to(model.device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        features = model._encode_vision_x(pixel_values)
    for key, feature in zip(keys, features):
        transaction.put(key, serialize_tensor(feature))


def extract_features(
    image_lmdb,
    output_lmdb,
    model_path,
    batch_size=64,
    map_size=1_000_000_000_000,
):
    source = lmdb.open(
        str(image_lmdb),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    Path(output_lmdb).mkdir(parents=True, exist_ok=True)
    target = lmdb.open(str(output_lmdb), map_size=int(map_size), subdir=True)
    model = FlamingoForConditionalGeneration.from_pretrained(
        model_path, device_map="cuda"
    )
    model.eval()
    processor = CLIPImageProcessor()
    keys, images, count = [], [], 0
    try:
        with source.begin(write=False) as source_txn, target.begin(write=True) as target_txn:
            cursor = source_txn.cursor()
            total = source.stat()["entries"]
            for key, payload in tqdm(cursor, total=total, desc="Extracting features"):
                array = cv2.imdecode(
                    np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if array is None:
                    raise ValueError(f"Failed to decode LMDB key: {key!r}")
                image = Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB))
                images.append(processor(image, return_tensors="pt")["pixel_values"][0])
                keys.append(key)
                if len(keys) == batch_size:
                    flush_batch(model, keys, images, target_txn)
                    count += len(keys)
                    keys, images = [], []
            flush_batch(model, keys, images, target_txn)
            count += len(keys)
        target.sync()
    finally:
        source.close()
        target.close()
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-lmdb", required=True)
    parser.add_argument("--output-lmdb", required=True)
    parser.add_argument("--model-path", default="xyz9911/FLAME-init")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--map-size", type=int, default=1_000_000_000_000)
    args = parser.parse_args()
    count = extract_features(
        args.image_lmdb,
        args.output_lmdb,
        args.model_path,
        args.batch_size,
        args.map_size,
    )
    print(f"Wrote {count} FLAME feature tensors.")


if __name__ == "__main__":
    main()
