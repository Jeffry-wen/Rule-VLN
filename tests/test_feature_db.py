import io
import json

import lmdb
import torch
from PIL import Image

from llm_nav.data import FeatureDB


def write_tensor_lmdb(path, values):
    env = lmdb.open(str(path), map_size=10_000_000)
    with env.begin(write=True) as txn:
        for key, tensor in values.items():
            stream = io.BytesIO()
            torch.save(tensor, stream)
            txn.put(key.encode(), stream.getvalue())
    env.close()


def write_image_lmdb(path, values):
    env = lmdb.open(str(path), map_size=10_000_000)
    with env.begin(write=True) as txn:
        for key, color in values.items():
            stream = io.BytesIO()
            Image.new("RGB", (4, 4), color).save(stream, format="JPEG")
            txn.put(key.encode(), stream.getvalue())
    env.close()


def test_insert_overlay_and_manifest_filter(tmp_path):
    base, inserted = tmp_path / "base", tmp_path / "inserted"
    write_tensor_lmdb(
        base,
        {
            "node_a_0": torch.tensor([1.0]),
            "node_b_0": torch.tensor([2.0]),
        },
    )
    write_tensor_lmdb(
        inserted,
        {
            "node_a_0": torch.tensor([10.0]),
            "node_b_0": torch.tensor([20.0]),
        },
    )
    manifest = tmp_path / "task.json"
    manifest.write_text(json.dumps({"node_a": {}}), encoding="utf-8")
    database = FeatureDB(base, inserted, manifest, decode_type="torch")
    assert database.get("node_a_0").item() == 10
    assert database.get("node_b_0").item() == 2


def test_pil_decode_and_nearest_heading(tmp_path):
    image_db = tmp_path / "images"
    write_image_lmdb(image_db, {"pano_90": (255, 0, 0)})
    database = FeatureDB(image_db, decode_type="pil")
    image = database.get("pano_91")
    assert image.mode == "RGB"
    assert image.size == (4, 4)
