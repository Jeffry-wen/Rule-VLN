"""Unified access to FLAME feature tensors and raw navigation images."""

from __future__ import annotations

import io
import json
import pickle
import weakref
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import lmdb
import torch
from PIL import Image


class FeatureDB:
    """Read a base store with an optional rule-inserted overlay.

    Stores may be LMDB directories or flat folders. Inserted entries override
    base entries. When a task manifest is provided, only inserted entries whose
    panorama ID appears in the manifest are indexed.
    """

    def __init__(
        self,
        default_dir: str | Path | None,
        insert_dir: str | Path | None = None,
        insert_task_dataset: str | Path | None = None,
        decode_type: str = "torch",
        max_cache: int = 4096,
        cache_index: bool = True,
    ) -> None:
        if decode_type not in {"torch", "pil"}:
            raise ValueError("decode_type must be 'torch' or 'pil'")
        self.default_dir = Path(default_dir).expanduser() if default_dir else None
        self.insert_dir = Path(insert_dir).expanduser() if insert_dir else None
        self.decode_type = decode_type
        self.max_cache = max_cache
        self.cache_index = cache_index
        self.insert_whitelist = self._load_whitelist(insert_task_dataset)
        self.task_name = (
            Path(insert_task_dataset).stem if insert_task_dataset else "all"
        )
        self.map: dict[str, dict[str, Any]] = {}
        self.prefix_index: dict[str, list[tuple[int, str]]] = defaultdict(list)
        self.cache: OrderedDict[str, Any] = OrderedDict()
        self.lmdb_envs: dict[str, lmdb.Environment] = {}
        self._load_or_build_index()

    @staticmethod
    def _load_whitelist(path: str | Path | None) -> set[str] | None:
        if not path:
            return None
        manifest_path = Path(path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Task manifest not found: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            return set(data)
        if isinstance(data, list):
            return {
                str(item.get("node", item.get("panoid")))
                for item in data
                if isinstance(item, dict) and (item.get("node") or item.get("panoid"))
            }
        raise ValueError(f"Unsupported task manifest format: {manifest_path}")

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["lmdb_envs"] = {}
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__ = state
        self.lmdb_envs = {}

    def close(self) -> None:
        for env in self.lmdb_envs.values():
            env.close()
        self.lmdb_envs.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def _is_lmdb(path: Path) -> bool:
        return path.is_dir() and (path / "data.mdb").exists()

    def _get_lmdb_env(self, path: str) -> lmdb.Environment:
        if path not in self.lmdb_envs:
            self.lmdb_envs[path] = lmdb.open(
                path,
                subdir=True,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_dbs=0,
            )
        return self.lmdb_envs[path]

    @staticmethod
    def _clean_id(raw_id: str) -> str:
        suffix = Path(raw_id).suffix.lower()
        return raw_id[: -len(suffix)] if suffix in {".pt", ".jpg", ".jpeg", ".png"} else raw_id

    def _allowed_insert_key(self, clean_id: str) -> bool:
        if self.insert_whitelist is None:
            return True
        if clean_id in self.insert_whitelist:
            return True
        if "_" not in clean_id:
            return False
        return clean_id.rsplit("_", 1)[0] in self.insert_whitelist

    def _add_to_index(self, clean_id: str, metadata: dict[str, Any]) -> None:
        self.map[clean_id] = metadata
        if "_" in clean_id:
            prefix, suffix = clean_id.rsplit("_", 1)
            if suffix.isdigit():
                self.prefix_index[prefix].append((int(suffix), clean_id))

    def _scan_lmdb(self, path: Path, inserted: bool) -> None:
        path_str = str(path)
        env = lmdb.open(
            path_str,
            subdir=True,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        try:
            with env.begin(write=False) as txn:
                for key in txn.cursor().iternext(keys=True, values=False):
                    raw_id = key.decode("utf-8")
                    clean_id = self._clean_id(raw_id)
                    if inserted and not self._allowed_insert_key(clean_id):
                        continue
                    self._add_to_index(
                        clean_id,
                        {"type": "lmdb", "path": path_str, "key": key},
                    )
        finally:
            env.close()

    def _scan_folder(self, path: Path, inserted: bool) -> None:
        patterns = ("*.pt",) if self.decode_type == "torch" else ("*.jpg", "*.jpeg", "*.png")
        for pattern in patterns:
            for file_path in path.glob(pattern):
                clean_id = file_path.stem
                if inserted and not self._allowed_insert_key(clean_id):
                    continue
                self._add_to_index(
                    clean_id,
                    {"type": "file", "path": str(file_path), "key": None},
                )

    def _cache_path(self) -> Path | None:
        if not self.cache_index or not self.default_dir:
            return None
        insert_name = self.insert_dir.name if self.insert_dir else "none"
        name = (
            f".feature_index_{self.default_dir.name}_{insert_name}_"
            f"{self.task_name}_{self.decode_type}.pkl"
        )
        return self.default_dir.parent / name

    def _load_or_build_index(self) -> None:
        for path in (self.default_dir, self.insert_dir):
            if path and not path.exists():
                raise FileNotFoundError(f"Feature store not found: {path}")

        cache_path = self._cache_path()
        if cache_path and cache_path.exists():
            try:
                with cache_path.open("rb") as handle:
                    cached = pickle.load(handle)
                self.map = cached["map"]
                self.prefix_index = cached["prefix_index"]
                return
            except (OSError, KeyError, pickle.PickleError):
                pass

        for path, inserted in ((self.default_dir, False), (self.insert_dir, True)):
            if not path:
                continue
            if self._is_lmdb(path):
                self._scan_lmdb(path, inserted)
            else:
                self._scan_folder(path, inserted)

        for values in self.prefix_index.values():
            values.sort()
        if cache_path:
            try:
                with cache_path.open("wb") as handle:
                    pickle.dump(
                        {"map": self.map, "prefix_index": self.prefix_index},
                        handle,
                    )
            except OSError:
                pass

    def _find_nearest_id(self, image_id: str) -> str | None:
        if "_" not in image_id:
            return None
        prefix, suffix = image_id.rsplit("_", 1)
        if not suffix.isdigit() or prefix not in self.prefix_index:
            return None
        target = int(suffix)
        return min(self.prefix_index[prefix], key=lambda item: abs(item[0] - target))[1]

    def _read(self, metadata: dict[str, Any]) -> Any:
        if metadata["type"] == "lmdb":
            env = self._get_lmdb_env(metadata["path"])
            with env.begin(write=False) as txn:
                payload = txn.get(metadata["key"])
            if payload is None:
                raise KeyError(f"Missing LMDB key: {metadata['key']!r}")
            stream = io.BytesIO(payload)
            if self.decode_type == "pil":
                with Image.open(stream) as image:
                    return image.convert("RGB")
            return torch.load(stream, map_location="cpu", weights_only=False)

        if self.decode_type == "pil":
            with Image.open(metadata["path"]) as image:
                return image.convert("RGB")
        return torch.load(metadata["path"], map_location="cpu", weights_only=False)

    def get(self, image_id: str) -> Any:
        if image_id in self.cache:
            cached = self.cache[image_id]
            value = cached if self.decode_type == "pil" else cached()
            if value is not None:
                self.cache.move_to_end(image_id)
                return value
            del self.cache[image_id]

        clean_id = self._clean_id(image_id)
        lookup_id = clean_id if clean_id in self.map else self._find_nearest_id(clean_id)
        if lookup_id is None:
            raise FileNotFoundError(f"Image or feature not found: {image_id}")
        value = self._read(self.map[lookup_id])
        self.cache[image_id] = value if self.decode_type == "pil" else weakref.ref(value)
        if len(self.cache) > self.max_cache:
            self.cache.popitem(last=False)
        return value
