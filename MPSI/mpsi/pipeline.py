"""End-to-end MPSI benchmark construction pipeline."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from mpsi.geometry import (
    find_insert_point,
    full_rule_mask,
    place_shape,
    segmentation_mask,
)
from mpsi.graph import TouchdownGraph
from mpsi.rules import RuleCatalog


def read_scores(path):
    scores = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.strip().split("\t")
            if not fields:
                continue
            scores.append((fields[0], float(fields[-1].split(":")[-1])))
    return sorted(scores, key=lambda item: item[1], reverse=True)


def level_nodes(scores, level, group_count=4):
    if level < 1 or level > group_count:
        raise ValueError(f"level must be in [1, {group_count}]")
    group_size = len(scores) // group_count
    if group_size == 0:
        raise ValueError("The score file has fewer nodes than benchmark levels")
    groups = [
        scores[index : index + group_size]
        for index in range(0, len(scores), group_size)
    ]
    return [str(item[0]) for item in groups[group_count - level]]


def stable_seed(global_seed, level, node):
    digest = hashlib.sha256(f"{global_seed}:{level}:{node}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


class MPSIPipeline:
    def __init__(self, config, level, output_dir):
        self.config = config
        self.level = level
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.panorama_output = self.output_dir / "inserted_panoramas"
        self.panorama_output.mkdir(exist_ok=True)
        self.graph = TouchdownGraph(config.nodes_file, config.links_file)
        self.catalog = RuleCatalog(config.rule_image_dir, config.rule_labels)
        from mpsi.models import ModelBundle

        self.models = ModelBundle(config)
        self.shapes = sorted(Path(config.shape_dir).glob("*.png"))
        if not self.shapes:
            raise FileNotFoundError(f"No shape templates found in {config.shape_dir}")

    def _process_node(self, node):
        seed = stable_seed(self.config.seed, self.level, node)
        python_rng = random.Random(seed)
        numpy_rng = np.random.default_rng(seed)
        yaw, actions, action_codes = self.graph.action_space(node)
        rule_path, rule_entry = self.catalog.select(action_codes, python_rng)
        shape_path = python_rng.choice(self.shapes)
        source_path = Path(self.config.panorama_dir) / f"{node}.jpg"
        if not source_path.exists():
            raise FileNotFoundError(f"Panorama not found: {source_path}")

        with Image.open(source_path) as source_image:
            source_image = source_image.convert("RGB")
        with Image.open(shape_path) as shape_image:
            shape_image = shape_image.convert("RGB")
        with Image.open(rule_path) as rule_image:
            rule_image = rule_image.convert("RGB")

        width = source_image.width
        crop = source_image.crop((width // 3, 0, 2 * width // 3, source_image.height))
        road_mask = segmentation_mask(
            crop,
            self.config.road_label,
            self.models.segformer_processor,
            self.models.segformer,
        )
        insert_point = find_insert_point(road_mask)
        shape_mask = segmentation_mask(
            shape_image,
            self.config.shape_label,
            self.models.segformer_processor,
            self.models.segformer,
        )
        prepared_crop, target_mask = place_shape(
            crop, shape_image, shape_mask, insert_point, numpy_rng
        )
        if self.config.use_background_segmentation:
            from mpsi.optional_background import segment_rule

            rule_mask = segment_rule(rule_path, self.config)
        else:
            rule_mask = full_rule_mask(rule_image)
        from third_party.insert_anything.insert_anything_lora import (
            insert_object_pipeline,
        )

        generated_crop = insert_object_pipeline(
            prepared_crop.convert("RGB"),
            target_mask,
            rule_image,
            rule_mask,
            self.models.flux,
            self.models.redux,
            seed=self.config.diffusion_seed,
        )
        final = source_image.copy()
        final.paste(generated_crop.convert("RGB"), (width // 3, 0))
        output_path = self.panorama_output / f"{node}.jpg"
        final.save(output_path, quality=95)

        clip_inputs = self.models.clip_processor(
            text=[f"There is a {rule_entry['label']} traffic sign"],
            images=final,
            return_tensors="pt",
            padding=True,
        )
        clip_score = float(
            self.models.clip(**clip_inputs).logits_per_image.detach().cpu().item()
        )
        return {
            "node": node,
            "level": self.level,
            "node_seed": seed,
            "diffusion_seed": self.config.diffusion_seed,
            "panorama_yaw": yaw,
            "available_actions": actions,
            "action_node": rule_entry["action_node"],
            "rule_img": rule_entry["file"].replace("\\", "/"),
            "shape_img": shape_path.name,
            "clip_score": clip_score,
            "output_image": str(output_path),
            "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        }

    def run(self, limit=None, skip_quality_filter=False):
        selected = level_nodes(
            read_scores(self.config.score_file),
            self.level,
            self.config.group_count,
        )
        if limit is not None:
            selected = selected[:limit]
        records, failures = [], []
        for node in tqdm(selected, desc=f"MPSI level {self.level}"):
            try:
                records.append(self._process_node(node))
            except Exception as error:
                failures.append({"node": node, "error": str(error)})

        accepted = records
        if (
            len(records) >= 2
            and len({item["clip_score"] for item in records}) >= 2
            and not skip_quality_filter
        ):
            from sklearn.mixture import GaussianMixture

            values = np.array([item["clip_score"] for item in records]).reshape(-1, 1)
            gmm = GaussianMixture(n_components=2, random_state=0).fit(values)
            labels = gmm.predict(values)
            probabilities = gmm.predict_proba(values)
            high_cluster = int(np.argmax(gmm.means_.reshape(-1)))
            accepted = [
                item
                for item, label, probability in zip(records, labels, probabilities)
                if label == high_cluster
                and probability[high_cluster] > self.config.quality_probability
            ]
        accepted_nodes = {item["node"] for item in accepted}
        for record in records:
            record["accepted"] = record["node"] in accepted_nodes

        manifest_path = self.output_dir / "generation_manifest.jsonl"
        with manifest_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        task_path = self.output_dir / f"insert_task{self.level}_dataset.json"
        task_path.write_text(
            json.dumps(
                {
                    item["node"]: {
                        "node": item["node"],
                        "rule_img": item["rule_img"],
                        "clip_score": str(item["clip_score"]),
                        "action_node": item["action_node"],
                    }
                    for item in accepted
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (self.output_dir / "failures.json").write_text(
            json.dumps(failures, indent=2), encoding="utf-8"
        )
        return len(records), len(accepted), len(failures)
