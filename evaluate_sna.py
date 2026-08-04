#!/usr/bin/env python3
"""Run Rule-VLN SNA evaluation on Touchdown."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import transformers

from llm_nav.agent import run_navigation
from llm_nav.arguments import DataArguments, ModelArguments
from llm_nav.config import FlamingoConfig
from llm_nav.configuration import (
    apply_overrides,
    common_config_parser,
    flatten_sections,
    load_yaml,
)
from llm_nav.model.modeling_flamingo import FlamingoForConditionalGeneration
from llm_nav.sim.env import TouchdownBatch
from llm_nav.sna import SemanticNavigationAssistant


def main():
    parser = common_config_parser("Evaluate FLAME with the Semantic Navigation Assistant.")
    parser.add_argument("--level", type=int, choices=(1, 2, 3, 4), default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = apply_overrides(load_yaml(args.config), args.set)
    level = args.level or int(config.get("evaluation", {}).get("level", 1))
    if level not in {1, 2, 3, 4}:
        raise ValueError("Rule level must be one of 1, 2, 3, or 4")

    model_values = flatten_sections(config, "model")
    data_values = flatten_sections(config, "data", "sna", "evaluation")
    data_values["task_level"] = level
    for key in ("insert_task_dataset", "debug_dir"):
        if isinstance(data_values.get(key), str):
            data_values[key] = data_values[key].format(level=level)

    hf_parser = transformers.HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = hf_parser.parse_dict(
        {**model_values, **data_values}, allow_extra_keys=True
    )
    if data_args.offline:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ.setdefault("WANDB_MODE", "disabled")

    sna = SemanticNavigationAssistant(
        {
            "dino_det_path": data_args.dino_det_path,
            "siglip_path": data_args.siglip_path,
            "qwen_path": data_args.qwen_path,
            "label_file": data_args.label_file,
            "debug_dir": data_args.debug_dir,
            "save_debug": data_args.save_debug,
            "object_detect_threshold": data_args.object_detect_threshold,
            "object_detect_prompt": data_args.object_detect_prompt,
            **config.get("sna_runtime", {}),
        }
    )

    model_config = FlamingoConfig.from_pretrained(model_args.checkpoint_path)
    model_config.only_attend_immediate_media = model_args.only_attend_immediate_media
    model = FlamingoForConditionalGeneration.from_pretrained(
        model_args.checkpoint_path,
        torch_dtype=torch.float16,
        config=model_config,
        device_map=config.get("runtime", {}).get("device_map", "cuda"),
    )
    model.eval()
    tokenizer = model.text_tokenizer

    dataset_name = Path(data_args.dataset).name
    environment = TouchdownBatch(
        data_args,
        splits=[data_args.eval_split],
        name=dataset_name,
    )
    metrics, trajectories = run_navigation(
        environment, model, tokenizer, data_args, sna_module=sna
    )

    configured_output = config.get("evaluation", {}).get(
        "output", f"outputs/eval/task{level}_{data_args.eval_split}.json"
    )
    if isinstance(configured_output, str):
        configured_output = configured_output.format(
            level=level, split=data_args.eval_split
        )
    output_path = args.output or Path(configured_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "config": str(Path(args.config).resolve()),
                "level": level,
                "metrics": metrics,
                "trajs": trajectories,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))
    print(f"Saved evaluation results to {output_path}")


if __name__ == "__main__":
    main()
