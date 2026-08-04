#!/usr/bin/env python3
"""Fine-tune the original FLAME model on Touchdown."""

from __future__ import annotations

import os
from pathlib import Path

import transformers

from llm_nav.arguments import DataArguments, ModelArguments, TrainingArguments
from llm_nav.config import FlamingoConfig
from llm_nav.configuration import (
    apply_overrides,
    common_config_parser,
    flatten_sections,
    load_yaml,
)
from llm_nav.dataset import make_supervised_data_module
from llm_nav.model.modeling_flamingo import FlamingoForConditionalGeneration
from llm_nav.sim.env import TouchdownBatch
from llm_nav.trainer import FlameTrainer


def main():
    parser = common_config_parser("Fine-tune FLAME on Touchdown.")
    args = parser.parse_args()
    config = apply_overrides(load_yaml(args.config), args.set)
    values = flatten_sections(config, "model", "data", "training")
    hf_parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = hf_parser.parse_dict(
        values, allow_extra_keys=True
    )

    if data_args.offline:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    global_batch = int(config.get("training", {}).get("global_batch_size", data_args.batch_size))
    denominator = data_args.micro_batch_size * world_size
    if global_batch % denominator:
        raise ValueError(
            "global_batch_size must be divisible by micro_batch_size * WORLD_SIZE"
        )
    training_args.gradient_accumulation_steps = global_batch // denominator
    training_args.per_device_train_batch_size = data_args.micro_batch_size

    model_config = FlamingoConfig.from_pretrained(model_args.model_path)
    model_config.only_attend_immediate_media = model_args.only_attend_immediate_media
    model_config.feature_as_input = data_args.store_feature
    model = FlamingoForConditionalGeneration.from_pretrained(
        model_args.model_path,
        config=model_config,
    )
    model.lang_encoder.config.vocab_size = len(model.text_tokenizer)
    tokenizer = model.text_tokenizer
    tokenizer.model_max_length = training_args.model_max_length
    data_module = make_supervised_data_module(tokenizer, data_args)

    navigation_env = None
    if training_args.navigation_eval:
        navigation_env = TouchdownBatch(
            data_args,
            splits=[data_args.eval_split],
            name=Path(data_args.dataset).name,
        )
    trainer = FlameTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        navigation_env=navigation_env,
        navigation_args=data_args,
        **data_module,
    )
    trainer.train(resume_from_checkpoint=config.get("training", {}).get("resume_from_checkpoint"))
    trainer.save_model()


if __name__ == "__main__":
    main()
