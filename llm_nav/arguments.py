"""Dataclass definitions shared by training and SNA evaluation."""

from dataclasses import dataclass, field
from typing import Literal

import transformers


@dataclass
class ModelArguments:
    model_path: str = "xyz9911/FLAME-init"
    checkpoint_path: str = "checkpoints/Flame_checkpoint-2500"
    only_attend_immediate_media: bool = True


@dataclass
class DataArguments:
    dataset: str = "data/touchdown"
    train_if_data_path: str = "data/touchdown/ft_data/train.json"
    eval_if_data_path: str = "data/touchdown/ft_data/test.json"
    eval_split: str = "test"
    img_db: str = "data/features/flame_clip_touchdown_120_crop_lmdb"
    img_insert_db: str | None = "data/features/flame_clip_touchdown_insert_120_crop_lmdb"
    img_raw_db: str | None = "data/images/touchdown_crop_120_lmdb"
    img_raw_insert_db: str | None = "data/images/touchdown_insert_crop_120_lmdb"
    insert_task_dataset: str | None = "data/touchdown/insert_touchdown/insert_task1_dataset.json"
    task_level: int = 1
    legacy_navigation_mode: bool = True
    store_feature: bool = True
    task: Literal["instruction_following", "pretraining"] = "instruction_following"
    batch_size: int = 128
    micro_batch_size: int = 1
    env_batch_size: int = 1
    eval_data_size: int = -1
    max_route_len: int = 60
    temperature: float = 0.0
    decoding_paths: int = 1
    dino_det_path: str = "IDEA-Research/grounding-dino-base"
    siglip_path: str = "google/siglip2-so400m-patch16-naflex"
    qwen_path: str = "Qwen/Qwen3-VL-4B-Instruct"
    label_file: str = "llm_nav/sna/labels.txt"
    object_detect_threshold: float = 0.34
    object_detect_prompt: str = "A traffic sign pole."
    debug_dir: str = "outputs/sna_debug"
    save_debug: bool = False
    offline: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: str = "outputs/checkpoints"
    learning_rate: float = 1e-4
    optim: str = "adamw_torch"
    bf16: bool = True
    tf32: bool = False
    model_max_length: int = 2048
    warmup_ratio: float = 0.01
    num_train_epochs: float = 25
    save_strategy: str = "steps"
    eval_strategy: str = "steps"
    eval_steps: int = 200
    save_steps: int = 100
    lr_scheduler_type: str = "cosine"
    report_to: str = "tensorboard"
    remove_unused_columns: bool = False
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    save_total_limit: int = 25
    navigation_eval: bool = False
