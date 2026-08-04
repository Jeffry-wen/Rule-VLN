"""Touchdown instruction-following datasets used by FLAME."""

from __future__ import annotations

import copy
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import cv2
import lmdb
import numpy as np
import torch
import transformers
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor

from llm_nav import conversation as conversation_lib
from llm_nav.data import FeatureDB

IGNORE_INDEX = -100


def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    tokenized = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = [item.input_ids[0] for item in tokenized]
    lengths = [
        item.input_ids.ne(tokenizer.pad_token_id).sum().item() for item in tokenized
    ]
    return {
        "input_ids": input_ids,
        "labels": input_ids,
        "input_ids_lens": lengths,
        "labels_lens": lengths,
    }


def _mask_targets(target, tokenized_lens, speakers, answer_token_id):
    current = tokenized_lens[0]
    target[:current] = IGNORE_INDEX
    for length, speaker in zip(tokenized_lens[1:], speakers):
        if speaker == conversation_lib.default_conversation.roles[0]:
            target[current : current + length] = IGNORE_INDEX
        else:
            label_index = current
            while label_index < len(target) and target[label_index] != answer_token_id:
                target[label_index] = IGNORE_INDEX
                label_index += 1
        current += length
    target[target == answer_token_id] = IGNORE_INDEX


def _add_speaker_and_signal(header, source, concat_mode=False):
    end_signal = "<|endofchunk|>"
    separator = conversation_lib.default_conversation.sep
    conversation = separator if concat_mode else header
    for sentence in source:
        role = sentence["from"]
        if role.lower() == "user":
            role = conversation_lib.default_conversation.roles[0]
        elif role.lower() == "gpt":
            role = conversation_lib.default_conversation.roles[1]
        else:
            role = "unknown"
        if role.lower() == "gpt":
            if "num_images" in sentence:
                role = "<image>" + role
            sentence["value"] = (
                role + ":<answer>" + sentence["value"] + end_signal + separator
            )
        else:
            for _ in range(sentence.get("num_images", 0)):
                role = "<image>" + role
            sentence["value"] = role + ": " + sentence["value"] + separator
        conversation += sentence["value"]
    return conversation


def preprocess(
    sources: Sequence,
    tokenizer: transformers.PreTrainedTokenizer,
    inference_mode: bool = False,
    concat_mode: bool = False,
    task: str = "instruction_following",
) -> Dict:
    if task not in {"instruction_following", "pretraining"}:
        raise ValueError(f"Unsupported task: {task}")
    answer_token_id = tokenizer("<answer>", add_special_tokens=False)["input_ids"][-1]
    header = conversation_lib.default_conversation.system
    conversations = []
    for source in sources:
        text = _add_speaker_and_signal(header, source, concat_mode=concat_mode)
        if inference_mode:
            text += conversation_lib.default_conversation.roles[1] + ":<answer>"
        conversations.append(text)
    tokenized = _tokenize_fn(conversations, tokenizer)
    input_ids = tokenized["input_ids"]
    if concat_mode:
        input_ids = [ids[1:] for ids in input_ids]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        lengths = _tokenize_fn(
            [header] + [sentence["value"] for sentence in source], tokenizer
        )["input_ids_lens"]
        lengths = [length - 1 if index else length for index, length in enumerate(lengths)]
        _mask_targets(target, lengths, [item["from"] for item in source], answer_token_id)
    return {"input_ids": input_ids, "labels": targets}


class LazySupervisedDataset(Dataset):
    """Lazy Touchdown dataset for supervised FLAME fine-tuning."""

    def __init__(
        self,
        data_path: str,
        img_db: str,
        img_insert_db: str | None,
        tokenizer: transformers.PreTrainedTokenizer,
        store_feature: bool,
        inference_mode: bool,
        task: str = "instruction_following",
        data_size: int | None = None,
    ):
        with Path(data_path).open("r", encoding="utf-8") as handle:
            self.items = json.load(handle)
        if data_size is not None and data_size > 0:
            self.items = self.items[:data_size]
        self.tokenizer = tokenizer
        self.task = task
        self.store_feature = store_feature
        self.inference_mode = inference_mode
        self.image_processor = CLIPImageProcessor()
        self.image_cache: dict[str, torch.Tensor] = {}
        self.feature_db = None
        self.raw_env = None
        if store_feature:
            self.feature_db = FeatureDB(
                default_dir=img_db,
                insert_dir=img_insert_db,
                decode_type="torch",
                max_cache=512,
            )
        else:
            self.raw_env = lmdb.open(
                img_db,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        source = self.items[index]
        images = []
        for image_id in source.get("images", []):
            images.append(
                self.feature_db.get(image_id)
                if self.store_feature
                else self._get_raw_image(image_id)
            )
        conversations = copy.deepcopy(source["conversations"])
        data = preprocess(
            [conversations],
            self.tokenizer,
            inference_mode=self.inference_mode,
            task=self.task,
        )
        result = {"input_ids": data["input_ids"][0], "labels": data["labels"][0]}
        if images:
            result["images_ft" if self.store_feature else "images"] = images
        return result

    def _get_raw_image(self, image_id: str) -> torch.Tensor:
        if image_id in self.image_cache:
            return self.image_cache[image_id]
        key = image_id.rsplit("_", 1)[0]
        if not key.lower().endswith(".jpg"):
            key += ".jpg"
        with self.raw_env.begin() as txn:
            payload = txn.get(key.encode("ascii"))
            if payload is None:
                payload = txn.get(Path(key).name.encode("ascii"))
        if payload is None:
            raise KeyError(f"Raw panorama not found in LMDB: {image_id}")
        image_array = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image_array is None:
            raise ValueError(f"Failed to decode panorama: {image_id}")
        image = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
        tensor = self.image_processor(image)["pixel_values"][0]
        self.image_cache[image_id] = tensor
        return tensor


@dataclass
class DataCollatorForSupervisedDataset:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]
        batch = {
            "lang_x": nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            ),
            "labels": nn.utils.rnn.pad_sequence(
                labels, batch_first=True, padding_value=IGNORE_INDEX
            ),
        }
        batch["attention_mask"] = batch["lang_x"].ne(self.tokenizer.pad_token_id)
        key = "images_ft" if "images_ft" in instances[0] else "images"
        if key == "images_ft":
            vision = [torch.stack(instance[key]) for instance in instances]
            batch["vision_x"] = nn.utils.rnn.pad_sequence(vision, batch_first=True).unsqueeze(2)
        elif key in instances[0]:
            vision = [torch.stack(instance[key]) for instance in instances]
            batch["vision_x"] = nn.utils.rnn.pad_sequence(vision, batch_first=True).unsqueeze(2)
        return batch


def make_supervised_data_module(tokenizer, data_args):
    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        data_path=data_args.train_if_data_path,
        img_db=data_args.img_db,
        img_insert_db=data_args.img_insert_db,
        task=data_args.task,
        store_feature=data_args.store_feature,
        inference_mode=False,
    )
    eval_dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        data_path=data_args.eval_if_data_path,
        img_db=data_args.img_db,
        img_insert_db=data_args.img_insert_db,
        task=data_args.task,
        data_size=data_args.eval_data_size,
        store_feature=data_args.store_feature,
        inference_mode=False,
    )
    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": DataCollatorForSupervisedDataset(tokenizer),
    }
