"""Hugging Face trainer customizations used by FLAME."""

from __future__ import annotations

import json
from pathlib import Path

from torch import nn
from transformers import Trainer
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names

from llm_nav.agent import run_navigation


def unwrap_model(model: nn.Module) -> nn.Module:
    return unwrap_model(model.module) if hasattr(model, "module") else model


class FlameTrainer(Trainer):
    """Trainer with FLAME optimizer grouping and optional navigation evaluation."""

    def __init__(self, *args, navigation_env=None, navigation_args=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.navigation_env = navigation_env
        self.navigation_args = navigation_args

    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix,
        )
        if self.navigation_env is None:
            return output
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        tokenizer = self.data_collator.tokenizer
        metrics, trajectories = run_navigation(
            self.navigation_env,
            model,
            tokenizer,
            self.navigation_args,
        )
        result_path = Path(self.args.output_dir) / f"{metric_key_prefix}_navigation.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps({"metrics": metrics, "trajs": trajectories}, indent=2),
            encoding="utf-8",
        )
        output.metrics.update(
            {f"{metric_key_prefix}_{key}": value for key, value in metrics.items()}
        )
        return output

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        decay_names = get_parameter_names(self.model, ALL_LAYERNORM_LAYERS)
        decay_names = [
            name
            for name in decay_names
            if not any(token in name for token in ("bias", "ff_gate", "attn_gate", "norm"))
        ]
        grouped = [
            {
                "params": [
                    parameter
                    for name, parameter in self.model.named_parameters()
                    if name in decay_names and parameter.requires_grad
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    parameter
                    for name, parameter in self.model.named_parameters()
                    if name not in decay_names and parameter.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(grouped, **optimizer_kwargs)
        return self.optimizer
