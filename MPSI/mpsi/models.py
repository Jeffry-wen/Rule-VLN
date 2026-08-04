"""Lazy loading of the models required by MPSI."""

from __future__ import annotations

import torch
from diffusers import FluxFillPipeline, FluxPriorReduxPipeline
from nunchaku.models.transformers.transformer_flux import (
    NunchakuFluxTransformer2dModel,
)
from transformers import (
    CLIPModel,
    CLIPProcessor,
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)


class ModelBundle:
    def __init__(self, config):
        self.segformer_processor = SegformerImageProcessor.from_pretrained(
            config.segformer_model
        )
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            config.segformer_model
        )
        transformer = NunchakuFluxTransformer2dModel.from_pretrained(
            config.flux_transformer
        )
        self.flux = FluxFillPipeline.from_pretrained(
            config.flux_fill_model,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )
        transformer.update_lora_params(config.insert_anything_lora)
        transformer.set_lora_strength(1)
        self.redux = FluxPriorReduxPipeline.from_pretrained(
            config.flux_redux_model
        ).to(dtype=torch.bfloat16)
        self.flux.enable_model_cpu_offload()
        self.flux.enable_vae_slicing()
        self.redux.enable_model_cpu_offload()
        self.clip = CLIPModel.from_pretrained(config.clip_model)
        self.clip_processor = CLIPProcessor.from_pretrained(config.clip_model)
