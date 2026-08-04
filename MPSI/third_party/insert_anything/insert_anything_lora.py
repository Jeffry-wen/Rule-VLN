import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# print("sys.path is", sys.path)
from PIL import Image
import torch
import os
import numpy as np
import cv2
from diffusers import FluxFillPipeline, FluxPriorReduxPipeline
from .utils.utils import (
    box2squre,
    crop_back,
    expand_bbox,
    expand_image_mask,
    get_bbox_from_mask,
    pad_to_square,
)
from nunchaku.models.transformers.transformer_flux import NunchakuFluxTransformer2dModel

device = torch.device(f"cuda")
def insert_object_pipeline(
        source_image_pil,
        source_mask_pil,
        ref_image_pil,
        ref_mask_pil,
        pipe,
        redux,
        size=(1120, 1520),
        seed=666
):
    """

    - pipe: FluxFillPipeline
    - redux: FluxPriorReduxPipeline

    """

    ref_image = np.array(ref_image_pil)
    tar_image = np.array(source_image_pil)
    # print(ref_image.shape, tar_image.shape)
    ref_mask = (np.array(ref_mask_pil) > 128).astype(np.uint8)
    tar_mask = (np.array(source_mask_pil) > 128).astype(np.uint8)
    tar_mask = cv2.resize(tar_mask, (tar_image.shape[1], tar_image.shape[0]))

    ref_box_yyxx = get_bbox_from_mask(ref_mask)
    ref_mask_3 = np.stack([ref_mask] * 3, axis=-1)
    masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1 - ref_mask_3)

    y1, y2, x1, x2 = ref_box_yyxx
    masked_ref_image = masked_ref_image[y1:y2, x1:x2, :]
    ref_mask = ref_mask[y1:y2, x1:x2]

    masked_ref_image, ref_mask = expand_image_mask(masked_ref_image, ref_mask, ratio=1.3)
    masked_ref_image = pad_to_square(masked_ref_image, pad_value=255, random=False)

    kernel = np.ones((7, 7), np.uint8)
    tar_mask = cv2.dilate(tar_mask, kernel, iterations=2)

    tar_box_yyxx = get_bbox_from_mask(tar_mask)
    tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=1.2)

    tar_box_yyxx_crop = expand_bbox(tar_image, tar_box_yyxx, ratio=2)
    tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop)
    y1, y2, x1, x2 = tar_box_yyxx_crop

    old_tar_image = tar_image.copy()
    tar_image = tar_image[y1:y2, x1:x2, :]
    tar_mask = tar_mask[y1:y2, x1:x2]

    H1, W1 = tar_image.shape[:2]
    tar_mask = pad_to_square(tar_mask, pad_value=0)
    tar_mask = cv2.resize(tar_mask, size)

    masked_ref_image = cv2.resize(masked_ref_image.astype(np.uint8), size).astype(np.uint8)
    pipe_prior_output = redux(Image.fromarray(masked_ref_image))

    tar_image = pad_to_square(tar_image, pad_value=255)
    H2, W2 = tar_image.shape[:2]
    tar_image = cv2.resize(tar_image, size)

    diptych_ref_tar = np.concatenate([masked_ref_image, tar_image], axis=1)
    tar_mask_3 = np.stack([tar_mask] * 3, axis=-1)
    mask_black = np.zeros_like(tar_image)
    mask_diptych = np.concatenate([mask_black, tar_mask_3], axis=1)

    diptych_ref_tar = Image.fromarray(diptych_ref_tar)
    mask_diptych[mask_diptych == 1] = 255
    mask_diptych = Image.fromarray(mask_diptych)

    generator = torch.Generator(device).manual_seed(seed)

    edited_image = pipe(
        image=diptych_ref_tar,
        mask_image=mask_diptych,
        height=mask_diptych.size[1],
        width=mask_diptych.size[0],
        max_sequence_length=512,
        generator=generator,
        **pipe_prior_output,
    ).images[0]

    width, height = edited_image.size
    left = width // 2
    edited_image = edited_image.crop((left, 0, width, height))

    edited_image = np.array(edited_image)
    edited_image = crop_back(edited_image, old_tar_image, np.array([H1, W1, H2, W2]), np.array(tar_box_yyxx_crop))
    edited_image = Image.fromarray(edited_image)

    return edited_image
