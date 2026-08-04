"""Segmentation, placement, and photometric preparation for MPSI."""

from __future__ import annotations

import numpy as np
import torch
import torchvision.transforms.functional as vision_functional
from PIL import Image, ImageEnhance


def segmentation_mask(image, label, processor, model):
    inputs = processor(images=image, return_tensors="pt")
    with torch.inference_mode():
        logits = model(**inputs).logits
    predicted = torch.argmax(logits, dim=1).squeeze().cpu().numpy()
    resized = np.array(
        Image.fromarray(predicted.astype(np.uint8)).resize(
            image.size, Image.Resampling.NEAREST
        )
    )
    return Image.fromarray((resized == int(label)).astype(np.uint8) * 255)


def find_insert_point(mask):
    import cv2

    binary = np.array(mask.convert("L"))
    _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise ValueError("Road segmentation did not produce an insertion region")
    contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        raise ValueError("Largest road contour has zero area")
    center = (
        int(moments["m10"] / moments["m00"]),
        int(moments["m01"] / moments["m00"]),
    )
    intersections = []
    for index in range(len(contour)):
        first = contour[index][0]
        second = contour[(index + 1) % len(contour)][0]
        if (first[1] <= center[1] <= second[1]) or (
            second[1] <= center[1] <= first[1]
        ):
            delta_y = second[1] - first[1]
            if delta_y:
                x_value = int(
                    first[0]
                    + (center[1] - first[1])
                    * (second[0] - first[0])
                    / delta_y
                )
                intersections.append((x_value, center[1]))
    if not intersections:
        return center
    point = min(
        intersections, key=lambda item: abs(item[0] - binary.shape[1] // 2)
    )
    return (
        int(point[0] - (point[0] - center[0]) * 0.4),
        int(point[1] - (point[1] - center[1]) * 0.4),
    )


def placement_scale(y, image_height, minimum=0.03, maximum=0.5):
    ratio = y / image_height
    scale = -3.7467 * ratio**2 + 5.0448 * ratio - 1.5383
    return max(min(scale, maximum), minimum)


def place_shape(background, shape, shape_mask, point, rng):
    background = background.convert("RGBA")
    shape = shape.convert("RGBA")
    shape_mask = shape_mask.convert("L")
    scale = placement_scale(point[1], background.height)
    new_height = max(1, int(scale * background.height))
    new_width = max(1, int(new_height * shape.width / shape.height))
    shape = shape.resize((new_width, new_height), Image.Resampling.LANCZOS)
    shape_mask = shape_mask.resize(
        (new_width, new_height), Image.Resampling.NEAREST
    )
    x, y = point[0] - new_width // 2, point[1] - new_height

    left, top = max(0, point[0] - 100), max(0, point[1] - 100)
    right, bottom = min(background.width, point[0] + 100), min(
        background.height, point[1] + 100
    )
    region = np.array(background.crop((left, top, right, bottom)))[:, :, :3]
    source = np.array(shape)[:, :, :3]
    factor = np.clip(
        1.4 - 0.3 * (region.mean() / (source.mean() + 1e-5)),
        0.5,
        1.5,
    )
    shape = ImageEnhance.Brightness(shape).enhance(float(factor))
    array = np.array(shape).astype(np.int16)
    noise = rng.integers(-5, 6, array.shape, dtype=np.int16)
    shape = Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8))
    background.paste(shape, (x, y), shape)
    background = ImageEnhance.Color(background).enhance(1.1)
    target_mask = Image.new("L", background.size, 0)
    target_mask.paste(shape_mask, (x, y))
    return background, target_mask


def full_rule_mask(rule_image):
    white = Image.new("L", rule_image.size, 255)
    return vision_functional.to_pil_image(vision_functional.to_tensor(white))
