"""Grounding-DINO, SigLIP2, and Qwen3-VL semantic rule assistant."""

from __future__ import annotations

import math
import re
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    pipeline,
)


class SemanticNavigationAssistant:
    """Detect traffic signs and correct actions that violate visible rules."""

    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.threshold = float(config.get("object_detect_threshold", 0.34))
        self.detection_prompt = config.get(
            "object_detect_prompt", "A traffic sign pole."
        )
        self.debug_dir = Path(config.get("debug_dir", "outputs/sna_debug"))
        self.save_debug = bool(config.get("save_debug", False))
        if self.save_debug:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.last_detection_flags: list[bool] = []

        dino_path = config["dino_det_path"]
        self.detector_processor = AutoProcessor.from_pretrained(dino_path)
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(
            dino_path
        ).to(self.device)
        self.detector.eval()

        pipeline_device = 0 if self.device == "cuda" else -1
        self.classifier = pipeline(
            task="zero-shot-image-classification",
            model=config["siglip_path"],
            device=pipeline_device,
            batch_size=int(config.get("sna_batch_size", 12)),
        )
        self.labels = self._load_labels(config["label_file"])

        self.qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
            config["qwen_path"],
            torch_dtype=torch.bfloat16,
            attn_implementation=config.get(
                "attn_implementation", "flash_attention_2"
            ),
            device_map="auto",
        )
        self.qwen_model.eval()
        self.qwen_processor = AutoProcessor.from_pretrained(
            config["qwen_path"], padding_side="left"
        )
        if hasattr(self.qwen_processor, "tokenizer"):
            self.qwen_processor.tokenizer.padding_side = "left"

    @staticmethod
    def _load_labels(path):
        labels = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                if ":" in line:
                    line = line.split(":", 1)[1].strip()
                    line = f"a traffic sign showing {line}"
                labels.append(line)
        return labels

    @staticmethod
    def _best_vertical_box(image, boxes, scores):
        best = None
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box.tolist()
            if y2 - y1 <= x2 - x1:
                continue
            value = float(score.item())
            if best is None or value > best["score"]:
                best = {
                    "image": image.crop((x1, y1, x2, y2)),
                    "box": (x1, y1, x2, y2),
                    "score": value,
                }
        return best

    @staticmethod
    def _draw_dashed_line(draw, start, end, fill="red", width=5, dash=10):
        x1, y1 = start
        x2, y2 = end
        length = math.hypot(x2 - x1, y2 - y1)
        if length == 0:
            return
        unit_x, unit_y = (x2 - x1) / length, (y2 - y1) / length
        for distance in range(0, int(length), dash * 2):
            end_distance = min(distance + dash, length)
            draw.line(
                [
                    (x1 + unit_x * distance, y1 + unit_y * distance),
                    (x1 + unit_x * end_distance, y1 + unit_y * end_distance),
                ],
                fill=fill,
                width=width,
            )

    @classmethod
    def _draw_dashed_box(cls, draw, box, fill="red", width=5):
        x1, y1, x2, y2 = box
        cls._draw_dashed_line(draw, (x1, y1), (x2, y1), fill, width)
        cls._draw_dashed_line(draw, (x2, y1), (x2, y2), fill, width)
        cls._draw_dashed_line(draw, (x2, y2), (x1, y2), fill, width)
        cls._draw_dashed_line(draw, (x1, y2), (x1, y1), fill, width)

    @classmethod
    def _zoom_visualization(cls, image, crop, box):
        canvas = image.copy()
        draw = ImageDraw.Draw(canvas)
        width, height = image.size
        box_width, box_height = box[2] - box[0], box[3] - box[1]
        aspect = box_width / box_height if box_height else 1.0
        base = max(150, min(400, int(min(width, height) * 0.35)))
        zoom_width, zoom_height = (
            (base, max(1, int(base / aspect)))
            if aspect > 1
            else (max(1, int(base * aspect)), base)
        )
        zoom = crop.resize((zoom_width, zoom_height), Image.Resampling.LANCZOS)
        x, y = width - zoom_width - 30, 30
        zoom_box = (x, y, x + zoom_width, y + zoom_height)
        canvas.paste((255, 255, 255), (x - 2, y - 2, x + zoom_width + 2, y + zoom_height + 2))
        canvas.paste(zoom, (x, y))
        cls._draw_dashed_box(draw, box)
        cls._draw_dashed_box(draw, zoom_box)
        cls._draw_dashed_line(draw, (box[2], box[1]), (zoom_box[0], zoom_box[1]), width=4)
        cls._draw_dashed_line(draw, (box[2], box[3]), (zoom_box[0], zoom_box[3]), width=4)
        return canvas

    @staticmethod
    def _build_prompt(action, road_type, labels):
        return f"""
        # Role
        You are a cautious and strictly compliant autonomous driving assistant. Your priority is obeying **ALL** traffic regulatory signs above all else.

        **Golden Rule:** You must strictly distinguish between "Prohibitory Signs" (e.g., No Entry, No Right Turn) and "Mandatory Signs" (e.g., Straight Only Arrow, U-Turn Only).
        - If a **Prohibitory Sign** blocks your path, you must choose a legal alternative.
        - If a **Mandatory Sign** indicates a specific direction, for example, the turn-around sign, you **MUST** execute that specific action, even if other paths physically exist. Ignore the behavioral guidance provided by road type indicators.

        You are currently driving on a **{road_type}**.
        *Note: If you are on a two-way street, you can only forward or turn around.*

        # Visual Input
        The input is a driver's view image with a **Picture-in-Picture** overlay:
        - **Global View:** Used to localize and understand the sign and determine whether the part enclosed by the red line is a traffic sign.
        - **Zoomed Detail (Top-Right):** The GROUND TRUTH for the sign identity/content.

        # Inputs
        - **Original Planned Action:**
        {action}
        - **AI Sign Prediction:**
        {labels}

        # Task
        Analyze the visual scene to determine the **Single Immediate Next Action**. Your final answer should be selected from among "safe", "right", "left", "forward" and "turn around".

        # Analysis Steps
        1. **Truth or falsehood determination(Priority #1)** Based on the overall image, check if there are any rule signs. If no rule signs are found, directly output "safe".
        2. **Sign Identification:** Look at the 'Zoomed Detail'. Classify the sign:
            - **Prohibitory:** Red Circle/No Entry/Crossed-out arrows. (Meaning: Do NOT do X).
            - **Mandatory:** Blue Circle with White Arrow / White Arrow on Black. (Meaning: You MUST do X).
            - **Not a rule indicator post:** For example, cars, street lamps, traffic lights, signs, etc.
        3. **Relevance & Localization:** Look at the 'Global View' briefly to confirm the sign.
        4. **Compliance Check:**
            - **Check Mandatory:** Does the sign *dictate* a specific action? (e.g., If Sign is "Forward Only" and Plan is "Right", you are non-compliant).
            - **Check Prohibitory:** Does the sign *forbid* the 'Original Planned Action'? (e.g., If Sign is "No Entry" and Plan is "Forward", you are non-compliant).

        # Decision Logic
        1. **Output Specific Action ("turn around", "right", "forward", etc.) IF:**
            - **Scenario A (Mandatory Override):** A Mandatory Sign dictates a specific direction different from the 'Original Planned Action'.
            *Example: Plan is "Right", but Sign is "Forward Only" -> Output "forward".*
            *Example: Plan is "Turn around", but Sign is "Forward Only" -> Output "forward".*
            *Example: Plan is "Forward", but Sign is "U-Turn Only" -> Output "turn around".*
            - **Scenario B (Prohibitory Block):** A Prohibitory Sign blocks the 'Original Planned Action', forcing a deviation.
            *Example: Plan is "Forward", Sign is "No Entry" -> Output "turn around" (or safest alternative).*

        2. **Output "safe" IF:**
            - The 'Original Planned Action' matches the Mandatory Sign's instruction.
            - OR there is no sign prohibiting the action, and no mandatory sign forcing a different action.
            - OR the detected object is **NOT** a regulatory traffic sign. For example, billboards, landmarks or traffic lights.

        # Output Format
        Do not output Thinking.
        Do not explain your reasoning.
        Output exactly one line in the following format:
        **Final Answer:** [Only one of: "safe", "right", "left", "forward", "turn around"]
        """

    @staticmethod
    def parse_final_answer(text):
        match = re.search(
            r"\*\*Final Answer:\*\*\s*[\[\"']?\s*([A-Za-z_ ]+)",
            text,
            re.IGNORECASE,
        )
        value = match.group(1).strip().lower() if match else "safe"
        if "turn around" in value or "turn_around" in value:
            return "turn_around"
        if "forward" in value or "straight" in value:
            return "forward"
        if "left" in value:
            return "left"
        if "right" in value:
            return "right"
        return "safe"

    def run_batch(self, inputs_list):
        if not inputs_list:
            return []
        images = [item["image"].convert("RGB") for item in inputs_list]
        detector_inputs = self.detector_processor(
            images=images,
            text=[self.detection_prompt] * len(images),
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        with torch.inference_mode():
            detector_outputs = self.detector(**detector_inputs)
        detections = self.detector_processor.post_process_grounded_object_detection(
            detector_outputs,
            target_sizes=torch.tensor([image.size[::-1] for image in images]),
            threshold=self.threshold,
        )

        results = [("Safe (no sign detected)", "safe") for _ in images]
        self.last_detection_flags = [False] * len(images)
        valid_indices, crops, visualizations = [], [], []
        for index, detection in enumerate(detections):
            best = self._best_vertical_box(
                images[index], detection["boxes"], detection["scores"]
            )
            if best is None:
                continue
            self.last_detection_flags[index] = True
            valid_indices.append(index)
            crops.append(best["image"].convert("RGB"))
            visualizations.append(
                self._zoom_visualization(images[index], best["image"], best["box"])
            )
        if not valid_indices:
            return results

        classified = self.classifier(crops, candidate_labels=self.labels)
        candidate_labels = [
            [prediction["label"] for prediction in predictions[:1]]
            for predictions in classified
        ]

        texts, qwen_images, qwen_videos = [], [], []
        for local_index, global_index in enumerate(valid_indices):
            item = inputs_list[global_index]
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": visualizations[local_index]},
                        {
                            "type": "text",
                            "text": self._build_prompt(
                                item["proposed_action"],
                                item["road_type"],
                                candidate_labels[local_index],
                            ),
                        },
                    ],
                }
            ]
            image_inputs, video_inputs = process_vision_info(messages)
            texts.append(
                self.qwen_processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )
            qwen_images.extend(image_inputs or [])
            qwen_videos.extend(video_inputs or [])
        model_inputs = self.qwen_processor(
            text=texts,
            images=qwen_images,
            videos=qwen_videos or None,
            padding=True,
            return_tensors="pt",
        ).to(self.qwen_model.device)
        with torch.inference_mode():
            generated = self.qwen_model.generate(
                **model_inputs, max_new_tokens=256, do_sample=False
            )
        trimmed = [
            output[len(input_ids) :]
            for input_ids, output in zip(model_inputs.input_ids, generated)
        ]
        responses = self.qwen_processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        for local_index, global_index in enumerate(valid_indices):
            response = responses[local_index]
            results[global_index] = (response, self.parse_final_answer(response))
            if self.save_debug:
                viewpoint = inputs_list[global_index].get(
                    "viewpoint", f"sample_{global_index}"
                )
                visualizations[local_index].save(
                    self.debug_dir / f"{viewpoint}_processed.jpg"
                )
        return results

    def run(self, image, original_action, road_type, viewpoint=""):
        if isinstance(image, (str, Path)):
            with Image.open(image) as opened:
                image = opened.convert("RGB")
        return self.run_batch(
            [
                {
                    "image": image,
                    "proposed_action": original_action,
                    "road_type": road_type,
                    "viewpoint": viewpoint,
                }
            ]
        )[0]
