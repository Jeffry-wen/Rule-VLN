#!/usr/bin/env python3
"""Project StreetLearn panoramas into the 120-degree FLAME views."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def relative_angle(current, target):
    angle = target - current
    if angle > 180:
        angle -= 360
    if angle <= -180:
        angle += 360
    return angle


def classify_direction(angle):
    if -45 <= angle <= 45:
        return "forward"
    if 45 < angle <= 135:
        return "right"
    if -135 <= angle < -45:
        return "left"
    return "backward"


class Equirectangular:
    def __init__(self, image_path):
        self.image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if self.image is None:
            raise ValueError(f"Failed to read panorama: {image_path}")
        self.height, self.width = self.image.shape[:2]

    def perspective(self, fov, theta, phi, height, width, radius=128):
        center_x, center_y = (width - 1) / 2.0, (height - 1) / 2.0
        horizontal_angle = (180 - fov) / 2.0
        horizontal_length = (
            2 * radius * np.sin(np.radians(fov / 2.0))
            / np.sin(np.radians(horizontal_angle))
        )
        horizontal_interval = horizontal_length / (width - 1)
        vertical_fov = height / width * fov
        vertical_angle = (180 - vertical_fov) / 2.0
        vertical_length = (
            2 * radius * np.sin(np.radians(vertical_fov / 2.0))
            / np.sin(np.radians(vertical_angle))
        )
        vertical_interval = vertical_length / (height - 1)

        x_map = np.full((height, width), radius, dtype=np.float32)
        y_map = np.tile(
            (np.arange(width) - center_x) * horizontal_interval, (height, 1)
        )
        z_map = -np.tile(
            (np.arange(height) - center_y) * vertical_interval, (width, 1)
        ).T
        distance = np.sqrt(x_map**2 + y_map**2 + z_map**2)
        xyz = np.stack(
            (
                radius / distance * x_map,
                radius / distance * y_map,
                radius / distance * z_map,
            ),
            axis=-1,
        )
        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        z_axis = np.array([0.0, 0.0, 1.0], np.float32)
        rotation_1 = cv2.Rodrigues(z_axis * np.radians(theta))[0]
        rotation_2 = cv2.Rodrigues(
            np.dot(rotation_1, y_axis) * np.radians(-phi)
        )[0]
        xyz = (rotation_2 @ rotation_1 @ xyz.reshape(-1, 3).T).T
        latitude = -np.arcsin(xyz[:, 2] / radius).reshape(height, width)
        longitude = np.arctan2(xyz[:, 1], xyz[:, 0]).reshape(height, width)
        map_x = (
            longitude / np.pi * ((self.width - 1) / 2)
            + (self.width - 1) / 2
        )
        map_y = (
            latitude / (np.pi / 2) * ((self.height - 1) / 2)
            + (self.height - 1) / 2
        )
        return cv2.remap(
            self.image,
            map_x.astype(np.float32),
            map_y.astype(np.float32),
            cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_WRAP,
        )


def load_graph(nodes_path, links_path):
    yaw = {}
    with Path(nodes_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            pano_id, heading, _, _ = line.strip().split(",")
            yaw[pano_id] = int(float(heading))
    links = {}
    with Path(links_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            pano_id, heading, _ = line.strip().split(",")
            links.setdefault(pano_id, []).append(int(round(float(heading))))
    return yaw, links


def project_directory(
    panorama_dir,
    nodes_path,
    links_path,
    output_dir,
    fov=120,
    size=1500,
    selected_nodes=None,
):
    panorama_dir, output_dir = Path(panorama_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    yaw, links = load_graph(nodes_path, links_path)
    selected = set(selected_nodes) if selected_nodes else None
    panorama_paths = sorted(panorama_dir.glob("*.jpg"))
    written = 0
    for panorama_path in tqdm(panorama_paths, desc="Projecting panoramas"):
        pano_id = panorama_path.stem
        if selected is not None and pano_id not in selected:
            continue
        if pano_id not in yaw or pano_id not in links:
            continue
        projector = Equirectangular(panorama_path)
        views = {
            "forward": projector.perspective(fov, 0, 0, size, size),
            "right": projector.perspective(fov, 90, 0, size, size),
            "backward": projector.perspective(fov, 180, 0, size, size),
            "left": projector.perspective(fov, 270, 0, size, size),
        }
        for heading in links[pano_id]:
            direction = classify_direction(relative_angle(yaw[pano_id], heading))
            output_path = output_dir / f"{pano_id}_{heading}.jpg"
            if not cv2.imwrite(str(output_path), views[direction]):
                raise OSError(f"Failed to write projected view: {output_path}")
            written += 1
    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panorama-dir", required=True)
    parser.add_argument("--nodes", required=True)
    parser.add_argument("--links", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fov", type=float, default=120)
    parser.add_argument("--size", type=int, default=1500)
    args = parser.parse_args()
    count = project_directory(
        args.panorama_dir,
        args.nodes,
        args.links,
        args.output_dir,
        fov=args.fov,
        size=args.size,
    )
    print(f"Wrote {count} projected views.")


if __name__ == "__main__":
    main()
