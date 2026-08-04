"""Convert official StreetLearn metadata into the FLAME graph format."""

from __future__ import annotations

import csv
import math
from pathlib import Path


def initial_bearing(lat1, lon1, lat2, lon2):
    lat1, lat2 = math.radians(lat1), math.radians(lat2)
    delta_lon = math.radians(lon2 - lon1)
    x = math.sin(delta_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(
        lat2
    ) * math.cos(delta_lon)
    return int(round(math.degrees(math.atan2(x, y)) % 360))


def read_nodes(metadata_path):
    nodes = {}
    with Path(metadata_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"pano_id", "coords.lat", "coords.lng", "heading_deg"}
        if not required.issubset(reader.fieldnames or []):
            missing = required - set(reader.fieldnames or [])
            raise ValueError(f"Missing node metadata columns: {sorted(missing)}")
        for row in reader:
            nodes[row["pano_id"]] = {
                "heading": int(round(float(row["heading_deg"]))),
                "lat": float(row["coords.lat"]),
                "lng": float(row["coords.lng"]),
            }
    return nodes


def convert_graph(node_metadata, link_metadata, output_dir):
    nodes = read_nodes(node_metadata)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    node_path, link_path = output / "nodes.txt", output / "links.txt"
    with node_path.open("w", encoding="utf-8") as handle:
        for pano_id, node in nodes.items():
            handle.write(
                f"{pano_id},{node['heading']},{node['lat']},{node['lng']}\n"
            )

    link_count = 0
    with Path(link_metadata).open("r", encoding="utf-8") as source, link_path.open(
        "w", encoding="utf-8"
    ) as target:
        reader = csv.reader(source, delimiter="\t")
        for row in reader:
            if len(row) != 2:
                continue
            start, end = row
            if start == "pano_id" or start not in nodes or end not in nodes:
                continue
            heading = initial_bearing(
                nodes[start]["lat"],
                nodes[start]["lng"],
                nodes[end]["lat"],
                nodes[end]["lng"],
            )
            target.write(f"{start},{heading},{end}\n")
            link_count += 1
    return len(nodes), link_count
