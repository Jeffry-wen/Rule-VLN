#!/usr/bin/env python3
"""Prepare the official Touchdown graph and optional 120-degree views."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.project_panoramas import project_directory
from tools.touchdown_graph import convert_graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--streetlearn-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/touchdown"))
    parser.add_argument(
        "--project-views",
        action="store_true",
        help="Also create 120-degree views from the official panoramas.",
    )
    parser.add_argument(
        "--rebuild-graph",
        action="store_true",
        help=(
            "Reconstruct graph headings from coordinates. The bundled graph "
            "should be retained when using the released feature LMDBs."
        ),
    )
    parser.add_argument(
        "--view-output", type=Path, default=Path("data/images/touchdown_crop_120")
    )
    args = parser.parse_args()

    node_metadata = args.streetlearn_root / "manhattan_touchdown_metadata_nodes.tsv"
    link_metadata = args.streetlearn_root / "manhattan_touchdown_metadata_links.tsv"
    panorama_dir = args.streetlearn_root / "jpegs_manhattan_touchdown_2022"
    graph_dir = args.output_root / "graph"
    nodes_path, links_path = graph_dir / "nodes.txt", graph_dir / "links.txt"
    if args.rebuild_graph or not (nodes_path.exists() and links_path.exists()):
        node_count, link_count = convert_graph(node_metadata, link_metadata, graph_dir)
        print(
            f"Reconstructed graph with {node_count} nodes and "
            f"{link_count} directed links."
        )
        print(
            "Coordinate-derived headings are not guaranteed to match the "
            "released feature keys. Prefer the bundled graph for evaluation."
        )
    else:
        node_count = sum(1 for _ in nodes_path.open(encoding="utf-8"))
        link_count = sum(1 for _ in links_path.open(encoding="utf-8"))
        print(
            f"Using bundled graph with {node_count} nodes and "
            f"{link_count} directed links."
        )
    if args.project_views:
        count = project_directory(
            panorama_dir,
            nodes_path,
            links_path,
            args.view_output,
            fov=120,
            size=1500,
        )
        print(f"Wrote {count} projected 120-degree views.")


if __name__ == "__main__":
    main()
