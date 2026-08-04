"""Graph-importance scoring used to define Rule-VLN levels."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
from tqdm import tqdm


def load_route_nodes(route_data):
    with Path(route_data).open("r", encoding="utf-8") as handle:
        routes = json.load(handle)
    frequency = defaultdict(int)
    normalized_routes = []
    for route in routes:
        images = route.get("node", route.get("images", []))
        nodes = [image.rsplit("_", 1)[0] for image in images]
        normalized_routes.append(nodes)
        for node in nodes:
            frequency[node] += 1
    return normalized_routes, frequency


def path_dependency(graph, routes):
    values = defaultdict(list)
    for route in tqdm(routes, desc="Computing path dependency"):
        if len(route) < 3 or route[0] not in graph or route[-1] not in graph:
            continue
        try:
            original = nx.shortest_path_length(graph, route[0], route[-1])
        except nx.NetworkXNoPath:
            continue
        for node in route[1:-1]:
            if node not in graph:
                continue
            reduced = graph.copy()
            reduced.remove_node(node)
            try:
                new_length = nx.shortest_path_length(reduced, route[0], route[-1])
                values[node].append(new_length - original)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                values[node].append(float("inf"))
    return {
        node: (
            sum(finite) / len(finite)
            if (finite := [value for value in scores if np.isfinite(value)])
            else float("inf")
        )
        for node, scores in values.items()
    }


def normalized(values, nodes, infinite_value=0.0):
    raw = np.array(
        [
            values.get(node, 0.0)
            if np.isfinite(values.get(node, 0.0))
            else infinite_value
            for node in nodes
        ],
        dtype=np.float64,
    )
    minimum, maximum = raw.min(), raw.max()
    if maximum == minimum:
        return np.zeros_like(raw)
    return (raw - minimum) / (maximum - minimum)


def compute_scores(graph, route_data):
    routes, frequency = load_route_nodes(route_data)
    route_nodes = {node for route in routes for node in route}
    undirected = graph.to_undirected()
    degree = nx.degree_centrality(undirected)
    betweenness = nx.betweenness_centrality(undirected, normalized=True)
    dependency = path_dependency(undirected, routes)
    nodes = sorted(route_nodes & set(graph.nodes))
    degree_norm = normalized(degree, nodes)
    betweenness_norm = normalized(betweenness, nodes)
    dependency_values = {
        node: -value if np.isfinite(value) else -1e9
        for node, value in dependency.items()
    }
    dependency_norm = normalized(dependency_values, nodes, infinite_value=-1e9)
    frequency_norm = normalized(frequency, nodes)
    return {
        node: {
            "degree": float(degree.get(node, 0.0)),
            "betweenness": float(betweenness.get(node, 0.0)),
            "path_dependency": float(dependency.get(node, 0.0)),
            "path_frequency": float(frequency.get(node, 0.0)),
            "score": float(
                (
                    degree_norm[index]
                    + betweenness_norm[index]
                    + dependency_norm[index]
                    + frequency_norm[index]
                )
                / 4
            ),
        }
        for index, node in enumerate(nodes)
    }


def write_scores(scores, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for node in sorted(scores, key=lambda item: scores[item]["score"], reverse=True):
            value = scores[node]
            handle.write(
                f"{node}\tdegree: {value['degree']:.6f}\t"
                f"betweenness: {value['betweenness']:.6f}\t"
                f"path_dependency: {value['path_dependency']:.6f}\t"
                f"path_frequency: {value['path_frequency']:.6f}\t"
                f"score: {value['score']:.6f}\n"
            )
