"""Touchdown graph access and action-space reasoning."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx


def relative_angle(base, target):
    value = target - base
    if value > 180:
        value -= 360
    if value <= -180:
        value += 360
    return value


@dataclass
class Node:
    pano_id: str
    yaw: float
    lat: float
    lng: float
    neighbors: dict[float, str] = field(default_factory=dict)


class TouchdownGraph:
    def __init__(self, nodes_file, links_file):
        self.nodes: dict[str, Node] = {}
        with Path(nodes_file).open("r", encoding="utf-8") as handle:
            for line in handle:
                pano_id, yaw, lat, lng = line.strip().split(",")
                self.nodes[pano_id] = Node(
                    pano_id, float(yaw), float(lat), float(lng)
                )
        self.nx = nx.DiGraph()
        with Path(links_file).open("r", encoding="utf-8") as handle:
            for line in handle:
                start, heading, end = line.strip().split(",")
                if start in self.nodes and end in self.nodes:
                    self.nodes[start].neighbors[float(heading)] = end
                    self.nx.add_edge(start, end)

    def action_space(self, pano_id):
        node = self.nodes[pano_id]
        actions = set()
        for heading in node.neighbors:
            angle = relative_angle(node.yaw, heading)
            if -45 <= angle <= 45:
                actions.add("forward")
            elif 45 < angle <= 135:
                actions.add("right")
            elif -135 <= angle < -45:
                actions.add("left")
            else:
                actions.add("turn_around")
        actions.add("turn_around")
        codes = [
            code
            for action, code in (
                ("forward", 1),
                ("left", 2),
                ("right", 3),
                ("turn_around", 4),
            )
            if action in actions
        ]
        if set(codes) == {1, 2, 3, 4}:
            codes.append(0)
        return node.yaw, sorted(actions), codes
