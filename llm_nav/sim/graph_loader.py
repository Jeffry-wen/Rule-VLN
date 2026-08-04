"""Touchdown graph structures."""

from __future__ import annotations

from collections import deque

import networkx as nx


class Node:
    def __init__(self, panoid, pano_yaw_angle, lat, lng):
        self.panoid = panoid
        self.pano_yaw_angle = pano_yaw_angle
        self.neighbors = {}
        self.coordinate = (lat, lng)

    def get_neighbor_heading(self, panoid):
        for heading, neighbor in self.neighbors.items():
            if neighbor.panoid == panoid:
                return heading
        raise ValueError(f'Neighbor "{panoid}" not found for "{self.panoid}"')


class Graph:
    def __init__(self):
        self.nodes = {}
        self.nx_graph = None

    def add_node(self, panoid, pano_yaw_angle, lat, lng):
        self.nodes[panoid] = Node(panoid, pano_yaw_angle, lat, lng)

    def add_edge(self, start_panoid, end_panoid, heading):
        self.nodes[start_panoid].neighbors[heading] = self.nodes[end_panoid]

    def remove_edge(self, start_panoid, end_panoid):
        node = self.nodes[start_panoid]
        for heading, neighbor in list(node.neighbors.items()):
            if neighbor.panoid == end_panoid:
                del node.neighbors[heading]
                return

    def get_num_neighbors(self, panoid):
        return len(self.nodes[panoid].neighbors)

    def get_target_neighbors(self, panoid):
        return list(nx.all_neighbors(self.nx_graph, panoid))

    def get_shortest_path_length(self, start, end):
        return nx.dijkstra_path_length(self.nx_graph, start, end)

    def get_shortest_path(self, start, end):
        return nx.dijkstra_path(self.nx_graph, start, end)

    def shortest_path(self, start, end):
        if start not in self.nodes or end not in self.nodes:
            raise ValueError("Start or end panorama is absent from the graph")
        queue = deque([[start]])
        visited = set()
        while queue:
            path = queue.popleft()
            current = path[-1]
            if current == end:
                return path
            if current in visited:
                continue
            visited.add(current)
            for neighbor in self.nodes[current].neighbors.values():
                if neighbor.panoid not in visited:
                    queue.append(path + [neighbor.panoid])
        return []


class GraphLoader:
    def __init__(self, opts):
        self.node_file = f"{opts.dataset}/graph/nodes.txt"
        self.link_file = f"{opts.dataset}/graph/links.txt"

    def construct_graph(self):
        graph = Graph()
        with open(self.node_file, encoding="utf-8") as handle:
            for line in handle:
                panoid, yaw, lat, lng = line.strip().split(",")
                graph.add_node(panoid, int(float(yaw)), float(lat), float(lng))
        with open(self.link_file, encoding="utf-8") as handle:
            for line in handle:
                start, heading, end = line.strip().split(",")
                graph.add_edge(start, end, float(heading))
        graph.nx_graph = nx.DiGraph(
            (node.panoid, neighbor.panoid)
            for node in graph.nodes.values()
            for neighbor in node.neighbors.values()
        )
        return graph
