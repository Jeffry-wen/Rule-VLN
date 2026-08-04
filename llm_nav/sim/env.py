"""Touchdown navigation environment and Rule-VLN evaluation metrics."""

from __future__ import annotations

import random

import networkx as nx
from PIL import Image
from pyxdameraulevenshtein import damerau_levenshtein_distance

from llm_nav.data import FeatureDB
from llm_nav.sim.base_navigator import (
    BaseNavigator,
    get_closest_heading,
    get_relative_angle,
)
from llm_nav.sim.utils import load_datasets, load_nav_graph


class EnvBatch:
    """A small batch of graph navigators sharing feature stores."""

    def __init__(self, opts, name=None):
        self.opts = opts
        self.name = name
        self.navs = [BaseNavigator(opts) for _ in range(opts.env_batch_size)]
        self.feature_db = FeatureDB(
            default_dir=opts.img_db,
            insert_dir=getattr(opts, "img_insert_db", None),
            insert_task_dataset=getattr(opts, "insert_task_dataset", None),
            decode_type="torch",
        )
        raw_base = getattr(opts, "img_raw_db", None)
        self.raw_db = (
            FeatureDB(
                default_dir=raw_base,
                insert_dir=getattr(opts, "img_raw_insert_db", None),
                insert_task_dataset=getattr(opts, "insert_task_dataset", None),
                decode_type="pil",
                max_cache=100,
            )
            if raw_base
            else None
        )

    def newEpisodes(self, pano_ids, headings):
        for navigator, pano_id, heading in zip(self.navs, pano_ids, headings):
            navigator.init_state(pano_id, heading)

    def _get_gt_action(self, batch):
        actions = []
        for item, navigator in zip(batch, self.navs):
            route = item["route_panoids"]
            target = route[-1]
            current, current_heading = navigator.get_state()
            current_node = navigator.graph.nodes[current]
            if current in route:
                occurrences = [index for index, node in enumerate(route) if node == current]
                route_index = occurrences[min(len(occurrences), navigator.pano_path.count(current)) - 1]
                if route_index == len(route) - 1:
                    actions.append("stop")
                    continue
                next_node = route[route_index + 1]
            else:
                shortest = navigator.graph.get_shortest_path(current, target)
                if len(shortest) <= 1:
                    actions.append("stop")
                    continue
                next_node = shortest[1]
            target_heading = current_node.get_neighbor_heading(next_node)
            forward_node, forward_heading = navigator.get_next_state("forward")
            if next_node == forward_node:
                if (
                    len(current_node.neighbors) != 3
                    or abs(get_relative_angle(forward_heading, target_heading)) < 45
                ):
                    actions.append("forward")
                    continue
            _, reverse_heading = navigator.get_next_state("turn_around")
            if target_heading == reverse_heading:
                actions.append("turn_around")
                continue
            _, left_heading = navigator.get_next_state("left")
            _, right_heading = navigator.get_next_state("right")
            if target_heading == left_heading:
                actions.append("left")
            elif target_heading == right_heading:
                actions.append("right")
            else:
                closest = get_closest_heading(
                    target_heading, [left_heading, right_heading]
                )
                actions.append("left" if closest == left_heading else "right")
        return actions

    def _get_observations(self, batch):
        observations = []
        for navigator in self.navs[: len(batch)]:
            previous = navigator.states[-2] if len(navigator.states) > 1 else (None, None)
            pano_id, heading = navigator.states[-1]
            previous_pano, _ = previous
            item = {"viewpoint": f"{pano_id}_{int(heading)}"}
            neighbors = navigator.graph.get_num_neighbors(pano_id)
            if neighbors > 2 and pano_id != previous_pano:
                item["intersection"] = neighbors
            item["image"] = self.feature_db.get(item["viewpoint"])
            item["raw_image"] = self._get_raw_image(item["viewpoint"])
            observations.append(item)
        return observations

    def _get_raw_image(self, image_id: str) -> Image.Image | None:
        if self.raw_db is None:
            return None
        try:
            return self.raw_db.get(image_id)
        except FileNotFoundError:
            return None

    @staticmethod
    def _shortest_path_length(graph, start, end):
        if start == end:
            return 0.0
        try:
            return float(nx.dijkstra_path_length(graph, start, end))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return float("inf")

    @classmethod
    def _trajectory_path_length(cls, graph, trajectory):
        length = 0.0
        for start, end in zip(trajectory[:-1], trajectory[1:]):
            step_length = cls._shortest_path_length(graph, start, end)
            if step_length == float("inf"):
                return step_length
            length += step_length
        return length

    @classmethod
    def _spl(cls, graph, trajectory, reference, success):
        if not success:
            return 0.0
        shortest = cls._shortest_path_length(graph, reference[0], reference[-1])
        predicted = cls._trajectory_path_length(graph, trajectory)
        if shortest == float("inf") or predicted == float("inf"):
            return 0.0
        if shortest == 0:
            return float(predicted == 0)
        return shortest / max(shortest, predicted)

    def _eva_metrics(self, trajectories, batch, graph, metrics):
        for trajectory, item, navigator in zip(trajectories, batch, self.navs):
            reference = item["route_panoids"]
            edit_score = 1 - damerau_levenshtein_distance(
                trajectory, reference
            ) / max(len(trajectory), len(reference))
            targets = list(nx.all_neighbors(graph, reference[-1])) + [reference[-1]]
            success = int(trajectory[-1] in targets)
            metrics[0] += success
            metrics[1] += self._shortest_path_length(
                graph, trajectory[-1], reference[-1]
            )
            metrics[2] += edit_score if success else 0.0
            metrics[3] += (
                navigator.error_action / navigator.total_rule_influence
                if navigator.total_rule_influence
                else 0.0
            )
            metrics[4] += self._spl(graph, trajectory, reference, success)

    def action_step(
        self,
        actions,
        ended,
        active_count,
        trajectories,
        total_steps,
        step_model=False,
    ):
        for index, navigator in enumerate(self.navs[: len(ended)]):
            if ended[index] or actions[index] == "<detour_skip>":
                continue
            action = actions[index]
            if action == "stop":
                ended[index] = 1
                active_count[0] -= 1
            navigator.step(action, step_model)
            if navigator.states[-1][0] != navigator.states[-2][0]:
                trajectories[index].append(navigator.states[-1][0])
            if None in navigator.states[-1]:
                ended[index] = 1
                active_count[0] -= 1
            total_steps[0] += 1


class TouchdownBatch:
    """Dataset wrapper used by the FLAME rollout loop."""

    def __init__(self, opts, seed=10, splits=("train",), name=None):
        self.env = EnvBatch(opts, name)
        self.opts = opts
        self.data = [dict(item) for item in load_datasets(splits, opts)]
        self.dict_data = {
            item.get("route_id", item.get("id")): item for item in self.data
        }
        self.random = random.Random(seed)
        self.random.shuffle(self.data)
        self.ix = 0
        self.batch_size = opts.env_batch_size
        self.splits = splits
        self.graph = load_nav_graph(opts)

    def _next_minibatch(self):
        self.batch = self.data[self.ix : self.ix + self.batch_size]
        self.ix += len(self.batch)
        if self.ix >= len(self.data):
            self.ix = 0
            self.random.shuffle(self.data)

    def reset(self, print_info=False, route_ids=None):
        del print_info
        self.batch = (
            [self.dict_data[route_id] for route_id in route_ids]
            if route_ids
            else None
        )
        if self.batch is None:
            self._next_minibatch()
        pano_ids, headings, trajectories = [], [], []
        for item in self.batch:
            current, next_pano = item["route_panoids"][:2]
            pano_ids.append(current)
            trajectories.append([current])
            neighbors = self.env.navs[0].graph.nodes[current].neighbors
            if item["start_heading"] == 0:
                ground_truth = next(
                    heading
                    for heading, node in neighbors.items()
                    if node.panoid == next_pano
                )
                heading = (
                    self.random.choice(list(neighbors))
                    if self.opts.legacy_navigation_mode
                    else ground_truth
                )
            else:
                heading = item["start_heading"]
            headings.append(heading)
        self.env.newEpisodes(pano_ids, headings)
        return trajectories

    def get_gt_action(self):
        return self.env._get_gt_action(self.batch)

    def get_instructions(self):
        return [item["navigation_text"] for item in self.batch]

    def get_observations(self):
        return self.env._get_observations(self.batch)

    def reset_epoch(self):
        self.ix = 0

    def eva_metrics(self, trajectories, metrics):
        self.env._eva_metrics(trajectories, self.batch, self.graph, metrics)
