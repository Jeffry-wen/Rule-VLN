"""Graph navigator with temporary rule-constrained edges."""

from __future__ import annotations

import json
from pathlib import Path

from llm_nav.sim.graph_loader import GraphLoader


def get_relative_angle(current_heading, heading):
    angle = heading - current_heading
    if angle > 180:
        angle -= 360
    if angle <= -180:
        angle += 360
    return angle


def get_closest_heading(heading, headings):
    if not headings:
        return heading
    return min(headings, key=lambda item: 180 - abs(abs(heading - item) - 180))


class BaseNavigator:
    """Navigate one Touchdown graph episode and track rule violations."""

    action_list = ["forward", "left", "right", "stop", "turn_around"]
    action_mapping = {
        "turnaround": "turn_around",
        "turn around": "turn_around",
        "turn_left": "left",
        "turn left": "left",
        "turn_right": "right",
        "turn right": "right",
    }

    def __init__(self, opts):
        self.graph = GraphLoader(opts).construct_graph()
        self.states = []
        self.actions = []
        self.pano_path = []
        manifest = getattr(opts, "insert_task_dataset", None)
        self.insert_data = (
            json.loads(Path(manifest).read_text(encoding="utf-8"))
            if manifest
            else None
        )
        self.graph_change = True
        self.error_action = 0
        self.total_rule_influence = 0

    def init_state(self, pano_id, heading):
        self.states = []
        self.actions = ["init"]
        self.pano_path = []
        self.error_action = 0
        self.total_rule_influence = 0
        self._set_state(pano_id, heading)

    def _set_state(self, pano_id, heading):
        previous, _ = self.get_state()
        if pano_id != previous:
            self.pano_path.append(pano_id)
        self.states.append((pano_id, round(heading)))

    def get_state(self):
        return self.states[-1] if self.states else (None, None)

    def get_prev_state(self):
        return self.states[-2] if len(self.states) >= 2 else (None, None)

    def validate_action(self, action):
        action = self.action_mapping.get(action, action)
        current, _ = self.get_state()
        neighbor_count = self.graph.get_num_neighbors(current)
        if neighbor_count <= 1 and action not in {"stop", "turn_around"}:
            return "stop"
        if neighbor_count == 2 and action in {"left", "right"}:
            return "forward"
        if action not in self.action_list:
            raise ValueError(f"Unsupported navigation action: {action}")
        return action

    def step(self, action, step_model=False):
        if action == "init":
            return
        if action not in self.action_list:
            raise ValueError(f"Unsupported navigation action: {action}")
        next_pano, next_heading = self.get_next_state(action, step_model=step_model)
        self.actions.append(action)
        self._set_state(next_pano, next_heading)

    @staticmethod
    def get_rule_directions(action_codes):
        if action_codes == [0]:
            return None
        mapping = {
            1: "forward",
            2: "left",
            3: "right",
            4: "turn_around",
        }
        return [mapping[code] for code in action_codes if code in mapping]

    def _get_nearest_heading(self, state, node, action):
        _, current_heading = state
        headings = [float(heading) for heading in node.neighbors]
        if not headings:
            return current_heading
        ordered = sorted(
            headings, key=lambda heading: get_relative_angle(current_heading, heading)
        )
        count = len(ordered)
        if count % 2 == 0:
            forward = get_closest_heading(
                current_heading, [ordered[count // 2 - 1], ordered[count // 2]]
            )
        else:
            forward = ordered[count // 2]
        if count == 3 and action == "left":
            return ordered[0]
        if count == 3 and action == "right":
            return ordered[-1]
        if action == "forward":
            return forward
        if action == "turn_around":
            return get_closest_heading((current_heading + 180) % 360, headings)
        candidates = set(headings) - {forward}
        if not candidates:
            return forward
        if action == "left":
            return min(candidates, key=lambda heading: (forward - heading) % 360)
        if action == "right":
            return min(candidates, key=lambda heading: (heading - forward) % 360)
        return current_heading

    def _rule_filter(self, current, current_heading, action, step_model):
        if (
            step_model
            or not self.insert_data
            or current not in self.insert_data
        ):
            return []
        node = self.graph.nodes[current]
        directions = self.get_rule_directions(
            self.insert_data[current].get("action_node", [])
        )
        if not directions:
            return []
        self.total_rule_influence += 1
        allowed_nodes = set()
        base_state = (current, float(node.pano_yaw_angle))
        for direction in directions:
            heading = self._get_nearest_heading(base_state, node, direction)
            if heading in node.neighbors:
                allowed_nodes.add(node.neighbors[heading].panoid)

        intended_node = None
        if action == "forward":
            headings = list(node.neighbors)
            intended_heading = (
                get_closest_heading(current_heading, headings)
                if len(headings) <= 2
                else self._get_next_heading_intersection(action)
            )
            if intended_heading in node.neighbors:
                intended_node = node.neighbors[intended_heading].panoid
        if action not in self.action_list or (
            action == "forward"
            and intended_node is not None
            and intended_node not in allowed_nodes
        ):
            self.error_action += 1

        removed = []
        for heading, neighbor in list(node.neighbors.items()):
            if neighbor.panoid not in allowed_nodes:
                removed.append((current, neighbor.panoid, heading))
                if self.graph_change:
                    self.graph.remove_edge(current, neighbor.panoid)
        return removed

    def _restore_edges(self, removed):
        for start, end, heading in removed:
            self.graph.add_edge(start, end, heading)

    def get_next_state(self, action, step_model=False, ass_nav=None):
        del ass_nav
        current, current_heading = self.get_state()
        node = self.graph.nodes[current]
        original_headings = list(node.neighbors)
        removed = self._rule_filter(current, current_heading, action, step_model)
        try:
            neighbors = node.neighbors
            headings = list(neighbors)
            if action == "stop":
                return current, current_heading
            if action == "turn_around":
                return current, get_closest_heading(
                    (current_heading - 180) % 360, original_headings
                )
            if not headings:
                return current, current_heading
            if action == "forward" and current_heading in neighbors:
                return neighbors[current_heading].panoid, current_heading
            next_heading = (
                get_closest_heading(current_heading, headings)
                if len(headings) <= 2
                else self._get_next_heading_intersection(action)
            )
            if action == "forward" and next_heading in neighbors:
                return neighbors[next_heading].panoid, next_heading
            return current, next_heading
        finally:
            self._restore_edges(removed)

    def _get_next_heading_intersection(self, action):
        current, current_heading = self.get_state()
        previous, _ = self.get_prev_state()
        node = self.graph.nodes[current]
        headings = list(node.neighbors)
        if not headings:
            return current_heading
        forward = (
            current_heading
            if current_heading in node.neighbors
            else get_closest_heading(current_heading, headings)
        )
        if current != previous and previous is not None:
            ordered = sorted(
                headings,
                key=lambda heading: get_relative_angle(current_heading, heading),
            )
            count = len(ordered)
            if count % 2 == 0:
                forward = get_closest_heading(
                    current_heading,
                    [ordered[count // 2 - 1], ordered[count // 2]],
                )
            else:
                forward = ordered[count // 2]
            if len(headings) == 3 and action == "left":
                return ordered[0]
            if len(headings) == 3 and action == "right":
                return ordered[-1]
        if action == "forward":
            return forward
        candidates = set(headings) - {forward}
        if not candidates:
            return forward
        if action == "left":
            return min(candidates, key=lambda heading: (forward - heading) % 360)
        if action == "right":
            return min(candidates, key=lambda heading: (heading - forward) % 360)
        return forward

    def get_available_actions(self):
        current, _ = self.get_state()
        node = self.graph.nodes[current]
        if len(node.neighbors) == 1:
            return ["forward"]
        _, forward = self.get_next_state("forward")
        _, left = self.get_next_state("left")
        _, right = self.get_next_state("right")
        if left == right:
            return ["forward"]
        actions = {"forward"}
        if left != forward:
            actions.add("left")
        if right != forward:
            actions.add("right")
        return sorted(actions)
