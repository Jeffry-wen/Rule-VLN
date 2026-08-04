import math
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image


def normalize_angle(angle):
    while angle > 180: angle -= 360
    while angle <= -180: angle += 360
    return angle

def get_rel_angle(base, target):
    diff = target - base
    if diff > 180: diff -= 360
    if diff <= -180: diff += 360
    return diff


class RelativeGreedyAgent:
    def __init__(self, navigator, feature_db):
        self.navigator = navigator
        self.graph = navigator.graph
        self.feature_db = feature_db

        self.virtual_x = 0.0
        self.virtual_y = 0.0
        self.current_virtual_heading = 0.0
        self.target_virtual_pos = None
        self.target_virtual_heading = 0.0

        self.visual_memory = []
        self.path_memory = []
        self.start_node_feature = None
        self.first_step_feature = None
        self.force_stop_next = False
        self.is_aligning_target = False

        self.VISUAL_SIM_THRESHOLD = 0.99
        self.MAX_DETOUR_STEPS = 200
        self.HARD_STOP_THRESHOLD = 0.0
        self.PENALTY_BACKTRACK = 100000.0

    def _get_displacement_vector(self, heading_degrees, distance=1.0):
        rad = math.radians(heading_degrees)
        dx = math.sin(rad) * distance
        dy = math.cos(rad) * distance
        return dx, dy

    def _calc_distance(self, x1, y1, x2, y2):
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def _fetch_feature_by_id(self, full_id):
        try:
            data = self.feature_db.get(full_id)
            if isinstance(data, torch.Tensor):
                return data.view(-1)
            elif isinstance(data, np.ndarray):
                return torch.from_numpy(data).float().view(-1)
            elif isinstance(data, Image.Image):
                arr = np.array(data)
                return torch.from_numpy(arr).float().view(-1)
        except Exception as e:
            pass
        return None

    def _get_panorama_feature(self, panoid):
        feats = []
        if hasattr(self.feature_db, 'prefix_index') and panoid in self.feature_db.prefix_index:
            candidates = self.feature_db.prefix_index[panoid]
            for heading_int, full_id in candidates:
                feat = self._fetch_feature_by_id(full_id)
                if feat is not None: feats.append(feat)
        if not feats:
            for h in range(0, 360, 45):
                view_id = f"{panoid}_{h}"
                feat = self._fetch_feature_by_id(view_id)
                if feat is not None: feats.append(feat)
        if not feats: return torch.zeros(1)
        feats_stack = torch.stack(feats)
        return torch.mean(feats_stack, dim=0)

    def _update_visual_memory(self, panoid):
        feat = self._get_panorama_feature(panoid)
        if feat.norm() > 0:
            self.visual_memory.append(feat)

    def _resolve_action_consistent(self, curr_heading, target_heading, all_neighbor_headings):
        curr_h = float(curr_heading)
        target_h = float(target_heading)

        if len(all_neighbor_headings) <= 2:
            diff = get_rel_angle(curr_h, target_h)
            if abs(diff) > 135:
                return 'turn_around'
            return 'forward'

        out_headings = [float(h) for h in all_neighbor_headings]
        out_headings_sorted = sorted(out_headings, key=lambda h: get_rel_angle(curr_h, h))
        n = len(out_headings_sorted)

        forward_heading = curr_h
        if n > 0:
            if n % 2 == 0:
                h1, h2 = out_headings_sorted[n // 2 - 1], out_headings_sorted[n // 2]
                diff1, diff2 = abs(get_rel_angle(curr_h, h1)), abs(get_rel_angle(curr_h, h2))
                forward_heading = h1 if diff1 < diff2 else h2
            else:
                forward_heading = out_headings_sorted[n // 2]

        if target_h == forward_heading: return 'forward'

        if n == 3:
            if target_h == out_headings_sorted[0]: return 'left'
            if target_h == out_headings_sorted[-1]: return 'right'

        candidates = set(out_headings) - {forward_heading}
        if not candidates: return 'forward'

        pred_right = min(candidates, key=lambda h: (h - forward_heading) % 360)
        pred_left = min(candidates, key=lambda h: (forward_heading - h) % 360)

        if target_h == pred_left:
            return 'left'
        elif target_h == pred_right:
            return 'right'

        if abs(get_rel_angle(curr_h, target_h)) > 135: return 'turn_around'

        return 'forward'

    def get_next_action(self, curr_panoid, curr_abs_heading):
        if self.force_stop_next:
            return None

        self._update_visual_memory(curr_panoid)

        if self.start_node_feature is not None and len(self.path_memory) > 1:
            curr_feat = self.visual_memory[-1]
            if curr_feat.device != self.start_node_feature.device:
                curr_feat = curr_feat.to(self.start_node_feature.device)
            sim_current_to_start = F.cosine_similarity(curr_feat.unsqueeze(0),
                                                       self.start_node_feature.unsqueeze(0)).item()

            if sim_current_to_start > self.VISUAL_SIM_THRESHOLD:
                return 'LOOP_STOP'

        if self.first_step_feature is not None and len(self.path_memory) > 2:
            curr_feat = self.visual_memory[-1]
            if curr_feat.device != self.first_step_feature.device:
                curr_feat = curr_feat.to(self.first_step_feature.device)

            sim_current_to_first = F.cosine_similarity(curr_feat.unsqueeze(0),
                                                       self.first_step_feature.unsqueeze(0)).item()

            if sim_current_to_first > self.VISUAL_SIM_THRESHOLD:
                return 'BACKTRACK_STOP'

        current_dist = self._calc_distance(self.virtual_x, self.virtual_y, self.target_virtual_pos[0],
                                           self.target_virtual_pos[1])

        if current_dist < self.HARD_STOP_THRESHOLD:
            self.is_aligning_target = False
            return None

        self.is_aligning_target = False
        if len(self.path_memory) > self.MAX_DETOUR_STEPS: return None

        should_add = False
        if not self.path_memory:
            should_add = True
        else:
            last_x, last_y = self.path_memory[-1]['x'], self.path_memory[-1]['y']
            if self._calc_distance(self.virtual_x, self.virtual_y, last_x, last_y) > 0.1:
                should_add = True
        if should_add:
            self.path_memory.append({'id': curr_panoid, 'x': self.virtual_x, 'y': self.virtual_y})

        last_visited_pos = self.path_memory[-2] if len(self.path_memory) >= 2 else None
        curr_node = self.graph.nodes.get(curr_panoid)
        neighbors = curr_node.neighbors
        all_neighbor_headings = list(neighbors.keys())
        candidates = []


        for abs_h_str, nb_node in neighbors.items():
            abs_h = float(abs_h_str)
            real_turn = normalize_angle(abs_h - curr_abs_heading)
            pred_virtual_heading = normalize_angle(self.current_virtual_heading + real_turn)
            dx, dy = self._get_displacement_vector(pred_virtual_heading, 1.0)
            pred_next_x, pred_next_y = self.virtual_x + dx, self.virtual_y + dy

            future_dist = self._calc_distance(pred_next_x, pred_next_y, self.target_virtual_pos[0],
                                              self.target_virtual_pos[1])

            action_name = self._resolve_action_consistent(curr_abs_heading, abs_h, all_neighbor_headings)

            penalty = 0.0
            if last_visited_pos is not None:
                if self._calc_distance(pred_next_x, pred_next_y, last_visited_pos['x'], last_visited_pos['y']) < 0.5:
                    penalty += self.PENALTY_BACKTRACK

            total_score = future_dist + penalty

            candidates.append({
                'action': action_name,
                'heading': abs_h,
                'dist': future_dist,
                'penalty': penalty,
                'score': total_score,
                'next_pos': (pred_next_x, pred_next_y)
            })

        candidates.sort(key=lambda x: x['score'])


        if len(candidates) <= 1:
            return 'DEAD_END_FAILURE'

        if candidates[0]['score'] >= self.PENALTY_BACKTRACK:
            return None

        best_action = candidates[0]['action']
        return best_action

    def set_goal_by_expectation(self, start_node_id, start_abs_heading, action_expected):
        rel_angle = {'left': -90, 'right': 90, 'turn_around': 180}.get(action_expected, 0)
        target_heading = normalize_angle(0 + rel_angle)
        dx, dy = self._get_displacement_vector(target_heading, distance=1.0)

        self.virtual_x, self.virtual_y = 0.0, 0.0
        self.current_virtual_heading = 0.0
        self.target_virtual_pos = (dx, dy)
        self.target_virtual_heading = target_heading
        self.is_aligning_target = False

        self.visual_memory.clear()
        self._update_visual_memory(start_node_id)
        if len(self.visual_memory) > 0:
            self.start_node_feature = self.visual_memory[0].clone()
        else:
            self.start_node_feature = None

        self.path_memory = [{'id': start_node_id, 'x': 0.0, 'y': 0.0}]

    def update_virtual_state_by_heading(self, current_real_heading, previous_real_heading):
        diff = normalize_angle(current_real_heading - previous_real_heading)
        self.current_virtual_heading = normalize_angle(self.current_virtual_heading + diff)
        dx, dy = self._get_displacement_vector(self.current_virtual_heading, distance=1.0)
        self.virtual_x += dx
        self.virtual_y += dy

    def _simulate_graph_traversal(self, curr_node_id, curr_heading, action):
        curr_node_obj = self.graph.nodes.get(curr_node_id)
        if not curr_node_obj:
            return curr_node_id, curr_heading

        neighbors = curr_node_obj.neighbors
        all_neighbor_headings = list(neighbors.keys())

        candidates = []

        for h_str, nb_node in neighbors.items():
            h_val = float(h_str)
            predicted_action = self._resolve_action_consistent(curr_heading, h_val, all_neighbor_headings)

            if predicted_action == action:
                candidates.append((h_val, nb_node.panoid))

        if candidates:

            action_rel_map = {'forward': 0, 'left': -90, 'right': 90, 'turn_around': 180}
            target_rel = action_rel_map.get(action, 0)

            ideal_heading = normalize_angle(curr_heading + target_rel)

            best_match = min(candidates, key=lambda x: abs(get_rel_angle(ideal_heading, x[0])))

            best_match_heading, best_match_node = best_match

            return best_match_node, best_match_heading


        return None, None
    def generate_detour_sequence(self, start_node, start_heading, forced_action_actual, proposed_action):
        if proposed_action in ['left', 'right', 'turn_around']:
            fallback_result = [proposed_action, "forward"]
        else:
            fallback_result = [proposed_action]

        stop_flag = "fail_detour"

        full_action_plan = []
        full_action_plan.append(forced_action_actual)

        if forced_action_actual in ['left', 'right', 'turn_around']:
            full_action_plan.append("forward")

        prev_heading = start_heading

        self.first_step_feature = None

        sim_node, sim_heading = self._simulate_graph_traversal(start_node, start_heading, forced_action_actual)
        if sim_node is None:
            return fallback_result

        feat = self._get_panorama_feature(sim_node)
        if feat.norm() > 0:
            self.first_step_feature = feat.clone()

        self.update_virtual_state_by_heading(sim_heading, prev_heading)

        curr_is_aligning = self.is_aligning_target

        for step in range(self.MAX_DETOUR_STEPS):
            next_action = self.get_next_action(sim_node, sim_heading)
            if next_action == 'DEAD_END_FAILURE':
                return fallback_result

            if next_action == 'BACKTRACK_STOP':
                return fallback_result

            if next_action is None:
                full_action_plan.append('turn_around')
                return full_action_plan

            if next_action == 'LOOP_STOP':
                if full_action_plan and full_action_plan[-1] == 'forward':
                    full_action_plan.pop()
                else:
                    full_action_plan.pop()
                full_action_plan.append('turn_around')
                return full_action_plan

            curr_is_aligning = self.is_aligning_target
            full_action_plan.append(next_action)

            if next_action in ['left', 'right', 'turn_around'] and not curr_is_aligning:
                full_action_plan.append("forward")

            prev_heading = sim_heading

            sim_node, sim_heading = self._simulate_graph_traversal(sim_node, sim_heading, next_action)

            if sim_node is None:
                return fallback_result


            self.update_virtual_state_by_heading(sim_heading, prev_heading)
        return fallback_result



