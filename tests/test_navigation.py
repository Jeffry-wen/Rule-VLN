import json
from types import SimpleNamespace

from llm_nav.agent import FLAME
from llm_nav.sim.base_navigator import BaseNavigator


def make_dataset(tmp_path):
    graph = tmp_path / "graph"
    graph.mkdir()
    (graph / "nodes.txt").write_text(
        "a,0,0,0\nb,0,0,1\nc,0,0,-1\n", encoding="utf-8"
    )
    (graph / "links.txt").write_text(
        "a,0,b\na,180,c\nb,180,a\nc,0,a\n", encoding="utf-8"
    )
    manifest = tmp_path / "task.json"
    manifest.write_text(
        json.dumps(
            {
                "a": {
                    "rule_img": "rule.png",
                    "action_node": [4],
                }
            }
        ),
        encoding="utf-8",
    )
    return SimpleNamespace(dataset=str(tmp_path), insert_task_dataset=str(manifest))


def test_rule_violation_and_edge_restoration(tmp_path):
    navigator = BaseNavigator(make_dataset(tmp_path))
    navigator.init_state("a", 0)
    next_pano, _ = navigator.get_next_state("forward")
    assert next_pano == "c"
    assert navigator.error_action == 1
    assert navigator.total_rule_influence == 1
    assert len(navigator.graph.nodes["a"].neighbors) == 2


def test_flame_action_parser():
    flame = object.__new__(FLAME)
    assert flame._get_action("The answer is Turn Around.") == "turn_around"
    assert flame._get_action("Forward.") == "forward"
    assert flame._get_action("Stop.") == "stop"
