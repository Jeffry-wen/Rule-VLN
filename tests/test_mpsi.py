import json
import random
import sys
from pathlib import Path

from PIL import Image

MPSI_ROOT = Path(__file__).resolve().parents[1] / "MPSI"
sys.path.insert(0, str(MPSI_ROOT))

from mpsi.graph import TouchdownGraph
from mpsi.pipeline import level_nodes, stable_seed
from mpsi.rules import RuleCatalog


def test_level_order_and_stable_seed():
    scores = [(f"node{index}", float(8 - index)) for index in range(8)]
    assert level_nodes(scores, 4) == ["node0", "node1"]
    assert level_nodes(scores, 1) == ["node6", "node7"]
    assert stable_seed(777, 1, "node") == stable_seed(777, 1, "node")
    assert stable_seed(777, 1, "node") != stable_seed(777, 2, "node")


def test_action_space_and_rule_selection(tmp_path):
    nodes = tmp_path / "nodes.txt"
    nodes.write_text(
        "a,0,0,0\nb,0,0,1\nc,0,0,-1\n", encoding="utf-8"
    )
    links = tmp_path / "links.txt"
    links.write_text("a,0,b\na,90,c\n", encoding="utf-8")
    graph = TouchdownGraph(nodes, links)
    _, actions, codes = graph.action_space("a")
    assert "forward" in actions and "right" in actions
    assert {1, 3, 4}.issubset(codes)

    image_root = tmp_path / "images"
    (image_root / "00001").mkdir(parents=True)
    Image.new("RGB", (8, 8), "white").save(image_root / "00001/rule.png")
    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps(
            [
                {
                    "label_id": "00001",
                    "file": "00001/rule.png",
                    "label": "turn right",
                    "action_node": [3],
                }
            ]
        ),
        encoding="utf-8",
    )
    path, entry = RuleCatalog(image_root, labels).select(codes, random.Random(1))
    assert path.name == "rule.png"
    assert entry["action_node"] == [3]
