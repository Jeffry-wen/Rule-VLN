from pathlib import Path

import cv2
import numpy as np

from tools.project_panoramas import (
    Equirectangular,
    classify_direction,
    relative_angle,
)
from tools.touchdown_graph import convert_graph, initial_bearing


def test_initial_bearing_cardinal_direction():
    assert initial_bearing(0, 0, 0, 1) == 90
    assert initial_bearing(0, 0, 1, 0) == 0


def test_metadata_conversion(tmp_path):
    nodes = tmp_path / "nodes.tsv"
    nodes.write_text(
        "pano_id\tcoords.lat\tcoords.lng\theading_deg\n"
        "a\t0\t0\t10\n"
        "b\t0\t1\t20\n",
        encoding="utf-8",
    )
    links = tmp_path / "links.tsv"
    links.write_text("a\tb\n", encoding="utf-8")
    count = convert_graph(nodes, links, tmp_path / "graph")
    assert count == (2, 1)
    assert (tmp_path / "graph/nodes.txt").read_text().splitlines()[0] == "a,10,0.0,0.0"
    assert (tmp_path / "graph/links.txt").read_text().strip() == "a,90,b"


def test_projection_shape_and_direction(tmp_path):
    panorama = np.zeros((64, 128, 3), dtype=np.uint8)
    panorama[:, :64] = (0, 0, 255)
    path = tmp_path / "pano.jpg"
    cv2.imwrite(str(path), panorama)
    view = Equirectangular(path).perspective(120, 0, 0, 32, 32)
    assert view.shape == (32, 32, 3)
    assert classify_direction(relative_angle(10, 100)) == "right"
