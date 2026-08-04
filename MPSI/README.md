# MPSI

MPSI (Mask-Prioritized Semantic Injection) constructs the rule-constrained
Touchdown environments used by Rule-VLN. It is packaged independently from the
main FLAME/SNA environment.

## Pipeline

For each benchmark level, MPSI:

1. Loads or computes graph-importance scores.
2. Splits scored route nodes into four quartiles.
3. Computes the physical action space at each selected node.
4. Selects a compatible CityNav-Rules example.
5. Uses SegFormer to locate the road and the insertion shape.
6. Applies geometry, brightness matching, and seeded image noise.
7. Uses Insert-Anything with FLUX Fill/Redux to synthesize the sign.
8. Scores generated images with CLIP and applies a two-component GMM filter.
9. Writes inserted panoramas, a task JSON, and a generation manifest.

Level 1 contains the lowest-scoring quartile and Level 4 contains the
highest-scoring quartile.

## Installation

From the Rule-VLN repository root:

```bash
conda env create -f environment-mpsi.yml
conda activate rule_vln_mpsi
pip install -e ./MPSI
```

Nunchaku may require a CUDA-specific wheel. If the generic `nunchaku`
installation fails, install the wheel recommended by the Nunchaku project for
your PyTorch and CUDA versions.

## Required assets

Edit `MPSI/configs/touchdown.yaml` when paths differ.

- Official StreetLearn/Touchdown panoramas.
- Rule-VLN `nodes.txt`, `links.txt`, and Touchdown route JSON.
- CityNav-Rules source images.
- SegFormer Cityscapes model.
- FLUX.1 Fill, FLUX.1 Redux, and the Nunchaku transformer.
- Insert-Anything LoRA.
- A CLIP model.

The default configuration uses local copies of the released node score file,
rule labels, and five shape templates under `MPSI/assets/`.

FLUX weights are not included. Accept the applicable model licenses and obtain
the weights from their official sources.

## Compute node scores

The released score file can be reused. To recompute it from the graph and test
routes:

```bash
python -m mpsi.cli score \
  --config MPSI/configs/touchdown.yaml \
  --output outputs/mpsi/test_node_scores.txt
```

Set `score_file` in the YAML to the new path before generation.

## Generate a benchmark level

```bash
python -m mpsi.cli build \
  --config MPSI/configs/touchdown.yaml \
  --level 1 \
  --output-dir outputs/mpsi/task1
```

Use `--limit 1 --skip-quality-filter` for a one-image smoke test.

```text
outputs/mpsi/task1/
├── inserted_panoramas/
├── insert_task1_dataset.json
├── generation_manifest.jsonl
└── failures.json
```

The node seed is derived from the global seed, level, and node ID. This makes a
single cleaned MPSI run repeatable when the same model revisions, hardware, and
software stack are used. It does not make the output pixel-identical to the
legacy MPSI images used to build the published inserted feature LMDB.

## Optional InSPyReNet masking

The default pipeline uses the entire rule image as the reference mask, matching
the active setting in the original generation code. To use InSPyReNet:

1. Set `use_background_segmentation: true`.
2. Set `inspyrenet_config` and `inspyrenet_checkpoint`.
3. Ensure the optional dependencies and checkpoint are available.

The preserved inference subset and its license are under
`MPSI/third_party/inspyrenet/`.

## Third-party code

- `third_party/insert_anything/`: minimal Insert-Anything inference subset.
- `third_party/inspyrenet/`: optional minimal InSPyReNet inference subset.

Their upstream licenses are preserved in the corresponding directories.
