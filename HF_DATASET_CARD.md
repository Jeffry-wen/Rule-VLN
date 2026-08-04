---
pretty_name: Rule-VLN
language:
  - en
license: other
tags:
  - vision-language-navigation
  - embodied-ai
  - rule-compliance
  - urban-navigation
  - touchdown
  - streetlearn
  - traffic-signs
  - image-classification
  - lmdb
task_categories:
  - text-generation
  - visual-question-answering
  - image-to-text
  - image-classification
---

# Rule-VLN Dataset

Rule-VLN is a rule-compliant outdoor vision-and-language navigation benchmark built on the Touchdown / StreetLearn urban navigation environment. It studies whether navigation agents can follow language instructions while also complying with semantic traffic rules, such as regulatory signs that prohibit otherwise reachable movements.

This dataset accompanies the paper:

**Rule-VLN: Bridging Perception and Compliance via Semantic Reasoning and Geometric Rectification**  
Jiawen Wen, Penglei Sun, Wenjie Zhang, Suixuan Qiu, Weisheng Xu, Xiaofei Yang, Xiaowen Chu

- Paper: https://arxiv.org/abs/2604.16993
- Project page: https://jeffry-wen.github.io/Rule-VLN/
- Code: https://github.com/Jeffry-wen/Rule-VLN

## Dataset Summary

Rule-VLN introduces semantic rule constraints into Touchdown-style street-level navigation. A route can be geometrically reachable but invalid under a visible traffic rule. The benchmark therefore evaluates both navigation ability and rule compliance.

The released assets include:

- Touchdown navigation splits and FLAME-format conversation data.
- Four curriculum levels of inserted rule tasks.
- The semantic rule image dataset used by the MPSI insertion pipeline.
- Precomputed FLAME visual feature LMDBs for original Touchdown observations.
- Precomputed FLAME visual feature LMDBs for rule-inserted observations.
- A released fine-tuned checkpoint used for evaluation.

Raw StreetLearn / Touchdown panoramic images are **not redistributed** in this dataset. Users who need to reconstruct raw image LMDBs or regenerate the MPSI insertion pipeline must obtain the official panoramas from the StreetLearn dataset provider and follow their license and usage terms.

## Dataset Contents

The dataset keeps the historical release layout used by the Rule-VLN codebase:

```text
Rule-VLN-Envirement/
|-- flame_clip_touchdown_120_crop_lmdb/
|-- flame_clip_touchdown_insert_120_crop_lmdb/
`-- insert_touchdown/

Rule-VLN-Environment/
`-- touchdown/
    |-- data/
    `-- ft_data/

City_Navigation_Rules_Dataset/
|-- label.json
`-- img/
    |-- 00000/
    |-- 00001/
    `-- ...

Finetune-ckpt/
`-- Flame_checkpoint-2500/
```

When downloaded with the official script, these assets are normalized into the following local paths:

```text
data/features/flame_clip_touchdown_120_crop_lmdb/
data/features/flame_clip_touchdown_insert_120_crop_lmdb/
data/touchdown/insert_touchdown/
data/touchdown/data/
data/touchdown/ft_data/
data/citynav_rules/
checkpoints/Flame_checkpoint-2500/
```

### Asset Description

| Asset | Local path | Purpose |
| --- | --- | --- |
| Base visual features | `data/features/flame_clip_touchdown_120_crop_lmdb/` | Precomputed FLAME features for original 120-degree Touchdown views. |
| Inserted visual features | `data/features/flame_clip_touchdown_insert_120_crop_lmdb/` | Precomputed FLAME features for MPSI rule-inserted views. |
| Rule task manifests | `data/touchdown/insert_touchdown/` | Level 1-4 rule-constrained benchmark manifests. |
| Touchdown navigation data | `data/touchdown/data/` | Navigation split data derived from Touchdown. |
| FLAME conversation data | `data/touchdown/ft_data/` | Training and evaluation conversations used by the FLAME-style navigation framework. |
| Semantic rule image dataset | `data/citynav_rules/` | Original rule-sign images and semantic labels used by MPSI for rule insertion. |
| Evaluation checkpoint | `checkpoints/Flame_checkpoint-2500/` | Released fine-tuned checkpoint for reproducing reported evaluation. |

The graph files `data/touchdown/graph/nodes.txt` and `data/touchdown/graph/links.txt` are included in the code repository.

### Semantic Rule Image Dataset

`City_Navigation_Rules_Dataset/` contains the semantic rule image set used to construct inserted regulatory signs in Rule-VLN.

```text
City_Navigation_Rules_Dataset/
|-- label.json
`-- img/
    |-- 00000/
    |   |-- GT-00000.csv
    |   |-- gtsrb_00000_00000.jpg
    |   `-- ...
    |-- 00001/
    `-- ...
```

The current release contains 177 label-id folders and 73,937 labeled rule-sign images. `label.json` is a list of records with fields such as:

```json
{
  "file": "00000/gtsrb_00000_00000.jpg",
  "label": "red and white circle 20 kph speed limit",
  "Influence_action": "0",
  "action_node": [0],
  "label_id": "00000"
}
```

In the Rule-VLN codebase this dataset is loaded from `data/citynav_rules/`, while MPSI reads rule images from `data/citynav_rules/img/`.

## Download

Install the Rule-VLN codebase and run:

```bash
git clone https://github.com/Jeffry-wen/Rule-VLN.git
cd Rule-VLN

python scripts/download_assets.py \
  --repo-id jeffry77/Rule-VLN \
  --output-root .
```

This script downloads the released Hugging Face assets and maps them to the paths expected by the training and evaluation configs.

## Quick Evaluation

After installing the main environment and downloading the assets:

```bash
conda env create -f environment.yml
conda activate rule_vln

python evaluate_sna.py \
  --config configs/eval_touchdown.yaml \
  --level 1
```

To evaluate all four curriculum levels:

```bash
bash scripts/evaluate_all_levels.sh
```

## Reconstructing Raw Image LMDBs

The raw panoramic images are not included here. To reconstruct the base 120-degree image LMDB, download the official Manhattan Touchdown / StreetLearn data from:

https://sites.google.com/view/streetlearn/dataset

Expected local layout:

```text
/path/to/touchdown/
|-- jpegs_manhattan_touchdown_2022/
|-- manhattan_touchdown_metadata_nodes.tsv
`-- manhattan_touchdown_metadata_links.tsv
```

Then run:

```bash
python scripts/prepare_touchdown.py \
  --streetlearn-root /path/to/touchdown \
  --output-root data/touchdown \
  --project-views \
  --view-output data/images/touchdown_crop_120

python tools/build_image_lmdb.py \
  --image-dir data/images/touchdown_crop_120 \
  --output-dir data/images/touchdown_crop_120_lmdb
```

To reconstruct inserted raw-image LMDBs, first run the MPSI generation pipeline from the code repository, then project the generated inserted panoramas:

```bash
mkdir -p data/images/touchdown_insert_crop_120

for level in 1 2 3 4; do
  python tools/project_panoramas.py \
    --panorama-dir "outputs/mpsi/task${level}/inserted_panoramas" \
    --nodes data/touchdown/graph/nodes.txt \
    --links data/touchdown/graph/links.txt \
    --output-dir data/images/touchdown_insert_crop_120 \
    --fov 120 \
    --size 1500
done

python tools/build_image_lmdb.py \
  --image-dir data/images/touchdown_insert_crop_120 \
  --output-dir data/images/touchdown_insert_crop_120_lmdb
```

The MPSI configuration expects the semantic rule image dataset at:

```text
data/citynav_rules/img/
```

## Intended Uses

Rule-VLN is intended for research on:

- Rule-compliant vision-and-language navigation.
- Outdoor embodied AI and urban navigation.
- Semantic constraint reasoning under visual observations.
- Navigation robustness under dynamic regulatory constraints.
- Evaluation of perception-to-planning rectification modules.

## Out-of-Scope Uses

This dataset is not intended for direct deployment in real-world navigation or safety-critical systems without independent validation. It should not be used in ways that violate the licenses or usage terms of Touchdown / StreetLearn, CityNav-Rules images, FLUX models, or other external assets.

## Limitations

- Rule insertions are generated by the MPSI pipeline and may not capture every visual condition of real-world regulatory signs.
- The released inserted feature LMDB was generated from a stochastic image-generation pipeline; newly regenerated inserted images are not expected to be pixel-identical.
- Raw StreetLearn / Touchdown panoramas are not redistributed here, so image-level reconstruction requires separately obtaining the official data.
- Results depend on the released graph, feature extraction pipeline, and model checkpoint. Changing these assets may change evaluation numbers.
- The benchmark is based on the Touchdown / StreetLearn urban environment and may not fully represent other cities, countries, traffic systems, or navigation domains.

## External Assets and Licensing

The Rule-VLN code repository is released under Apache-2.0. This Hugging Face dataset contains derived data, features, and model assets that may be subject to additional upstream terms.

Users are responsible for complying with the licenses and terms of:

- StreetLearn / Touchdown panoramas and metadata.
- The source datasets or assets used to construct `City_Navigation_Rules_Dataset`.
- FLUX and other image-generation model weights.
- FLAME / Hugging Face model checkpoints.
- Any other third-party models used during reconstruction.

Raw StreetLearn / Touchdown panoramas are not included in this dataset.

## Citation

If you find Rule-VLN useful, please cite:

```bibtex
@article{wen2026rule,
  title={Rule-VLN: Bridging Perception and Compliance via Semantic Reasoning and Geometric Rectification},
  author={Wen, Jiawen and Sun, Penglei and Zhang, Wenjie and Qiu, Suixuan and Xu, Weisheng and Yang, Xiaofei and Chu, Xiaowen},
  journal={arXiv preprint arXiv:2604.16993},
  year={2026}
}
```

## Acknowledgements

We thank FLAME for providing the navigation framework that Rule-VLN builds upon, and we thank the StreetLearn / Touchdown dataset for providing the Touchdown panoramic images used in this work.
