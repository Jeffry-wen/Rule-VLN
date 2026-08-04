#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/eval_touchdown.yaml}
for level in 1 2 3 4; do
  python evaluate_sna.py --config "$CONFIG" --level "$level"
done
