#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/train_touchdown.yaml}
deepspeed train_flame.py --config "$CONFIG"
