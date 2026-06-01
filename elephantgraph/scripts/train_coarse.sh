#!/usr/bin/env bash
set -euo pipefail

echo "=== Training Coarse H3 Generator (Level 1) ==="

python train.py \
    --config configs/coarse_config.yaml \
    --level coarse \
    "${@}"
