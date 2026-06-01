#!/usr/bin/env bash
set -euo pipefail

echo "=== Training Fine Diffusion Transformer (Level 2) ==="

python train.py \
    --config configs/fine_config.yaml \
    --level fine \
    "${@}"
