#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Training Fine Diffusion Transformer (Level 2) ==="

python -m elephantgraph.train \
    --config "${PROJECT_DIR}/configs/fine_config.yaml" \
    --level fine \
    --train-data "${PROJECT_DIR}/data/processed/windows/train_windows.npy" \
    --val-data "${PROJECT_DIR}/data/processed/windows/val_windows.npy" \
    --scaler-dir "${PROJECT_DIR}/data/scalers/" \
    "${@}"
