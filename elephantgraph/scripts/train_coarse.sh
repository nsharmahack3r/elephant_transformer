#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Training Coarse H3 Generator (Level 1) ==="

python -m elephantgraph.train \
    --config "${PROJECT_DIR}/configs/coarse_config.yaml" \
    --level coarse \
    "${@}"
