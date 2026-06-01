#!/usr/bin/env bash
set -euo pipefail

echo "=== Running Inference ==="

python -m elephantgraph.generate "${@}"
