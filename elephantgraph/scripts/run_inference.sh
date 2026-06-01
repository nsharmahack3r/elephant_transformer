#!/usr/bin/env bash
set -euo pipefail

echo "=== Running Inference ==="

python generate.py "${@}"
