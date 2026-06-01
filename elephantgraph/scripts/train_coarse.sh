#!/usr/bin/env bash
set -euo pipefail

echo "=== Training Coarse H3 Generator (Level 1) ==="

python -m elephantgraph.train --level coarse
