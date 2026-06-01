#!/usr/bin/env bash
set -euo pipefail

echo "=== Training Fine Diffusion Transformer (Level 2) ==="

python -m elephantgraph.train --level fine
