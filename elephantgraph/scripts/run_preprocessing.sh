#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== ElephantGraph Preprocessing Pipeline ==="
echo "Input: ${PROJECT_DIR}/data/raw/elephant_gps.csv"

# 1. Clean
python -m elephantgraph.preprocessing.clean \
    --input "${PROJECT_DIR}/data/raw/elephant_gps.csv" \
    --output "${PROJECT_DIR}/data/processed/clean.csv"

# 2. Hourly resample (for H3 graph)
python -m elephantgraph.preprocessing.resample \
    --input "${PROJECT_DIR}/data/processed/clean.csv" \
    --output "${PROJECT_DIR}/data/processed/hourly/hourly.csv"

# 3. Sliding windows + train/val/test split
python -m elephantgraph.preprocessing.windowing \
    --input "${PROJECT_DIR}/data/processed/clean.csv" \
    --output "${PROJECT_DIR}/data/processed/windows/" \
    --val-elephant AG005

# 4. H3 graph + node2vec embeddings
python -m elephantgraph.preprocessing.h3_builder \
    --input "${PROJECT_DIR}/data/processed/hourly/" \
    --output "${PROJECT_DIR}/data/processed/h3_graph/"

# 5. Fit and save scalers
python -m elephantgraph.preprocessing.scalers \
    --clean-data "${PROJECT_DIR}/data/processed/clean.csv" \
    --output "${PROJECT_DIR}/data/scalers/"

echo "=== Preprocessing Complete ==="
