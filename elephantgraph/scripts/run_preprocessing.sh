#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== ElephantGraph Preprocessing Pipeline ==="
echo "Input: ${PROJECT_DIR}/data/raw/elephant_gps.csv"

python -m elephantgraph.preprocessing.clean \
    --input "${PROJECT_DIR}/data/raw/elephant_gps.csv" \
    --output "${PROJECT_DIR}/data/processed/clean.csv"

python -m elephantgraph.preprocessing.resample \
    --input "${PROJECT_DIR}/data/processed/clean.csv" \
    --output "${PROJECT_DIR}/data/processed/hourly/"

python -m elephantgraph.preprocessing.windowing \
    --input "${PROJECT_DIR}/data/processed/clean.csv" \
    --output "${PROJECT_DIR}/data/processed/windows/"

python -m elephantgraph.preprocessing.h3_builder \
    --input "${PROJECT_DIR}/data/processed/hourly/" \
    --output "${PROJECT_DIR}/data/processed/h3_graph/"

echo "=== Preprocessing Complete ==="
