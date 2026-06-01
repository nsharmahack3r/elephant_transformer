#!/usr/bin/env bash
set -euo pipefail

echo "=== ElephantGraph Preprocessing Pipeline ==="

# 1. Clean
echo "Step 1/5: Cleaning data..."
python -m elephantgraph.preprocessing.clean

# 2. Hourly resample (for H3 graph)
echo "Step 2/5: Hourly resampling..."
python -m elephantgraph.preprocessing.resample

# 3. Sliding windows + train/val/test split
echo "Step 3/5: Creating windows..."
python -m elephantgraph.preprocessing.windowing

# 4. H3 graph + node2vec embeddings
echo "Step 4/5: Building H3 graph..."
python -m elephantgraph.preprocessing.h3_builder

# 5. Fit and save scalers
echo "Step 5/5: Fitting scalers..."
python -m elephantgraph.preprocessing.scalers

echo "=== Preprocessing Complete ==="
