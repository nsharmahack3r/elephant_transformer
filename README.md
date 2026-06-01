# ElephantGraph

Hierarchical behavior-conditioned diffusion transformer for generating realistic elephant GPS trajectories.

## Setup

```bash
cd elephantgraph
python3 -m venv .venv && source .venv/bin/activate
pip install -e ..
```

## Quick Start

### 1. Preprocessing

Place your CSV at `elephantgraph/data/raw/elephant_gps.csv` (you can symlink or copy `./data/sample_data.csv` there). Then run the pipeline:

```bash
# Clean: handles outliers, GPS gaps, missing kinematics, categorical encoding
python -m elephantgraph.preprocessing.clean \
    --input elephantgraph/data/raw/elephant_gps.csv \
    --output elephantgraph/data/processed/clean.csv

# Resample to hourly for the coarse-level H3 graph
python -m elephantgraph.preprocessing.resample \
    --input elephantgraph/data/processed/clean.csv \
    --output elephantgraph/data/processed/hourly/hourly.csv

# Create sliding windows (200-point, 50% overlap) and split by elephant
python -m elephantgraph.preprocessing.windowing \
    --input elephantgraph/data/processed/clean.csv \
    --output elephantgraph/data/processed/windows/ \
    --val-elephant AG005

# Build H3 graph with hierarchical splitting + train node2vec embeddings
python -m elephantgraph.preprocessing.h3_builder \
    --input elephantgraph/data/processed/hourly/ \
    --output elephantgraph/data/processed/h3_graph/

# Fit and save GPS, kinematic, and environmental scalers
python -m elephantgraph.preprocessing.scalers \
    --clean-data elephantgraph/data/processed/clean.csv \
    --output elephantgraph/data/scalers/
```

Or run all at once:

```bash
bash elephantgraph/scripts/run_preprocessing.sh
```

**Expected CSV columns** (see `elephantgraph/preprocessing/schema.py` for the full schema):

| Required | Optional |
|---|---|
| `elephant_id`, `timestamp`, `longitude`, `latitude` | `behavior` (STOP/MOVE), `behavior_code` |
| `speed_kmh`, `acceleration`, `turning_angle` | `season` (dry/wet), `season_code` |
| `bearing`, `dir_persistence`, `step_dist_m` | `time_of_day` (night/morning/afternoon/evening) |
| `NDVI`, `EVI`, `LST_celsius` | `LULC_class`, `human_settle`, `movement_type` |
| `elevation_m`, `slope_deg`, `water_occ_1km` | `NSD`, `dist_from_origin_m` |

Missing optional columns are filled with zeros. String categoricals (e.g. `behavior`) are auto-encoded during cleaning.

### 2. Train

```bash
python -m elephantgraph.train \
    --config elephantgraph/configs/fine_config.yaml \
    --level fine \
    --train-data elephantgraph/data/processed/windows/train_windows.npy \
    --val-data elephantgraph/data/processed/windows/val_windows.npy \
    --scaler-dir elephantgraph/data/scalers/
```

Checkpoints are saved to `checkpoints/fine/best_model.pt`.

### 3. Generate trajectories

```bash
python -m elephantgraph.generate \
    --checkpoint checkpoints/fine/best_model.pt \
    --n 500 \
    --behavior MOVE \
    --season dry \
    --time-of-day morning \
    --lulc 10 \
    --seq-len 200 \
    --scaler-dir elephantgraph/data/scalers/ \
    --output results/dry_move_trajectories.npy
```

Key generation arguments:

| Argument | Description | Options |
|---|---|---|
| `--behavior` | Movement state | `MOVE`, `STOP` |
| `--season` | Season | `dry`, `wet` |
| `--time-of-day` | Time period | `night`, `morning`, `afternoon`, `evening` |
| `--lulc` | Land cover class | 0–19 |
| `--human-settle` | Near settlement | flag |
| `--seq-len` | Points per trajectory | default 200 |
| `--n` | Number to generate | default 1000 |

### 4. Evaluate

```bash
python -m elephantgraph.evaluate \
    --generated results/dry_move_trajectories.npy \
    --real elephantgraph/data/processed/windows/test_windows.npy \
    --output results/evaluation_report.json
```

Outputs: average displacement error, final displacement error, coverage, speed distribution match, turning angle distribution match, density pattern score, r-coefficient, speed mean match, and movement consistency.

## Project Structure

```
elephantgraph/
├── data/                  # Raw, processed data, scalers
├── preprocessing/         # Cleaning, windowing, H3 graph, scalers
├── models/                # Diffusion transformer, embeddings, adaLN
├── training/              # Dataset, losses, training loops
├── inference/             # Generator, occupancy sampler
├── evaluation/            # Spatial, kinematic, distributional metrics
├── configs/               # YAML configs for fine/coarse levels
├── scripts/               # Shell scripts for full pipelines
├── train.py               # Training entry point
├── generate.py            # Inference entry point
└── evaluate.py            # Evaluation entry point
```

## Architecture

Two-level hierarchical model:

- **Coarse (Level 1):** H3 geographic discretization → predicts which regions an elephant visits per hour
- **Fine (Level 2):** Diffusion transformer at 10‑second resolution — generates actual lon/lat trajectories conditioned on behavior (MOVE/STOP), season, time of day, land cover, and environmental context

The fine model combines GPS and kinematic token embeddings with behavior-conditioned adaptive layer normalization (adaLN) blocks inside a DDIM diffusion framework.
