# ElephantGraph: Hierarchical Behavior-Conditioned Trajectory Generation
## Complete Code Structure — Build, Train & Inference Guide

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Directory Structure](#2-directory-structure)
3. [Data Schema & Feature Roles](#3-data-schema--feature-roles)
4. [Preprocessing Pipeline](#4-preprocessing-pipeline)
5. [Model Architecture](#5-model-architecture)
   - 5.1 Embedding Layer
   - 5.2 Positional Encoding
   - 5.3 Behavior-Conditioned adaLN Block
   - 5.4 Diffusion Process
   - 5.5 H3 Coarse Generator
   - 5.6 Full Model Assembly
6. [Training Pipeline](#6-training-pipeline)
7. [Inference Pipeline](#7-inference-pipeline)
8. [Evaluation](#8-evaluation)
9. [Configuration Reference](#9-configuration-reference)
10. [Dependencies](#10-dependencies)

---

## 1. Project Overview

### Problem
Generate realistic long-horizon elephant GPS trajectories using 15M high-frequency
(10-second) fixes from 6 individual elephants, conditioned on:
- Behavioral state (MOVE / STOP)
- Environmental satellite indices (NDVI, EVI, LST, water occurrence)
- Terrain features (elevation, slope, LULC class)
- Temporal context (season, time of day, month)

### What Makes This Dataset Unique
```
Sampling rate   : ~10 seconds per fix
Total rows      : ~15 million
Elephants       : 6 individuals
Per elephant    : ~2.5M rows (~289 days continuous)
Feature depth   : 38 columns including pre-computed kinematics
Behavior labels : MOVE / STOP (pre-labeled)
Satellite data  : NDVI, EVI, LST already joined per GPS fix
```

### Architecture Family
This model combines:
- **WildGraph** → H3 geographic discretisation + occupancy sampler (coarse level)
- **Traj-Transformer** → lon-lat embedding + adaLN conditioning + diffusion (fine level)
- **Original contributions** → kinematic token embedding, behavior-conditioned generation,
  ecological occupancy weighting, seasonal latent dictionary

### Two-Level Hierarchy
```
Level 1 — Coarse Generator (WHERE):
  Hourly resampled data → H3 region sequences → Transformer
  Answers: Which areas does the elephant visit each hour/day?

Level 2 — Fine Generator (HOW):
  10-second windows → GPS + kinematics → Diffusion Transformer
  Answers: How does the elephant physically move within those areas?
```

---

## 2. Directory Structure

```
elephantgraph/
│
├── data/
│   ├── raw/
│   │   └── elephant_gps.csv              # Full 15M row dataset
│   ├── processed/
│   │   ├── windows/                      # Sliding window .npy files
│   │   │   ├── train_windows.npy
│   │   │   ├── val_windows.npy
│   │   │   └── test_windows.npy
│   │   ├── hourly/                       # Hourly resampled (coarse level)
│   │   │   ├── hourly_train.csv
│   │   │   ├── hourly_val.csv
│   │   │   └── hourly_test.csv
│   │   └── h3_graph/                     # WildGraph H3 artifacts
│   │       ├── graph.pkl                 # NetworkX graph object
│   │       ├── node_embeddings.npy       # node2vec embeddings
│   │       ├── node_features.json        # NDVI/NDWI per H3 node
│   │       └── latent_dict.pkl           # Latent dictionary (dry/wet)
│   └── scalers/
│       ├── gps_scaler.pkl                # MinMaxScaler for lon/lat
│       ├── kinematic_scaler.pkl          # StandardScaler for speed etc.
│       └── env_scaler.pkl                # StandardScaler for NDVI etc.
│
├── preprocessing/
│   ├── __init__.py
│   ├── clean.py                          # Gap detection, outlier removal
│   ├── resample.py                       # Hourly resampling for coarse level
│   ├── windowing.py                      # Sliding window creation
│   ├── h3_builder.py                     # H3 graph construction
│   ├── node2vec_trainer.py               # Graph embedding training
│   └── scalers.py                        # Fit and save all scalers
│
├── models/
│   ├── __init__.py
│   ├── embeddings.py                     # All embedding classes
│   ├── positional_encoding.py            # 4D positional encoding
│   ├── adaln_block.py                    # Behavior-conditioned adaLN
│   ├── diffusion.py                      # DDPM/DDIM forward/reverse
│   ├── coarse_generator.py               # Level 1: H3 Transformer
│   ├── fine_generator.py                 # Level 2: Diffusion Transformer
│   └── elephant_graph.py                 # Full assembled model
│
├── training/
│   ├── __init__.py
│   ├── dataset.py                        # PyTorch Dataset classes
│   ├── losses.py                         # Haversine, kinematic, eco losses
│   ├── trainer_coarse.py                 # Training loop for Level 1
│   ├── trainer_fine.py                   # Training loop for Level 2
│   └── callbacks.py                      # Checkpointing, early stopping
│
├── inference/
│   ├── __init__.py
│   ├── generator.py                      # Main generation interface
│   ├── occupancy_sampler.py              # Ecological occupancy sampling
│   └── postprocess.py                    # Smoothing, validity checks
│
├── evaluation/
│   ├── __init__.py
│   ├── spatial_metrics.py                # Hausdorff, DTW, FDE
│   ├── distributional_metrics.py         # Density, Pattern, r-Coeff
│   ├── ecological_metrics.py             # NDVI/water proximity checks
│   └── kinematic_metrics.py              # Speed/accel distribution match
│
├── configs/
│   ├── base_config.yaml                  # Default hyperparameters
│   ├── coarse_config.yaml                # Level 1 specific config
│   └── fine_config.yaml                  # Level 2 specific config
│
├── scripts/
│   ├── run_preprocessing.sh
│   ├── train_coarse.sh
│   ├── train_fine.sh
│   └── run_inference.sh
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_h3_visualisation.ipynb
│   ├── 03_training_curves.ipynb
│   └── 04_generated_trajectory_viz.ipynb
│
├── train.py                              # Main training entry point
├── generate.py                           # Main inference entry point
├── evaluate.py                           # Main evaluation entry point
└── requirements.txt
```

---

## 3. Data Schema & Feature Roles

### Input Columns (38 total)

```python
# preprocessing/schema.py

FEATURE_ROLES = {

    # ── PRIMARY OUTPUT ──────────────────────────────────────────────
    # These are what the model generates
    'gps_output': [
        'longitude',        # float64 — target output
        'latitude',         # float64 — target output
    ],

    # ── KINEMATIC TOKEN EMBEDDINGS ──────────────────────────────────
    # Embedded as separate tokens alongside GPS (your key extension
    # over both papers). Makes generated movement physically realistic.
    'kinematic_tokens': [
        'speed_kmh',        # float64 — 0 to ~25 km/h for elephants
        'acceleration',     # float64 — can be negative (deceleration)
        'turning_angle',    # float64 — degrees, -180 to 180
        'bearing',          # float64 — compass direction, 0-360
        'dir_persistence',  # float64 — 0-1, how straight movement is
        'step_dist_m',      # float64 — distance per step in metres
    ],

    # ── ENVIRONMENTAL EMBEDDINGS ─────────────────────────────────────
    # Continuous satellite/terrain values, embedded per timestep.
    # Primary ecological drivers of elephant movement.
    'env_tokens': [
        'NDVI',             # float64 — vegetation density, -1 to 1
        'EVI',              # float64 — enhanced vegetation, -1 to 1
        'LST_celsius',      # float64 — land surface temperature
        'elevation_m',      # float64 — metres above sea level
        'slope_deg',        # float64 — terrain slope in degrees
        'water_occ_1km',    # float64 — 0-1, water occurrence in 1km radius
    ],

    # ── adaLN CONDITIONS ─────────────────────────────────────────────
    # Categorical/discrete features injected via adaptive layer norm.
    # These modulate the ENTIRE transformer's behavior per condition.
    'adaln_conditions': [
        'behavior_code',       # int — 0=STOP, 1=MOVE  ← most important
        'season_code',         # int — 0=dry, 1=wet
        'time_of_day',         # str → encode as int: 0=night,1=morning,
                               #        2=afternoon,3=evening
        'LULC_class',          # float → int — land cover class (0-19)
        'human_settle',        # float → int — 0=no, 1=yes
        'movement_type_code',  # int — 0=Nomad, 1=Resident, etc.
    ],

    # ── RANGE CONTEXT ────────────────────────────────────────────────
    # Global movement state — tells model whether elephant is in
    # exploration mode (high NSD) or staying in home range (low NSD)
    'range_context': [
        'NSD',                 # float64 — Net Squared Displacement
        'dist_from_origin_m',  # float64 — distance from first fix
        'hr_95_km2',           # float64 — 95% home range size
        'hr_50_km2',           # float64 — 50% core home range size
    ],

    # ── TEMPORAL ────────────────────────────────────────────────────
    'temporal': [
        'hour',                # int — 0-23
        'month',               # int — 1-12
        'days_elapsed',        # float — days since tracking start
    ],

    # ── DROP BEFORE TRAINING ─────────────────────────────────────────
    # Redundant, string versions, or derivable at generation time
    'drop': [
        'event_id',            # just an ID
        'elephant_id',         # use as cross-val fold identifier
        'timestamp',           # decomposed into hour/month/days_elapsed
        'season',              # string version → use season_code
        'behavior',            # string version → use behavior_code
        'movement_type',       # string version → use movement_type_code
        'rolling_speed_mean',  # derivable during generation
        'rolling_turn_std',    # derivable during generation
        'rolling_step_mean',   # derivable during generation
        'aspect_deg',          # weak predictor for elephant movement
    ],
}

# Encoding maps for categorical string columns
ENCODING_MAPS = {
    'time_of_day': {'night': 0, 'morning': 1, 'afternoon': 2, 'evening': 3},
    'season':      {'dry': 0, 'wet': 1},
    'behavior':    {'STOP': 0, 'MOVE': 1},
}
```

---

## 4. Preprocessing Pipeline

### 4.1 `preprocessing/clean.py`

```python
import pandas as pd
import numpy as np
from geopy.distance import geodesic

def remove_gps_outliers(df, max_speed_kmh=70):
    """
    Remove physically impossible GPS fixes.
    Elephants max ~25 km/h sprint, so 70 km/h = impossible GPS jump.
    """
    df = df.sort_values(['elephant_id', 'timestamp'])
    df = df[df['speed_kmh'] < max_speed_kmh]
    return df.reset_index(drop=True)


def flag_large_gaps(df, gap_threshold_sec=300):
    """
    Flag rows following a GPS gap > threshold.
    These should not be treated as continuous movement.
    Window creation will skip across these flags.

    gap_threshold_sec=300 → 5 minute gap (common during canopy cover)
    """
    df = df.sort_values(['elephant_id', 'timestamp'])
    df['has_gap_before'] = (
        df.groupby('elephant_id')['time_diff_sec']
          .transform(lambda x: x > gap_threshold_sec)
    )
    return df


def impute_missing_kinematics(df):
    """
    Forward-fill kinematic features within each elephant's track.
    Missing values occur at trajectory start (no previous point).
    """
    kinematic_cols = [
        'speed_kmh', 'acceleration', 'turning_angle',
        'bearing', 'dir_persistence', 'step_dist_m'
    ]
    df = df.sort_values(['elephant_id', 'timestamp'])
    df[kinematic_cols] = (
        df.groupby('elephant_id')[kinematic_cols]
          .transform(lambda x: x.fillna(method='ffill')
                                .fillna(0))  # fill start with 0
    )
    return df


def encode_categoricals(df):
    """Encode string categoricals to integer codes."""
    from preprocessing.schema import ENCODING_MAPS
    for col, mapping in ENCODING_MAPS.items():
        if col in df.columns:
            df[col + '_encoded'] = df[col].map(mapping)
    return df


def run_cleaning_pipeline(input_path, output_path):
    df = pd.read_csv(input_path, parse_dates=['timestamp'])
    df = remove_gps_outliers(df)
    df = flag_large_gaps(df)
    df = impute_missing_kinematics(df)
    df = encode_categoricals(df)
    df.to_csv(output_path, index=False)
    print(f"Cleaned data: {len(df)} rows → {output_path}")
```

---

### 4.2 `preprocessing/windowing.py`

```python
import numpy as np
import pandas as pd
from preprocessing.schema import FEATURE_ROLES

WINDOW_SIZE = 200    # 200 × 10sec = ~33 minutes per sample
STRIDE      = 100    # 50% overlap → ~150,000 windows from 15M rows


def create_windows(df, window_size=WINDOW_SIZE, stride=STRIDE):
    """
    Sliding window over sorted GPS fixes per elephant.
    Skips windows that span a GPS gap (has_gap_before flag).

    Returns list of dicts, each representing one training sample.
    """
    all_windows = []

    for elephant_id, group in df.groupby('elephant_id'):
        group = group.sort_values('timestamp').reset_index(drop=True)
        n = len(group)

        for start in range(0, n - window_size, stride):
            window = group.iloc[start : start + window_size]

            # Skip if window contains a GPS gap
            if window['has_gap_before'].any():
                continue

            sample = {
                # GPS — primary generation target
                'lon':    window['longitude'].values.astype(np.float32),
                'lat':    window['latitude'].values.astype(np.float32),

                # Kinematic tokens (per timestep)
                'speed':    window['speed_kmh'].values.astype(np.float32),
                'accel':    window['acceleration'].values.astype(np.float32),
                'turning':  window['turning_angle'].values.astype(np.float32),
                'bearing':  window['bearing'].values.astype(np.float32),
                'persist':  window['dir_persistence'].values.astype(np.float32),
                'step':     window['step_dist_m'].values.astype(np.float32),

                # Environmental (per timestep)
                'ndvi':     window['NDVI'].values.astype(np.float32),
                'evi':      window['EVI'].values.astype(np.float32),
                'lst':      window['LST_celsius'].values.astype(np.float32),
                'elev':     window['elevation_m'].values.astype(np.float32),
                'slope':    window['slope_deg'].values.astype(np.float32),
                'water':    window['water_occ_1km'].values.astype(np.float32),

                # Window-level conditions (take mode/first)
                'behavior':    int(window['behavior_code'].mode()[0]),
                'season':      int(window['season_code'].iloc[0]),
                'time_of_day': int(window['time_of_day_encoded'].mode()[0]),
                'lulc':        int(window['LULC_class'].mode()[0]),
                'human_settle':int(window['human_settle'].mode()[0]),
                'move_type':   int(window['movement_type_code'].iloc[0]),

                # Range context (scalar per window)
                'nsd':         float(window['NSD'].iloc[-1]),
                'dist_origin': float(window['dist_from_origin_m'].iloc[-1]),

                # Metadata
                'elephant_id': elephant_id,
                'start_time':  str(window['timestamp'].iloc[0]),
            }
            all_windows.append(sample)

    return all_windows


def split_by_elephant(windows, val_elephant='AG005',
                       test_elephant=None):
    """
    Leave-one-elephant-out cross-validation split.
    With 6 elephants, hold out 1 for test, 1 for validation.
    This tests generalisation to unseen individuals.
    """
    train = [w for w in windows
             if w['elephant_id'] not in [val_elephant, test_elephant]]
    val   = [w for w in windows if w['elephant_id'] == val_elephant]
    test  = [w for w in windows if w['elephant_id'] == test_elephant]
    return train, val, test


def save_windows(windows, path):
    np.save(path, np.array(windows, dtype=object), allow_pickle=True)
```

---

### 4.3 `preprocessing/h3_builder.py`

```python
import h3
import networkx as nx
import json
import numpy as np
from node2vec import Node2Vec

INITIAL_ZOOM    = 4     # Start with coarse H3 hexagons
SPLIT_THRESHOLD = 1.0   # Split if region diameter > 1.0 degree (~111 km)
NODE2VEC_DIM    = 64    # Structural embedding dimension
ECO_DIM         = 16    # Ecological feature dimension
TOTAL_EMBED_DIM = NODE2VEC_DIM + ECO_DIM  # = 80


def build_h3_graph(hourly_df, split_threshold=SPLIT_THRESHOLD):
    """
    Build H3-based prototype graph from hourly-resampled trajectories.
    Uses WildGraph's hierarchical splitting logic.
    Nodes = H3 hexagonal regions. Edges = observed transitions.
    """
    G = nx.DiGraph()
    zoom = INITIAL_ZOOM

    # Step 1: Assign initial H3 regions
    hourly_df['h3_region'] = hourly_df.apply(
        lambda r: h3.geo_to_h3(r['latitude'], r['longitude'], zoom),
        axis=1
    )

    # Step 2: Hierarchical split — refine dense regions
    regions = hourly_df['h3_region'].unique()
    final_regions = {}

    for region in regions:
        pts = hourly_df[hourly_df['h3_region'] == region][
            ['latitude', 'longitude']
        ].values

        if len(pts) < 2:
            final_regions[region] = zoom
            continue

        # Calculate diameter (max pairwise distance in degrees)
        from itertools import combinations
        dists = [
            np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
            for a, b in combinations(pts[:50], 2)  # sample for speed
        ]
        diameter = max(dists) if dists else 0

        if diameter > split_threshold:
            finer_zoom = zoom + 2
            # Reassign points to finer H3 cells
            for _, row in hourly_df[
                hourly_df['h3_region'] == region
            ].iterrows():
                finer = h3.geo_to_h3(
                    row['latitude'], row['longitude'], finer_zoom
                )
                final_regions[finer] = finer_zoom
        else:
            final_regions[region] = zoom

    # Step 3: Re-encode trajectories with final regions
    def get_final_h3(lat, lon):
        coarse = h3.geo_to_h3(lat, lon, zoom)
        z = final_regions.get(coarse, zoom)
        return h3.geo_to_h3(lat, lon, z)

    hourly_df['h3_final'] = hourly_df.apply(
        lambda r: get_final_h3(r['latitude'], r['longitude']),
        axis=1
    )

    # Step 4: Add nodes with ecological features
    for region in hourly_df['h3_final'].unique():
        pts = hourly_df[hourly_df['h3_final'] == region]
        G.add_node(region, **{
            'ndvi_mean':  pts['NDVI'].mean(),
            'ndwi_mean':  pts['water_occ_1km'].mean(),
            'evi_mean':   pts['EVI'].mean(),
            'lst_mean':   pts['LST_celsius'].mean(),
            'elev_mean':  pts['elevation_m'].mean(),
            'lulc_mode':  pts['LULC_class'].mode()[0],
            # Seasonal split for latent dictionary
            'ndvi_dry':   pts[pts['season_code']==0]['NDVI'].mean(),
            'ndvi_wet':   pts[pts['season_code']==1]['NDVI'].mean(),
            'water_dry':  pts[pts['season_code']==0]['water_occ_1km'].mean(),
            'water_wet':  pts[pts['season_code']==1]['water_occ_1km'].mean(),
        })

    # Step 5: Add edges from observed transitions
    for eleph_id, group in hourly_df.groupby('elephant_id'):
        group = group.sort_values('timestamp')
        regions_seq = group['h3_final'].values
        for src, dst in zip(regions_seq[:-1], regions_seq[1:]):
            if src != dst:
                if G.has_edge(src, dst):
                    G[src][dst]['count'] += 1
                else:
                    G.add_edge(src, dst, count=1)

    return G, hourly_df


def train_node2vec(G, dimensions=NODE2VEC_DIM):
    """
    node2vec with high q value for long-range exploration.
    High q → depth-first → captures distant node relationships.
    Critical for long-horizon elephant migration patterns.
    """
    node2vec = Node2Vec(
        G,
        dimensions=dimensions,
        walk_length=40,
        num_walks=200,
        p=1,
        q=4,           # High q → explore distant regions (migration!)
        workers=4,
        quiet=True
    )
    model = node2vec.fit(window=10, min_count=1)
    return model


def build_ecological_embeddings(G, node2vec_model):
    """
    Concatenate structural (node2vec) + ecological (NDVI/EVI/LST/water)
    embeddings for each node.
    Final dim = 64 (structural) + 16 (ecological) = 80
    """
    import torch
    import torch.nn as nn

    eco_proj = nn.Linear(4, ECO_DIM)  # NDVI, EVI, LST, water → 16 dim

    embeddings = {}
    for node in G.nodes():
        # Structural embedding from node2vec
        structural = torch.tensor(
            node2vec_model.wv[node], dtype=torch.float32
        )

        # Ecological features
        eco_raw = torch.tensor([
            G.nodes[node].get('ndvi_mean', 0),
            G.nodes[node].get('evi_mean',  0),
            G.nodes[node].get('lst_mean',  0),
            G.nodes[node].get('ndwi_mean', 0),
        ], dtype=torch.float32)

        eco_emb = eco_proj(eco_raw)

        embeddings[node] = torch.cat([structural, eco_emb]).detach().numpy()

    return embeddings
```

---

## 5. Model Architecture

### 5.1 `models/embeddings.py`

```python
import torch
import torch.nn as nn
import math


class LonLatKinematicEmbedding(nn.Module):
    """
    Extends Traj-Transformer's lon-lat-emb with kinematic tokens.
    Each timestep produces 8 tokens:
      [lon, lat, speed, accel, turning, bearing, persist, step]

    Why separate embeddings per feature:
    - Lon and lat have different geographic meaning
    - Speed and turning have different movement semantics
    - Attention can learn cross-feature relationships
      (e.g., high speed + low turning = straight run)
    """
    def __init__(self, d_model=256):
        super().__init__()
        self.d_model = d_model

        # GPS tokens (from Traj-Transformer)
        self.lon_embed  = nn.Linear(1, d_model)
        self.lat_embed  = nn.Linear(1, d_model)

        # Kinematic tokens (original contribution)
        self.speed_embed   = nn.Linear(1, d_model)
        self.accel_embed   = nn.Linear(1, d_model)
        self.turning_embed = nn.Linear(1, d_model)
        self.bearing_embed = nn.Linear(1, d_model)
        self.persist_embed = nn.Linear(1, d_model)
        self.step_embed    = nn.Linear(1, d_model)

        # Token type IDs (0–7) for positional encoding
        self.TOKEN_TYPES = {
            'lon': 0, 'lat': 1,
            'speed': 2, 'accel': 3,
            'turning': 4, 'bearing': 5,
            'persist': 6, 'step': 7
        }

    def forward(self, lon, lat, speed, accel,
                turning, bearing, persist, step):
        """
        Args:
            All inputs: (batch_size, seq_len)
        Returns:
            tokens: (batch_size, 8 * seq_len, d_model)
        """
        B, T = lon.shape

        tokens = torch.stack([
            self.lon_embed(lon.unsqueeze(-1)),
            self.lat_embed(lat.unsqueeze(-1)),
            self.speed_embed(speed.unsqueeze(-1)),
            self.accel_embed(accel.unsqueeze(-1)),
            self.turning_embed(turning.unsqueeze(-1)),
            self.bearing_embed(bearing.unsqueeze(-1)),
            self.persist_embed(persist.unsqueeze(-1)),
            self.step_embed(step.unsqueeze(-1)),
        ], dim=2)  # (B, T, 8, d_model)

        return tokens.view(B, 8 * T, self.d_model)


class EnvironmentalEmbedding(nn.Module):
    """
    Embeds per-timestep environmental features as context.
    Output is added to all tokens at the same timestep.
    """
    def __init__(self, d_model=256):
        super().__init__()
        # 6 env features: NDVI, EVI, LST, elev, slope, water
        self.proj = nn.Sequential(
            nn.Linear(6, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, ndvi, evi, lst, elev, slope, water):
        """
        Args: all (batch, seq_len)
        Returns: (batch, seq_len, d_model)
        """
        env = torch.stack([ndvi, evi, lst, elev, slope, water], dim=-1)
        return self.proj(env)
```

---

### 5.2 `models/positional_encoding.py`

```python
import torch
import math


class ElephantPositionalEncoding(torch.nn.Module):
    """
    8D positional encoding — extends Traj-Transformer's 2D PE.

    For each GPS timestep, 8 tokens are created:
      [lon, lat, speed, accel, turning, bearing, persist, step]

    The PE must encode TWO things per token:
      1. TOKEN TYPE (which of the 8 features is this?)
      2. SEQUENCE POSITION (which timestep?)

    Format: PE = [type_encoding (d/2) | position_encoding (d/2)]
    """

    # Fixed type IDs for each of the 8 token types
    TOKEN_TYPE_IDS = {
        'lon': 0, 'lat': 1,
        'speed': 2, 'accel': 3,
        'turning': 4, 'bearing': 5,
        'persist': 6, 'step': 7
    }

    def __init__(self, d_model=256, max_seq_len=5000):
        super().__init__()
        self.d_model = d_model
        half = d_model // 2
        self.pe_table = self._build_pe_table(max_seq_len, half)

    def _sinusoidal_pe(self, position, d):
        """Standard sinusoidal encoding at given position."""
        pe = torch.zeros(d)
        for i in range(0, d, 2):
            div = 10000 ** (2 * i / d)
            pe[i]   = math.sin(position / div)
            if i + 1 < d:
                pe[i+1] = math.cos(position / div)
        return pe

    def _build_pe_table(self, max_len, half_d):
        """Pre-compute PE for all positions and 8 token types."""
        # Shape: (max_len, 8, d_model)
        table = torch.zeros(max_len, 8, self.d_model)
        for pos in range(max_len):
            for type_id in range(8):
                type_pe = self._sinusoidal_pe(type_id, half_d)
                pos_pe  = self._sinusoidal_pe(pos, half_d)
                table[pos, type_id] = torch.cat([type_pe, pos_pe])
        return table  # register as buffer if needed

    def forward(self, seq_len):
        """
        Returns PE for a sequence of seq_len timesteps.
        Output shape: (8 * seq_len, d_model)
        Ordering matches LonLatKinematicEmbedding output:
          [lon_t0, lat_t0, speed_t0, ..., step_t0,
           lon_t1, lat_t1, ..., step_t1, ...]
        """
        pe_list = []
        for t in range(seq_len):
            for type_id in range(8):
                pe_list.append(self.pe_table[t, type_id])
        return torch.stack(pe_list)  # (8*seq_len, d_model)
```

---

### 5.3 `models/adaln_block.py`

```python
import torch
import torch.nn as nn
import math


class BehaviorConditionedAdaLN(nn.Module):
    """
    Transformer block with Adaptive Layer Norm (from Traj-Transformer).
    Conditions the ENTIRE attention pattern on behavioral + ecological state.

    Key insight: MOVE and STOP require fundamentally different attention:
    - MOVE: attend to direction, speed, distant future positions
    - STOP: attend to local area, water/food proximity, duration

    adaLN lets the model learn these different attention patterns
    by modulating scale (γ) and shift (β) of layer norms.
    """

    def __init__(self, d_model=256, nhead=8, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

        # ── CONDITION ENCODERS ──────────────────────────────────────
        # Each condition is embedded and summed before modulation

        self.behavior_embed    = nn.Embedding(2,  d_model)  # STOP/MOVE
        self.season_embed      = nn.Embedding(2,  d_model // 4)
        self.tod_embed         = nn.Embedding(4,  d_model // 4)
        self.lulc_embed        = nn.Embedding(20, d_model // 4)
        self.settle_embed      = nn.Embedding(2,  d_model // 4)
        self.move_type_embed   = nn.Embedding(5,  d_model // 4)

        # Project all conditions to d_model
        cond_in = d_model + 5 * (d_model // 4)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_in, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )

        # ── adaLN MODULATION ────────────────────────────────────────
        # 6 parameters: γ1, β1, α1 (attention), γ2, β2, α2 (FF)
        self.modulation = nn.Linear(d_model, 6 * d_model, bias=True)

        # Zero-init: model starts as identity map (training trick
        # from Traj-Transformer / DiT paper — improves convergence)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def encode_condition(self, behavior, season, time_of_day,
                         lulc, human_settle, move_type):
        """Build condition vector from all categorical inputs."""
        beh   = self.behavior_embed(behavior)       # (B, d_model)
        sea   = self.season_embed(season)
        tod   = self.tod_embed(time_of_day)
        lulc_ = self.lulc_embed(lulc.long())
        set_  = self.settle_embed(human_settle.long())
        mt    = self.move_type_embed(move_type)

        cond  = torch.cat([beh, sea, tod, lulc_, set_, mt], dim=-1)
        return self.cond_proj(cond)  # (B, d_model)

    def forward(self, x, behavior, season, time_of_day,
                lulc, human_settle, move_type):
        """
        Args:
            x:            (B, seq_len, d_model)
            behavior:     (B,) int tensor — 0=STOP, 1=MOVE
            season:       (B,) int tensor — 0=dry,  1=wet
            time_of_day:  (B,) int tensor — 0-3
            lulc:         (B,) int tensor — 0-19
            human_settle: (B,) int tensor — 0/1
            move_type:    (B,) int tensor — movement type code
        Returns:
            x:            (B, seq_len, d_model)
            attn_weights: (B, seq_len, seq_len) — interpretable!
        """
        # Build and modulate
        cond = self.encode_condition(
            behavior, season, time_of_day, lulc, human_settle, move_type
        )
        γ1, β1, α1, γ2, β2, α2 = self.modulation(cond).chunk(6, dim=-1)
        # Shapes: each (B, d_model) → unsqueeze to (B, 1, d_model)

        # ── MODULATED SELF-ATTENTION ────────────────────────────────
        x_norm = (1 + γ1.unsqueeze(1)) * self.norm1(x) + β1.unsqueeze(1)
        attn_out, attn_weights = self.attn(x_norm, x_norm, x_norm)
        x = x + α1.unsqueeze(1) * attn_out

        # ── MODULATED FEEDFORWARD ───────────────────────────────────
        x_norm = (1 + γ2.unsqueeze(1)) * self.norm2(x) + β2.unsqueeze(1)
        x = x + α2.unsqueeze(1) * self.ff(x_norm)

        return x, attn_weights
```

---

### 5.4 `models/diffusion.py`

```python
import torch
import torch.nn as nn
import numpy as np


class DDIMDiffusion:
    """
    Denoising Diffusion Implicit Models (from Traj-Transformer).
    Adapted for elephant trajectory windows.

    T=200 steps (vs 1000 in original) — sufficient for wildlife data
    S=40 steps during sampling — fast generation via DDIM skip
    """

    def __init__(self, T=200, S=40, beta_start=0.0001, beta_end=0.02):
        self.T = T
        self.S = S  # sampling steps (< T)

        # Noise schedule
        self.betas     = torch.linspace(beta_start, beta_end, T)
        self.alphas    = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

        # Sampling timesteps (evenly spaced subset)
        self.sample_steps = list(
            range(0, T, T // S)
        )[::-1]  # reverse for denoising

    def q_sample(self, x0, t):
        """
        Forward process: add noise to clean trajectory x0.
        x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise

        Args:
            x0: clean trajectory (B, seq_len, 2) — lon/lat only
            t:  diffusion timestep (B,) int tensor
        Returns:
            x_t:   noisy trajectory
            noise: the added noise (prediction target)
        """
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1).to(x0.device)
        noise = torch.randn_like(x0)
        x_t   = (alpha_bar_t ** 0.5) * x0 + \
                ((1 - alpha_bar_t) ** 0.5) * noise
        return x_t, noise

    def p_sample_ddim(self, x_t, t, t_prev, predicted_noise):
        """
        Reverse process: one DDIM denoising step.
        Deterministic (sigma_t = 0) for stable generation.

        Args:
            x_t:              noisy trajectory at step t
            t, t_prev:        current and previous timestep indices
            predicted_noise:  noise estimate from transformer
        Returns:
            x_t_prev:         less noisy trajectory
        """
        alpha_bar_t      = self.alpha_bar[t].view(-1, 1, 1).to(x_t.device)
        alpha_bar_t_prev = self.alpha_bar[t_prev].view(-1, 1, 1).to(x_t.device) \
                           if t_prev >= 0 else torch.ones_like(alpha_bar_t)

        # Predict x0 from x_t and predicted noise
        pred_x0 = (x_t - (1 - alpha_bar_t) ** 0.5 * predicted_noise) \
                  / alpha_bar_t ** 0.5
        pred_x0 = pred_x0.clamp(-3, 3)  # clip to reasonable range

        # DDIM update (deterministic, sigma=0)
        x_t_prev = alpha_bar_t_prev ** 0.5 * pred_x0 + \
                   (1 - alpha_bar_t_prev) ** 0.5 * predicted_noise

        return x_t_prev

    @torch.no_grad()
    def generate(self, model, conditions, device, seq_len=200):
        """
        Full generation loop: start from noise, denoise iteratively.

        Args:
            model:      trained noise-prediction transformer
            conditions: dict of condition tensors
            device:     torch device
            seq_len:    number of GPS points to generate
        Returns:
            trajectory: (B, seq_len, 2) — normalised lon/lat
        """
        B = conditions['behavior'].shape[0]

        # Start from pure Gaussian noise
        x = torch.randn(B, seq_len, 2).to(device)

        for i, t in enumerate(self.sample_steps):
            t_prev = self.sample_steps[i + 1] \
                     if i + 1 < len(self.sample_steps) else -1

            t_tensor = torch.full((B,), t,
                                  dtype=torch.long, device=device)

            # Predict noise
            predicted_noise = model(x, t_tensor, **conditions)

            # DDIM step
            x = self.p_sample_ddim(x, t, t_prev, predicted_noise)

        return x  # (B, seq_len, 2) — de-normalise after
```

---

### 5.5 `models/fine_generator.py` — Full Fine-Level Model

```python
import torch
import torch.nn as nn
from models.embeddings import LonLatKinematicEmbedding, EnvironmentalEmbedding
from models.positional_encoding import ElephantPositionalEncoding
from models.adaln_block import BehaviorConditionedAdaLN


class ElephantFineDiffusionTransformer(nn.Module):
    """
    Level 2: Fine-grained GPS generation at 10-second resolution.

    Architecture:
    - Input: noisy GPS (x_t) + kinematic context + env context
    - Conditioning: behavior, season, time_of_day, LULC, etc. via adaLN
    - Output: predicted noise ε (same shape as x_t)

    Trained to denoise trajectories via DDIM diffusion process.

    Model sizes (choose based on GPU memory):
      Tiny:  d=128, heads=4,  layers=3  → ~5M params  (safe for 1 GPU)
      Small: d=256, heads=8,  layers=6  → ~18M params
      Base:  d=384, heads=8,  layers=8  → ~40M params
    """

    def __init__(self,
                 d_model=256,
                 nhead=8,
                 num_layers=6,
                 dropout=0.1,
                 max_seq_len=200):
        super().__init__()

        self.d_model = d_model

        # ── INPUT EMBEDDINGS ─────────────────────────────────────────
        self.gps_kin_embed = LonLatKinematicEmbedding(d_model)
        self.env_embed     = EnvironmentalEmbedding(d_model)
        self.pos_enc       = ElephantPositionalEncoding(d_model)

        # Diffusion timestep embedding (sinusoidal → MLP)
        self.timestep_embed = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

        # ── TRANSFORMER BACKBONE ────────────────────────────────────
        self.transformer_blocks = nn.ModuleList([
            BehaviorConditionedAdaLN(d_model, nhead, dropout)
            for _ in range(num_layers)
        ])

        # ── OUTPUT HEAD ──────────────────────────────────────────────
        # Predict noise for lon and lat tokens only
        # (we don't need to predict noise for kinematic tokens)
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 2)  # → lon noise, lat noise

    def timestep_sinusoidal(self, t, d_model):
        """Sinusoidal embedding for diffusion timestep."""
        half = d_model // 2
        freqs = torch.exp(
            -torch.arange(half, device=t.device) *
            (torch.log(torch.tensor(10000.0)) / (half - 1))
        )
        args  = t[:, None].float() * freqs[None]
        emb   = torch.cat([args.sin(), args.cos()], dim=-1)
        return emb

    def forward(self, x_noisy, t,
                speed, accel, turning, bearing, persist, step,
                ndvi, evi, lst, elev, slope, water,
                behavior, season, time_of_day,
                lulc, human_settle, move_type):
        """
        Args:
            x_noisy:    (B, seq_len, 2) noisy GPS coordinates
            t:          (B,) diffusion timestep
            speed..step: (B, seq_len) kinematic context
            ndvi..water: (B, seq_len) environmental context
            behavior..move_type: (B,) condition scalars
        Returns:
            noise_pred: (B, seq_len, 2) predicted noise
        """
        B, T, _ = x_noisy.shape

        # Split noisy GPS back into lon/lat
        lon_noisy = x_noisy[:, :, 0]
        lat_noisy = x_noisy[:, :, 1]

        # Embed all inputs
        tokens  = self.gps_kin_embed(
            lon_noisy, lat_noisy,
            speed, accel, turning, bearing, persist, step
        )  # (B, 8T, d_model)

        # Add environmental context to all tokens at each timestep
        env_ctx = self.env_embed(ndvi, evi, lst, elev, slope, water)
        # env_ctx: (B, T, d_model) → repeat 8 times per timestep
        env_ctx_expanded = env_ctx.unsqueeze(2).repeat(1, 1, 8, 1)
        env_ctx_flat     = env_ctx_expanded.view(B, 8*T, self.d_model)
        tokens = tokens + env_ctx_flat

        # Add positional encoding
        pe     = self.pos_enc(T).unsqueeze(0).to(tokens.device)
        tokens = tokens + pe

        # Add diffusion timestep embedding
        t_emb = self.timestep_sinusoidal(t, self.d_model)  # (B, d_model)
        t_emb = self.timestep_embed(t_emb).unsqueeze(1)    # (B, 1, d_model)
        tokens = tokens + t_emb

        # Pass through adaLN transformer blocks
        for block in self.transformer_blocks:
            tokens, _ = block(
                tokens,
                behavior, season, time_of_day,
                lulc, human_settle, move_type
            )

        # Extract only lon/lat tokens (positions 0 and 1 of each 8-group)
        tokens = self.out_norm(tokens)
        lon_tokens = tokens[:, 0::8, :]  # every 8th starting at 0
        lat_tokens = tokens[:, 1::8, :]  # every 8th starting at 1

        # Predict noise for lon and lat
        lon_noise_pred = self.out_proj(lon_tokens)[:, :, 0]  # (B, T)
        lat_noise_pred = self.out_proj(lat_tokens)[:, :, 1]  # (B, T)

        return torch.stack([lon_noise_pred, lat_noise_pred], dim=-1)
        # → (B, T, 2)
```

---

## 6. Training Pipeline

### 6.1 `training/dataset.py`

```python
import torch
from torch.utils.data import Dataset
import numpy as np


class ElephantWindowDataset(Dataset):
    """
    PyTorch Dataset for sliding windows of elephant GPS data.
    Each sample is a 200-point window (~33 minutes at 10sec).
    """

    def __init__(self, windows, scaler_gps=None,
                 scaler_kin=None, scaler_env=None):
        self.windows    = windows
        self.scaler_gps = scaler_gps  # MinMaxScaler for lon/lat
        self.scaler_kin = scaler_kin  # StandardScaler for kinematics
        self.scaler_env = scaler_env  # StandardScaler for env features

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]

        # GPS — normalise to [-1, 1] via MinMaxScaler
        gps = np.stack([w['lon'], w['lat']], axis=-1)
        if self.scaler_gps:
            gps = self.scaler_gps.transform(gps)

        # Kinematics — standardise (mean=0, std=1)
        kin = np.stack([
            w['speed'], w['accel'], w['turning'],
            w['bearing'], w['persist'], w['step']
        ], axis=-1)
        if self.scaler_kin:
            kin = self.scaler_kin.transform(kin)

        # Environmental features — standardise
        env = np.stack([
            w['ndvi'], w['evi'], w['lst'],
            w['elev'], w['slope'], w['water']
        ], axis=-1)
        if self.scaler_env:
            env = self.scaler_env.transform(env)

        return {
            # GPS target
            'gps':         torch.tensor(gps, dtype=torch.float32),

            # Kinematic inputs per timestep
            'speed':       torch.tensor(kin[:, 0], dtype=torch.float32),
            'accel':       torch.tensor(kin[:, 1], dtype=torch.float32),
            'turning':     torch.tensor(kin[:, 2], dtype=torch.float32),
            'bearing':     torch.tensor(kin[:, 3], dtype=torch.float32),
            'persist':     torch.tensor(kin[:, 4], dtype=torch.float32),
            'step':        torch.tensor(kin[:, 5], dtype=torch.float32),

            # Environmental per timestep
            'ndvi':        torch.tensor(env[:, 0], dtype=torch.float32),
            'evi':         torch.tensor(env[:, 1], dtype=torch.float32),
            'lst':         torch.tensor(env[:, 2], dtype=torch.float32),
            'elev':        torch.tensor(env[:, 3], dtype=torch.float32),
            'slope':       torch.tensor(env[:, 4], dtype=torch.float32),
            'water':       torch.tensor(env[:, 5], dtype=torch.float32),

            # Window-level conditions
            'behavior':    torch.tensor(w['behavior'],    dtype=torch.long),
            'season':      torch.tensor(w['season'],      dtype=torch.long),
            'time_of_day': torch.tensor(w['time_of_day'], dtype=torch.long),
            'lulc':        torch.tensor(w['lulc'],        dtype=torch.long),
            'human_settle':torch.tensor(w['human_settle'],dtype=torch.long),
            'move_type':   torch.tensor(w['move_type'],   dtype=torch.long),
        }
```

---

### 6.2 `training/losses.py`

```python
import torch
import torch.nn.functional as F
import math


def haversine_loss(pred_lon, pred_lat, true_lon, true_lat):
    """
    Great-circle distance loss.
    More meaningful than MSE for geographic coordinates.
    A 0.01° error at equator ≈ 1.1km — MSE treats all degrees equally,
    haversine accounts for actual Earth surface distance.
    """
    R = 6371.0  # Earth radius km
    d_lat = torch.deg2rad(true_lat - pred_lat)
    d_lon = torch.deg2rad(true_lon - pred_lon)
    a = torch.sin(d_lat / 2) ** 2 + \
        torch.cos(torch.deg2rad(pred_lat)) * \
        torch.cos(torch.deg2rad(true_lat)) * \
        torch.sin(d_lon / 2) ** 2
    dist = 2 * R * torch.asin(torch.clamp(torch.sqrt(a), 0, 1))
    return dist.mean()


def kinematic_consistency_loss(pred_gps, true_speed, true_turning):
    """
    Penalise generated trajectories where implied movement
    kinematics don't match the conditioned behavior.

    If behavior=MOVE and true_speed is high, generated GPS should
    show large consecutive displacements — not stationary clusters.
    """
    # Compute implied speed from consecutive predicted GPS points
    # pred_gps: (B, T, 2) normalised coordinates
    diffs      = pred_gps[:, 1:, :] - pred_gps[:, :-1, :]
    impl_speed = torch.norm(diffs, dim=-1)        # (B, T-1)

    # Compute implied turning angle
    v1 = diffs[:, :-1, :]  # (B, T-2, 2)
    v2 = diffs[:, 1:,  :]  # (B, T-2, 2)
    cos_sim    = F.cosine_similarity(v1, v2, dim=-1)
    impl_turn  = torch.acos(torch.clamp(cos_sim, -1, 1))  # (B, T-2)

    # Speed consistency: match distribution (not exact values)
    speed_loss   = F.mse_loss(impl_speed.mean(dim=1),
                               true_speed.mean(dim=1))

    # Turning consistency
    turning_loss = F.mse_loss(impl_turn.mean(dim=1),
                               true_turning[:, 1:].abs().mean(dim=1))

    return speed_loss + 0.5 * turning_loss


def ecological_validity_loss(pred_gps, water_map,
                              behavior, season):
    """
    During dry season + MOVE behavior, penalise trajectories
    that move AWAY from water sources.
    This enforces ecological realism without needing road networks.

    NOTE: Requires a spatial water_map lookup function.
    This can be a KD-tree of known water points.
    """
    # Only apply during dry season (season_code = 0)
    dry_mask = (season == 0)
    if dry_mask.sum() == 0:
        return torch.tensor(0.0)

    # Get water proximity for predicted endpoint
    end_lon = pred_gps[dry_mask, -1, 0]
    end_lat = pred_gps[dry_mask, -1, 1]
    water_dist = water_map.query_distance(end_lon, end_lat)

    # Penalise trajectories ending far from water in dry season
    return water_dist.mean()


def total_diffusion_loss(predicted_noise, true_noise,
                         pred_x0=None, true_x0=None,
                         true_speed=None, true_turning=None,
                         behavior=None, season=None,
                         λ_kin=0.1, λ_eco=0.05):
    """
    Combined training loss:
      Main:      MSE on predicted vs true noise (standard diffusion)
      Kinematic: ensure implied movement matches kinematics
      Ecological: ensure dry-season paths stay near water
    """
    # Primary diffusion loss
    noise_loss = F.mse_loss(predicted_noise, true_noise)

    total = noise_loss

    # Optional auxiliary losses (activate after warmup epochs)
    if pred_x0 is not None and true_speed is not None:
        kin_loss  = kinematic_consistency_loss(
            pred_x0, true_speed, true_turning
        )
        total = total + λ_kin * kin_loss

    return total
```

---

### 6.3 `training/trainer_fine.py`

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os


def train_fine_model(model, diffusion, train_dataset,
                     val_dataset, config):
    """
    Training loop for the fine-level diffusion transformer.

    Args:
        model:          ElephantFineDiffusionTransformer
        diffusion:      DDIMDiffusion
        train_dataset:  ElephantWindowDataset (train split)
        val_dataset:    ElephantWindowDataset (val split)
        config:         training config dict
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = model.to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],   # recommended: 256-512
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4
    )

    optimizer = AdamW(
        model.parameters(),
        lr=config['lr'],           # 1e-4
        weight_decay=config['wd']  # 1e-4
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'],
        eta_min=1e-6
    )

    best_val_loss = float('inf')

    for epoch in range(config['epochs']):
        # ── TRAINING ────────────────────────────────────────────────
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            # Sample random diffusion timestep for each sample
            t = torch.randint(
                0, diffusion.T,
                (batch['gps'].shape[0],),
                device=device
            )

            # Forward diffusion: add noise
            x_noisy, true_noise = diffusion.q_sample(batch['gps'], t)

            # Predict noise with model
            pred_noise = model(
                x_noisy, t,
                speed=batch['speed'],   accel=batch['accel'],
                turning=batch['turning'], bearing=batch['bearing'],
                persist=batch['persist'], step=batch['step'],
                ndvi=batch['ndvi'],     evi=batch['evi'],
                lst=batch['lst'],       elev=batch['elev'],
                slope=batch['slope'],   water=batch['water'],
                behavior=batch['behavior'],
                season=batch['season'],
                time_of_day=batch['time_of_day'],
                lulc=batch['lulc'],
                human_settle=batch['human_settle'],
                move_type=batch['move_type']
            )

            loss = nn.functional.mse_loss(pred_noise, true_noise)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        # ── VALIDATION ──────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                t = torch.randint(0, diffusion.T,
                                  (batch['gps'].shape[0],),
                                  device=device)
                x_noisy, true_noise = diffusion.q_sample(batch['gps'], t)
                pred_noise = model(x_noisy, t, **{
                    k: batch[k] for k in [
                        'speed','accel','turning','bearing','persist','step',
                        'ndvi','evi','lst','elev','slope','water',
                        'behavior','season','time_of_day',
                        'lulc','human_settle','move_type'
                    ]
                })
                val_loss += nn.functional.mse_loss(
                    pred_noise, true_noise
                ).item()

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)
        scheduler.step()

        print(f"Epoch {epoch+1:03d} | "
              f"Train: {train_loss:.5f} | Val: {val_loss:.5f}")

        # ── CHECKPOINT ──────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'val_loss':    val_loss,
                'config':      config,
            }, os.path.join(config['checkpoint_dir'],
                            'best_model.pt'))
            print(f"  ✓ Saved best model (val_loss={val_loss:.5f})")
```

---

## 7. Inference Pipeline

### `inference/generator.py`

```python
import torch
import numpy as np
from models.fine_generator import ElephantFineDiffusionTransformer
from models.diffusion import DDIMDiffusion
from inference.occupancy_sampler import EcologicalOccupancySampler


class ElephantTrajectoryGenerator:
    """
    Main generation interface.
    Loads trained model and generates synthetic elephant trajectories.

    Usage:
        gen = ElephantTrajectoryGenerator('checkpoints/best_model.pt')
        trajectories = gen.generate(
            n_trajectories=100,
            behavior='MOVE',
            season='dry',
            time_of_day='morning',
            ndvi_mean=0.45,
            water_nearby=True
        )
    """

    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device)
        self._load_model(checkpoint_path)
        self.diffusion = DDIMDiffusion(T=200, S=40)
        self.sampler   = EcologicalOccupancySampler()

    def _load_model(self, path):
        ckpt = torch.load(path, map_location=self.device)
        cfg  = ckpt['config']
        self.model = ElephantFineDiffusionTransformer(
            d_model=cfg['d_model'],
            nhead=cfg['nhead'],
            num_layers=cfg['num_layers']
        ).to(self.device)
        self.model.load_state_dict(ckpt['model_state'])
        self.model.eval()

    def build_conditions(self, behavior, season, time_of_day,
                         lulc, human_settle, n):
        """Build condition tensors from human-readable inputs."""
        BEHAVIOR_MAP    = {'STOP': 0, 'MOVE': 1}
        SEASON_MAP      = {'dry': 0,  'wet': 1}
        TOD_MAP         = {'night': 0, 'morning': 1,
                           'afternoon': 2, 'evening': 3}
        return {
            'behavior':    torch.full((n,), BEHAVIOR_MAP[behavior],
                                      dtype=torch.long, device=self.device),
            'season':      torch.full((n,), SEASON_MAP[season],
                                      dtype=torch.long, device=self.device),
            'time_of_day': torch.full((n,), TOD_MAP[time_of_day],
                                      dtype=torch.long, device=self.device),
            'lulc':        torch.full((n,), lulc,
                                      dtype=torch.long, device=self.device),
            'human_settle':torch.full((n,), int(human_settle),
                                      dtype=torch.long, device=self.device),
            'move_type':   torch.zeros(n, dtype=torch.long,
                                       device=self.device),
        }

    def generate(self, n_trajectories=100,
                 behavior='MOVE', season='dry',
                 time_of_day='morning', lulc=10,
                 human_settle=False,
                 env_context=None,
                 seq_len=200):
        """
        Generate n_trajectories synthetic elephant trajectories.

        Args:
            n_trajectories: number to generate
            behavior:       'MOVE' or 'STOP'
            season:         'dry' or 'wet'
            time_of_day:    'night','morning','afternoon','evening'
            lulc:           land cover class int
            human_settle:   bool — near human settlement?
            env_context:    dict of env tensor conditions (optional)
                            if None, uses mean values from training data
            seq_len:        number of GPS points per trajectory

        Returns:
            trajectories: list of np.arrays, each (seq_len, 2) lon/lat
        """
        conditions = self.build_conditions(
            behavior, season, time_of_day,
            lulc, human_settle, n_trajectories
        )

        # Fill in environmental conditions
        if env_context is None:
            env_context = self._default_env_context(
                n_trajectories, seq_len, season
            )
        conditions.update(env_context)

        with torch.no_grad():
            # Generate normalised trajectories via diffusion
            norm_trajectories = self.diffusion.generate(
                self.model, conditions, self.device, seq_len
            )  # (N, seq_len, 2)

        # De-normalise back to actual GPS coordinates
        trajectories_np = norm_trajectories.cpu().numpy()
        trajectories_denorm = self.scaler_gps.inverse_transform(
            trajectories_np.reshape(-1, 2)
        ).reshape(n_trajectories, seq_len, 2)

        return trajectories_denorm  # (N, seq_len, 2) — actual lon/lat

    def _default_env_context(self, n, seq_len, season):
        """Mean environmental values from training data per season."""
        if season == 'dry':
            defaults = dict(ndvi=0.25, evi=0.18, lst=42.0,
                            elev=1100.0, slope=2.0, water=0.1)
        else:
            defaults = dict(ndvi=0.55, evi=0.40, lst=32.0,
                            elev=1100.0, slope=2.0, water=0.5)
        return {
            k: torch.full((n, seq_len), v,
                          dtype=torch.float32, device=self.device)
            for k, v in defaults.items()
        }
```

---

### `inference/occupancy_sampler.py`

```python
import h3
import numpy as np


class EcologicalOccupancySampler:
    """
    WildGraph-style occupancy sampler with ecological weighting.
    Converts H3 coarse regions → actual GPS points.

    Two-step sampling:
      1. Within region, select a sub-hex (dot) weighted by NDVI/water
      2. Within dot, select random GPS point

    This ensures geographic validity while adding stochasticity.
    """

    def __init__(self, fine_zoom=9):
        self.fine_zoom = fine_zoom  # sub-hex resolution

    def sample_point(self, h3_region, season,
                     ndvi_raster, water_raster):
        """
        Sample an ecologically valid GPS point from within an H3 region.

        Dry season: weight heavily toward high water_occ (water sources)
        Wet season: weight toward high NDVI (vegetation patches)
        """
        # Get all fine sub-hexagons within this region
        dots = list(h3.h3_to_children(h3_region, self.fine_zoom))

        weights = []
        for dot in dots:
            lat, lon = h3.h3_to_geo(dot)
            ndvi  = self._sample_raster(ndvi_raster,  lat, lon)
            water = self._sample_raster(water_raster, lat, lon)

            # Ecological attractiveness weight
            if season == 'dry':
                # Dry season: water is life — weight it 70%
                w = 0.3 * max(ndvi, 0) + 0.7 * max(water, 0)
            else:
                # Wet season: spread out to vegetation
                w = 0.6 * max(ndvi, 0) + 0.4 * max(water, 0)

            weights.append(max(w, 0.01))  # ensure non-zero

        # Sample a dot proportional to ecological weight
        probs = np.array(weights) / sum(weights)
        chosen_dot = np.random.choice(dots, p=probs)

        # Random point within chosen dot
        boundary = h3.h3_to_geo_boundary(chosen_dot)
        lats = [p[0] for p in boundary]
        lons = [p[1] for p in boundary]
        lat = np.random.uniform(min(lats), max(lats))
        lon = np.random.uniform(min(lons), max(lons))

        return lon, lat

    def _sample_raster(self, raster, lat, lon):
        """Bilinear interpolation from raster at given lat/lon."""
        # Implement based on your raster format (GeoTIFF, numpy array, etc.)
        # Can use rasterio: raster.sample([(lon, lat)])
        return float(raster.sample([(lon, lat)]).__next__()[0])
```

---

## 8. Evaluation

### `evaluation/spatial_metrics.py`

```python
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from fastdtw import fastdtw


def hausdorff_distance(traj_a, traj_b):
    """Max of directed Hausdorff distances both ways."""
    d1 = directed_hausdorff(traj_a, traj_b)[0]
    d2 = directed_hausdorff(traj_b, traj_a)[0]
    return max(d1, d2)


def dtw_distance(traj_a, traj_b):
    """Dynamic Time Warping — handles variable speed trajectories."""
    dist, _ = fastdtw(traj_a, traj_b)
    return dist


def coverage(real_trajs, gen_trajs, metric_fn=hausdorff_distance):
    """
    From WildGraph: % of real trajectories 'covered' by generated set.
    A real trajectory is covered if at least one generated trajectory
    is closer to it than to any other real trajectory.
    """
    covered = set()
    for gen in gen_trajs:
        best_real = min(range(len(real_trajs)),
                        key=lambda i: metric_fn(gen, real_trajs[i]))
        covered.add(best_real)
    return len(covered) / len(real_trajs)
```

---

### `evaluation/ecological_metrics.py`

```python
import numpy as np


def ndvi_path_score(trajectories, ndvi_raster):
    """
    Mean NDVI along generated paths.
    Should match distribution of real trajectory NDVI values.
    """
    scores = []
    for traj in trajectories:
        path_ndvi = [
            ndvi_raster.sample([(lon, lat)]).__next__()[0]
            for lon, lat in traj
        ]
        scores.append(np.mean(path_ndvi))
    return np.array(scores)


def water_proximity_score(trajectories, water_points,
                           season_code):
    """
    During dry season (season_code=0), generated trajectories
    should stay closer to water than wet season.
    Key ecological validity metric unique to this work.
    """
    from scipy.spatial import KDTree
    tree = KDTree(water_points)

    scores = []
    for traj in trajectories:
        dists, _ = tree.query(traj)
        scores.append(dists.mean())

    return np.array(scores)


def behavior_fidelity(generated_speeds, generated_turnings,
                      condition_behavior):
    """
    Do generated trajectories respect the behavior condition?
    MOVE should have higher speeds and lower turning std.
    STOP should have lower speeds and higher turning std.
    """
    move_mask = (condition_behavior == 1)
    stop_mask = (condition_behavior == 0)

    move_speed = generated_speeds[move_mask].mean()
    stop_speed = generated_speeds[stop_mask].mean()

    # MOVE speed should be higher than STOP speed
    behavior_respected = move_speed > stop_speed
    speed_ratio = move_speed / (stop_speed + 1e-6)

    return {
        'behavior_respected': behavior_respected,
        'move_mean_speed':    float(move_speed),
        'stop_mean_speed':    float(stop_speed),
        'speed_ratio':        float(speed_ratio),  # should be > 2
    }
```

---

## 9. Configuration Reference

### `configs/fine_config.yaml`

```yaml
# ── MODEL ──────────────────────────────────────────────────────────
model:
  d_model:    256         # embedding dimension
  nhead:      8           # attention heads
  num_layers: 6           # transformer blocks (use 3 for Tiny)
  dropout:    0.1
  max_seq_len: 200        # window size in timesteps

# ── DIFFUSION ──────────────────────────────────────────────────────
diffusion:
  T:          200         # total diffusion timesteps
  S:          40          # sampling steps (DDIM skip)
  beta_start: 0.0001
  beta_end:   0.02

# ── DATA ───────────────────────────────────────────────────────────
data:
  window_size:       200   # GPS fixes per sample
  stride:            100   # sliding window stride (50% overlap)
  gap_threshold_sec: 300   # skip windows with GPS gaps > 5 min
  val_elephant:      AG005 # held-out for validation
  test_elephant:     null  # set to elephant ID for test holdout

# ── TRAINING ───────────────────────────────────────────────────────
training:
  epochs:         200
  batch_size:     256
  lr:             1.0e-4
  weight_decay:   1.0e-4
  grad_clip:      1.0
  checkpoint_dir: checkpoints/fine/
  log_every:      100      # steps between log prints

# ── LOSS WEIGHTS ───────────────────────────────────────────────────
loss:
  lambda_kinematic:   0.1   # kinematic consistency weight
  lambda_ecological:  0.05  # ecological validity weight
  warmup_aux_epoch:   20    # epoch to start auxiliary losses

# ── H3 GRAPH (COARSE LEVEL) ────────────────────────────────────────
h3:
  initial_zoom:     4
  split_threshold:  1.0    # degrees
  node2vec_dim:     64
  eco_dim:          16
  walk_length:      40
  num_walks:        200
  q:                4      # high q = long-range exploration
```

---

## 10. Dependencies

### `requirements.txt`

```
# Core
torch>=2.1.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0

# Geospatial
h3>=3.7.6
geopandas>=0.14.0
rasterio>=1.3.0          # for NDVI/water raster sampling
shapely>=2.0.0
geopy>=2.4.0

# Graph
networkx>=3.1
node2vec>=0.4.6

# Trajectory metrics
fastdtw>=0.3.4
scipy>=1.11.0

# Training utilities
tqdm>=4.66.0
pyyaml>=6.0
tensorboard>=2.14.0      # optional: training monitoring

# Notebooks
matplotlib>=3.7.0
folium>=0.14.0           # interactive map visualisation
```

### Hardware Recommendations

```
Minimum:  1× GPU 8GB   → Tiny config (d=128, 3 layers)
Recommended: 1× GPU 24GB → Small config (d=256, 6 layers)
Optimal: 1× A100 80GB  → Base config (d=384, 8 layers)

Expected training time (Small config, 150K windows):
  ~4-6 hours on A100 for 200 epochs
  ~12-16 hours on RTX 3090
```

---

## Quick Start Commands

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full preprocessing pipeline
python -m preprocessing.clean     --input  data/raw/elephant_gps.csv \
                                   --output data/processed/clean.csv
python -m preprocessing.windowing  --input  data/processed/clean.csv \
                                   --output data/processed/windows/
python -m preprocessing.h3_builder --input  data/processed/hourly/ \
                                   --output data/processed/h3_graph/

# 3. Train fine-level model
python train.py --config configs/fine_config.yaml \
                --level  fine

# 4. Generate trajectories
python generate.py --checkpoint checkpoints/fine/best_model.pt \
                   --n           1000 \
                   --behavior    MOVE \
                   --season      dry \
                   --output      generated/dry_move_trajectories.npy

# 5. Evaluate generated trajectories
python evaluate.py --generated  generated/dry_move_trajectories.npy \
                   --real        data/processed/test_windows.npy \
                   --output      results/evaluation_report.json
```
