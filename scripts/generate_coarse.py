"""
Generate a coarse H3-cell trajectory using only the Level-1 CoarseGenerator.

The model autoregressively predicts the next H3 region one step at a time.
Each predicted node ID is converted to its H3 hex center (lat, lon) and the
hexagon boundary polygon for map visualisation.

Usage:
    python scripts/generate_coarse.py
    python scripts/generate_coarse.py --steps 72 --behavior MOVE --season dry --output results/coarse_traj.npy
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import torch
import h3

# Make the package importable when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from elephantgraph.models.coarse_generator import CoarseGenerator

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")

BEHAVIOR_MAP = {"STOP": 0, "MOVE": 1}
SEASON_MAP   = {"dry": 0, "wet": 1}


# ---------------------------------------------------------------------------
# H3 mapping helpers
# ---------------------------------------------------------------------------

def build_h3_mapping(hourly_csv_path, h3_graph_dir):
    """Rebuild the same h3_to_id mapping used by CoarseWindowDataset."""
    from elephantgraph.training.dataset_coarse import load_h3_zoom
    zoom = load_h3_zoom(h3_graph_dir)
    df = pd.read_csv(hourly_csv_path, parse_dates=["timestamp"])
    df["h3_idx"] = df.apply(
        lambda r: h3.latlng_to_cell(r["latitude"], r["longitude"], zoom), axis=1
    )
    all_h3 = sorted(df["h3_idx"].unique())
    h3_to_id = {h: i for i, h in enumerate(all_h3)}
    id_to_h3 = {i: h for h, i in h3_to_id.items()}
    print(f"  H3 zoom={zoom}  ({len(all_h3)} unique cells)")
    return h3_to_id, id_to_h3, df


def most_common_start_node(df, h3_to_id):
    """Return the integer node ID of the most frequently visited H3 cell."""
    most_common_h3 = df["h3_idx"].value_counts().index[0]
    return h3_to_id[most_common_h3]


# ---------------------------------------------------------------------------
# Autoregressive generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_coarse_sequence(
    model,
    node_embeddings,
    seed_node_id,
    steps,
    behavior_code,
    season_code,
    context_len,
    temperature,
    device,
):
    """
    Autoregressively generate a sequence of H3 node IDs.

    At each step:
      - feed the last `context_len` nodes into the model
      - take logits from the final position (next-step prediction)
      - sample the next node (temperature-scaled softmax)
      - append and repeat

    Args:
        seed_node_id:  integer node ID to start from
        steps:         number of H3 cells to generate
        context_len:   how many past nodes to feed as context (≤ window_size-1)
        temperature:   >1 = more random, <1 = more confident

    Returns:
        list of integer node IDs, length = steps + 1 (includes seed)
    """
    model.eval()
    node_embeddings = node_embeddings.to(device)

    behavior = torch.tensor([behavior_code], dtype=torch.long, device=device)
    season   = torch.tensor([season_code],   dtype=torch.long, device=device)

    generated   = [seed_node_id]
    dwell_hours = []          # hours spent in each visited cell
    current_dwell = 1

    # We run for up to max_raw_steps hourly ticks, but only count
    # a "step" when the cell actually changes. This reflects the real
    # 90% self-loop rate at zoom 6 without boring the output with
    # identical consecutive cells.
    max_raw_steps = steps * 50   # budget: up to 50 raw ticks per desired step

    raw_count = 0
    transitions = 0

    while transitions < steps and raw_count < max_raw_steps:
        ctx = generated[-context_len:]
        node_indices = torch.tensor(
            np.array([ctx], dtype=np.int64), dtype=torch.long, device=device
        )

        logits, _ = model(node_indices, node_embeddings, season, behavior)
        next_logits = logits[0, -1, :] / temperature
        probs       = torch.softmax(next_logits, dim=-1)
        next_node   = torch.multinomial(probs, num_samples=1).item()

        raw_count += 1

        if next_node == generated[-1]:
            current_dwell += 1          # stayed — just count the hour
        else:
            dwell_hours.append(current_dwell)
            generated.append(next_node)
            current_dwell = 1
            transitions += 1

    dwell_hours.append(current_dwell)   # dwell for the final cell

    print(f"  Raw ticks simulated : {raw_count} hours")
    print(f"  Cell transitions    : {transitions}")
    if raw_count > 0:
        print(f"  Avg dwell per cell  : {raw_count / max(transitions, 1):.1f} hours")

    return generated, dwell_hours


# ---------------------------------------------------------------------------
# Convert node ID sequence → GPS + hex boundaries
# ---------------------------------------------------------------------------

def node_ids_to_geodata(node_id_seq, id_to_h3):
    """
    Convert a list of integer node IDs to:
      - centers:    np.ndarray (N, 2) of (lon, lat) hex centres
      - boundaries: list of N polygon coords [(lat, lon), ...] for folium

    Consecutive duplicate nodes (elephant stayed in same cell) are kept
    so dwell time is visible on the map.
    """
    centers    = []
    boundaries = []
    h3_seq     = []

    for nid in node_id_seq:
        hex_str = id_to_h3[nid]
        lat, lon = h3.cell_to_latlng(hex_str)
        centers.append([lon, lat])

        boundary_latlon = h3.cell_to_boundary(hex_str)   # list of (lat, lon)
        boundaries.append(boundary_latlon)
        h3_seq.append(hex_str)

    return np.array(centers, dtype=np.float32), boundaries, h3_seq


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate a coarse H3 trajectory with the Level-1 model"
    )
    parser.add_argument("--checkpoint",  default=os.path.join(PROJECT_DIR, "checkpoints", "coarse", "best_coarse_model.pt"))
    parser.add_argument("--hourly-csv",  default=os.path.join(PROJECT_DIR, "elephantgraph", "data", "processed", "hourly", "hourly.csv"))
    parser.add_argument("--node-emb",    default=os.path.join(PROJECT_DIR, "elephantgraph", "data", "processed", "h3_graph", "node_embeddings.npy"))
    parser.add_argument("--steps",       type=int,   default=24,     help="Number of distinct cell transitions to generate")
    parser.add_argument("--context-len", type=int,   default=23,     help="Past context window fed to model")
    parser.add_argument("--behavior",    default="MOVE", choices=["MOVE", "STOP"])
    parser.add_argument("--season",      default="dry",  choices=["dry", "wet"])
    parser.add_argument("--temperature", type=float, default=1.0,    help="Sampling temperature (lower = more deterministic)")
    parser.add_argument("--seed-node",   type=int,   default=None,   help="Start node ID (default: most common)")
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--output",      default=os.path.join(PROJECT_DIR, "results", "coarse_trajectory.npz"),
                        help="Output .npz file (centers + boundaries)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load H3 mapping ─────────────────────────────────────────────────────
    print("Building H3 mapping from hourly CSV ...")
    h3_to_id, id_to_h3, hourly_df = build_h3_mapping(args.hourly_csv, args.node_emb.rsplit(os.sep, 1)[0])
    num_h3_nodes = len(h3_to_id)
    print(f"  {num_h3_nodes} unique H3 cells")

    # ── Load node embeddings ─────────────────────────────────────────────────
    emb_dict = np.load(args.node_emb, allow_pickle=True).item()
    embed_dim = next(iter(emb_dict.values())).shape[0]
    node_embeddings = torch.zeros(num_h3_nodes, embed_dim)
    for hex_str, emb in emb_dict.items():
        if hex_str in h3_to_id:
            node_embeddings[h3_to_id[hex_str]] = torch.tensor(emb, dtype=torch.float32)

    # ── Load model ───────────────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ms   = ckpt["model_state"]
    d_model    = ms["encoder.node_embed.weight"].shape[0]
    num_layers = sum(1 for k in ms if k.startswith("encoder.encoder.layers.")
                     and k.endswith(".norm1.weight"))
    # nhead: infer from in_proj_weight shape (3*d_model, d_model) → nhead=d_model/head_dim
    # Use default 8 — head_dim=16 for d_model=128 is valid
    nhead = 8 if d_model % 8 == 0 else 4

    # Seasonal latent dictionary dims (auto-detected from checkpoint)
    if "latent_dict.codebook" in ms:
        n_seasons, n_latent, _ = ms["latent_dict.codebook"].shape
    else:
        n_seasons, n_latent = 2, 16

    print(f"  d_model={d_model}  nhead={nhead}  num_layers={num_layers}  num_h3_nodes={num_h3_nodes}")
    if "latent_dict.codebook" in ms:
        print(f"  seasonal latent dictionary: {n_seasons} seasons x {n_latent} entries")

    model = CoarseGenerator(
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        num_h3_nodes=num_h3_nodes,
        num_latent_entries=n_latent,
        num_seasons=n_seasons,
    ).to(device)
    model.load_state_dict(ms)
    model.eval()

    # ── Seed node ────────────────────────────────────────────────────────────
    seed = args.seed_node
    if seed is None:
        seed = most_common_start_node(hourly_df, h3_to_id)
    print(f"Seed node: {seed} ({id_to_h3[seed]})")

    # ── Generate ─────────────────────────────────────────────────────────────
    print(f"Generating {args.steps} cell transitions  "
          f"[behavior={args.behavior}  season={args.season}  T={args.temperature}] ...")
    node_seq, dwell_hours = generate_coarse_sequence(
        model=model,
        node_embeddings=node_embeddings,
        seed_node_id=seed,
        steps=args.steps,
        behavior_code=BEHAVIOR_MAP[args.behavior],
        season_code=SEASON_MAP[args.season],
        context_len=args.context_len,
        temperature=args.temperature,
        device=device,
    )

    # ── Convert to GPS ────────────────────────────────────────────────────────
    centers, boundaries, h3_seq = node_ids_to_geodata(node_seq, id_to_h3)

    print(f"Distinct cells visited : {len(node_seq)}  ({len(set(node_seq))} unique)")
    print(f"Dwell times (hours)    : {dwell_hours}")
    print(f"Lon range: [{centers[:,0].min():.4f}, {centers[:,0].max():.4f}]")
    print(f"Lat range: [{centers[:,1].min():.4f}, {centers[:,1].max():.4f}]")

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez(
        args.output,
        centers=centers,
        node_ids=np.array(node_seq),
        h3_strings=np.array(h3_seq),
        boundaries=np.array(boundaries, dtype=object),
        dwell_hours=np.array(dwell_hours, dtype=np.int32),
    )
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
