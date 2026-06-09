"""
End-to-end ElephantGraph trajectory generation.

Pipeline
--------
Level 1 — Coarse (H3 Transformer):
    Autoregressively generate a sequence of H3 cell waypoints and the number
    of hours spent in each cell (dwell time).

Level 2 — Fine (DDIM Diffusion Transformer):
    For every consecutive pair of coarse waypoints (cell_i → cell_{i+1}),
    generate fine-resolution GPS points (10-sec intervals) that travel FROM
    the centre of cell_i TO the centre of cell_{i+1} over dwell_hours[i] hours.

    Technique: detrend + retrend ("rubber-band")
      - Generate a raw fine segment from the diffusion model
      - Remove the segment's own linear drift
      - Replace it with the desired drift (cell_i_centre → cell_{i+1}_centre)
      - Local movement wiggles are fully preserved; only the overall direction changes

Output
------
  results/full_trajectory.npy        — (1, T_total, 2) float32, columns (lon, lat)
  results/full_trajectory_map.html   — interactive Folium map

Usage
-----
  python scripts/generate_full.py
  python scripts/generate_full.py --coarse-steps 12 --behavior MOVE --season dry
  python scripts/generate_full.py --coarse-steps 20 --temperature 0.8 --device cuda
"""

import argparse
import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import h3
import folium
from folium.plugins import AntPath

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from elephantgraph.models.coarse_generator import CoarseGenerator
from elephantgraph.models.fine_generator import ElephantFineDiffusionTransformer
from elephantgraph.models.diffusion import DDIMDiffusion
from elephantgraph.training.dataset_coarse import load_h3_zoom
from elephantgraph.preprocessing.scalers import load_scalers, fit_scalers, save_scalers

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")

BEHAVIOR_MAP        = {"STOP": 0, "MOVE": 1}
SEASON_MAP          = {"dry": 0, "wet": 1}
FINE_STEP_SEC       = 10
FINE_STEPS_PER_HOUR = 3600 // FINE_STEP_SEC   # 360
FINE_MAX_SEQ        = 200                      # diffusion model context window


# ---------------------------------------------------------------------------
# Level 1 — Coarse generation
# ---------------------------------------------------------------------------

def h3_cell_circumradius_deg(h3_str):
    """
    Return the H3 cell circumradius (centre -> farthest vertex) in degrees.
    Used to size the corridor the fine path is allowed to wander within.
    """
    clat, clon = h3.cell_to_latlng(h3_str)
    verts = h3.cell_to_boundary(h3_str)            # list of (lat, lon)
    dists = [np.hypot(la - clat, lo - clon) for la, lo in verts]
    return float(max(dists))


def build_cell_point_lookup(df):
    """
    Group the real observed GPS points by the H3 cell they fall in.

    Returns: dict h3_str -> np.ndarray (M, 2) of real (lon, lat) points.
    Used by the WildGraph-style coordinate extractor so each generated H3
    block is grounded in actual elephant locations, not the bare hex centre.
    `df` must already carry an 'h3_idx' column (added by build_h3_mapping).
    """
    lookup = {}
    for h3_str, grp in df.groupby("h3_idx"):
        lookup[h3_str] = grp[["longitude", "latitude"]].to_numpy(dtype=np.float64)
    return lookup


def extract_exact_coord(h3_str, cell_points, method="medoid", rng=None,
                        max_medoid_pts=400):
    """
    WildGraph-style coordinate extraction: turn an abstract H3 block into a
    concrete (lon, lat) anchor drawn from the REAL points observed inside it.

    method:
      - "medoid"   : the real observed point with minimum total distance to the
                     others — a robust, central, guaranteed-on-land location
                     (recommended; avoids lakes / no-go areas inside the cell)
      - "sample"   : a uniformly random real observed point in the cell
      - "centroid" : mean of observed points (falls back toward the middle)

    Falls back to the hex geometric centre only if the cell has no observed
    points at all (rare for cells the coarse model actually visits).
    """
    pts = cell_points.get(h3_str)
    if pts is None or len(pts) == 0:
        lat, lon = h3.cell_to_latlng(h3_str)
        return np.array([lon, lat], dtype=np.float64)

    if method == "centroid":
        return pts.mean(axis=0)

    if method == "sample":
        rng = rng or np.random.default_rng()
        return pts[rng.integers(len(pts))]

    # medoid (default)
    sub = pts
    if len(pts) > max_medoid_pts:
        rng = rng or np.random.default_rng()
        sub = pts[rng.choice(len(pts), max_medoid_pts, replace=False)]
    # pairwise distances -> point minimizing total distance to the rest
    dmat = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=-1)
    return sub[dmat.sum(axis=1).argmin()]


def build_h3_mapping(hourly_csv_path, h3_graph_dir):
    zoom = load_h3_zoom(h3_graph_dir)
    df   = pd.read_csv(hourly_csv_path, parse_dates=["timestamp"])
    df["h3_idx"] = df.apply(
        lambda r: h3.latlng_to_cell(r["latitude"], r["longitude"], zoom), axis=1
    )
    all_h3   = sorted(df["h3_idx"].unique())
    h3_to_id = {h: i for i, h in enumerate(all_h3)}
    id_to_h3 = {i: h for h, i in h3_to_id.items()}
    print(f"  H3 zoom={zoom}  ({len(all_h3)} unique cells)")
    return h3_to_id, id_to_h3, df


@torch.no_grad()
def generate_coarse(model, node_embeddings, seed_node,
                    behavior_code, season_code, context_len,
                    temperature, device,
                    steps=None, target_hours=None,
                    latent_temperature=1.0, latent_mode=None):
    """
    Autoregressively generate an hourly H3 walk.

    Two stopping modes (exactly one should be set):
      - target_hours : simulate this many hourly ticks (use for "N days":
                        target_hours = days * 24).  This is the natural control
                        because each raw tick is one simulated hour.
      - steps        : stop after this many distinct cell transitions.

    The seasonal latent dictionary is consulted every step (driven by
    `season_code`); latent_temperature / latent_mode sharpen or hand-pick the
    seasonal movement mode.

    Returns:
        coarse_nodes  : list of int node IDs (includes seed)
        coarse_dwell  : list of int dwell hours, same length as coarse_nodes
                        (sum(coarse_dwell) == total simulated hours)
    """
    model.eval()
    node_embeddings = node_embeddings.to(device)
    behavior = torch.tensor([behavior_code], dtype=torch.long, device=device)
    season   = torch.tensor([season_code],   dtype=torch.long, device=device)

    if target_hours is not None:
        max_raw = int(target_hours)
    else:
        max_raw = steps * 50          # generous safety cap for transition mode

    generated     = [seed_node]
    dwell_hours   = []
    current_dwell = 1
    raw_count = transitions = 0

    while raw_count < max_raw:
        if target_hours is None and transitions >= steps:
            break

        ctx = generated[-context_len:]
        node_indices = torch.tensor(
            np.array([ctx], dtype=np.int64), dtype=torch.long, device=device
        )
        logits, _ = model(node_indices, node_embeddings, season, behavior,
                          latent_temperature=latent_temperature,
                          latent_mode=latent_mode)
        next_logits = logits[0, -1, :] / temperature
        probs       = torch.softmax(next_logits, dim=-1)
        next_node   = torch.multinomial(probs, num_samples=1).item()
        raw_count  += 1

        if next_node == generated[-1]:
            current_dwell += 1
        else:
            dwell_hours.append(current_dwell)
            generated.append(next_node)
            current_dwell = 1
            transitions  += 1

    dwell_hours.append(current_dwell)

    print(f"  {raw_count} raw ticks ({raw_count / 24:.1f} days) -> "
          f"{transitions} transitions  (avg dwell {raw_count / max(transitions, 1):.1f}h)")
    return generated, dwell_hours


# ---------------------------------------------------------------------------
# Level 2 — Fine generation
# ---------------------------------------------------------------------------

def _env_context(n, seq_len, season, device):
    d = (dict(ndvi=0.25, evi=0.18, lst=42.0, elev=1100.0, slope=2.0, water=0.1)
         if season == "dry" else
         dict(ndvi=0.55, evi=0.40, lst=32.0, elev=1100.0, slope=2.0, water=0.5))
    return {k: torch.full((n, seq_len), v, dtype=torch.float32, device=device)
            for k, v in d.items()}


@torch.no_grad()
def _run_diffusion(model, diffusion, behavior_code, season_code, seq_len, device):
    """Generate one raw normalized fine segment of length seq_len."""
    n = 1
    season_str = "dry" if season_code == 0 else "wet"
    cond = {
        "behavior":     torch.full((n,), behavior_code, dtype=torch.long,  device=device),
        "season":       torch.full((n,), season_code,   dtype=torch.long,  device=device),
        "time_of_day":  torch.zeros(n,   dtype=torch.long,  device=device),
        "lulc":         torch.full((n,), 10, dtype=torch.long, device=device),
        "human_settle": torch.zeros(n,   dtype=torch.long,  device=device),
        "move_type":    torch.zeros(n,   dtype=torch.long,  device=device),
    }
    cond.update(_env_context(n, seq_len, season_str, device))
    for k in ("speed", "accel", "turning", "bearing", "persist", "step"):
        cond[k] = torch.zeros(n, seq_len, dtype=torch.float32, device=device)
    raw = diffusion.generate(model, cond, device, seq_len)   # (1, seq_len, 2)
    return raw[0].cpu().numpy()                              # (seq_len, 2)


def rubber_band(segment, start_lonlat, end_lonlat, corridor_halfwidth):
    """
    Lay the fine model's movement down as ONE continuous curve from
    start_lonlat to end_lonlat, inside the corridor between the two anchors.

    Key idea — forward progress comes from the SPINE, lateral character comes
    from the MODEL:
      - The straight A->B line (the "spine") provides strictly monotonic
        forward progress, so the path always advances A -> B and never doubles
        back or fans out into multiple strands.
      - Only the model's PERPENDICULAR (cross-track) wiggle is borrowed and
        scaled into the corridor, giving the curve its realistic meander.

    We deliberately DROP the model's along-track motion: after MinMax
    normalization the diffusion output spans the whole training extent, so its
    raw forward motion is wildly out of scale for a single inter-cell hop and,
    if used, produces the back-and-forth "hairball" of straight lines.

    segment             : (T, 2) real lon/lat degrees (continuous strand)
    start_lonlat        : (lon, lat) segment start anchor
    end_lonlat          : (lon, lat) segment end anchor
    corridor_halfwidth  : max perpendicular deviation (degrees)
    """
    T = len(segment)
    t = np.linspace(0.0, 1.0, T)[:, None]              # (T, 1)

    start = np.asarray(start_lonlat, dtype=np.float64)
    end   = np.asarray(end_lonlat,   dtype=np.float64)

    # Model shape with its own linear drift removed -> pure local wiggle
    seg_drift = segment[-1] - segment[0]
    residual  = segment - (segment[0] + t * seg_drift)  # (T, 2)

    ab     = end - start
    ab_len = np.linalg.norm(ab)

    if ab_len < 1e-9:
        # Two anchors coincide (dwell in place): produce a compact local loop
        # around the anchor, scaled to the corridor radius.
        r = np.linalg.norm(residual, axis=1).max()
        scale = (corridor_halfwidth / r) if r > 1e-12 else 0.0
        return start + residual * scale

    u = ab / ab_len                                    # along-track unit
    v = np.array([-u[1], u[0]])                        # cross-track (perpendicular) unit
    spine = start + t * ab                             # monotonic A -> B

    # Perpendicular wiggle from the model, robustly scaled into the corridor.
    cross = residual @ v                               # (T,) signed cross-track
    s = cross.std()
    if s > 1e-12:
        # Map ~2.5 std to the corridor edge, then clip — fills the corridor
        # without one outlier flattening the whole curve.
        cross = np.clip(cross / (2.5 * s), -1.0, 1.0) * corridor_halfwidth
    else:
        cross = np.zeros_like(cross)

    return spine + cross[:, None] * v


def generate_fine_between(model, diffusion, gps_scaler,
                           start_lonlat, end_lonlat,
                           n_pts_total, behavior_code, season_code,
                           corridor_halfwidth, device):
    """
    Generate fine GPS points that travel from start_lonlat to end_lonlat
    in n_pts_total steps, constrained to the corridor between the two cells.
    Long segments are generated in chunks and concatenated, then warped as
    a whole so the path reaches end_lonlat without leaving the corridor.
    """
    # The fine model can only emit FINE_MAX_SEQ points per call.  For long
    # dwells we call it repeatedly, but each call is an INDEPENDENT sample at
    # its own absolute location.  If we just concatenated them we'd get a
    # teleport jump (a stray "strand") at every chunk boundary.  Instead we
    # chain chunks end-to-start: shift each new chunk so its first point sits
    # exactly on the previous chunk's last point, preserving each chunk's
    # internal wiggle while producing ONE continuous strand.
    placed   = []
    cursor   = None
    remaining = n_pts_total

    while remaining > 0:
        chunk_len = min(remaining, FINE_MAX_SEQ)
        raw  = _run_diffusion(model, diffusion, behavior_code, season_code,
                              chunk_len, device)           # normalized
        real = gps_scaler.inverse_transform(raw)           # real lon/lat

        if cursor is None:
            placed.append(real)                            # first chunk as-is
        else:
            # Shift so chunk starts where the previous one ended; drop the
            # duplicated join point to keep step spacing uniform.
            shifted = real - real[0] + cursor
            placed.append(shifted[1:])
        cursor = placed[-1][-1]
        remaining -= chunk_len

    segment = np.concatenate(placed, axis=0)               # continuous strand

    # Warp the continuous strand into the corridor between the two anchors
    segment = rubber_band(segment, start_lonlat, end_lonlat,
                          corridor_halfwidth)
    return segment


# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------

def build_full_map(full_traj, coarse_nodes, coarse_dwell,
                   id_to_h3, output_html, anchors=None, max_render_pts=2000):
    lats = full_traj[:, 1]
    lons = full_traj[:, 0]
    T    = len(full_traj)

    # Subsample fine path for browser performance
    if T > max_render_pts:
        step    = max(1, T // max_render_pts)
        indices = list(range(0, T, step))
        if indices[-1] != T - 1:
            indices.append(T - 1)
        render  = full_traj[indices]
        sub_note = f"1-in-{step} sub-sampled for rendering"
    else:
        render   = full_traj
        sub_note = "full resolution"

    centre = [float(lats.mean()), float(lons.mean())]
    fmap   = folium.Map(location=centre, zoom_start=8,
                        tiles="CartoDB positron", prefer_canvas=True)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite",
        overlay=False, control=True,
    ).add_to(fmap)

    # ── Coarse hexagons ────────────────────────────────────────────────────
    n_cells = len(coarse_nodes)
    coarse_centres = []
    for i, nid in enumerate(coarse_nodes):
        h3_str = id_to_h3[nid]
        t = i / max(n_cells - 1, 1)
        if t < 0.5:
            r, g, b = int(t*2*60), int(120+t*2*40), 200
        else:
            r, g, b = int(60+(t-0.5)*2*195), int(160-(t-0.5)*2*160), int(200-(t-0.5)*2*200)
        color = "#{:02x}{:02x}{:02x}".format(r, g, b)

        boundary = [(float(la), float(lo))
                    for la, lo in h3.cell_to_boundary(h3_str)]
        dwell = int(coarse_dwell[i]) if i < len(coarse_dwell) else 1
        folium.Polygon(
            locations=boundary,
            color=color, fill=True, fill_color=color,
            fill_opacity=0.15, weight=2,
            tooltip=f"Coarse step {i+1} | {dwell}h | {h3_str}",
        ).add_to(fmap)

        # Prefer the WildGraph-extracted real anchor; fall back to hex centre
        if anchors is not None:
            coarse_centres.append((float(anchors[i][1]), float(anchors[i][0])))
            folium.CircleMarker(
                location=(float(anchors[i][1]), float(anchors[i][0])),
                radius=4, color="#e67e22", fill=True, fill_color="#e67e22",
                fill_opacity=1.0,
                tooltip=f"Anchor {i+1} (extracted real coord) | {h3_str}",
            ).add_to(fmap)
        else:
            clat, clon = h3.cell_to_latlng(h3_str)
            coarse_centres.append((float(clat), float(clon)))

    # Animated orange path connecting the extracted coarse anchors
    if len(coarse_centres) >= 2:
        AntPath(
            locations=coarse_centres,
            color="#e67e22", weight=3, opacity=0.9, delay=600,
            tooltip="Coarse H3 waypoints",
        ).add_to(fmap)

    # ── Fine GPS polyline ──────────────────────────────────────────────────
    latlon_fine = [(float(r[1]), float(r[0])) for r in render]
    folium.PolyLine(
        locations=latlon_fine,
        color="#1a1a2e", weight=1.5, opacity=0.7,
        tooltip=f"Fine GPS path — {T:,} pts @ 10 sec  ({sub_note})",
    ).add_to(fmap)

    folium.CircleMarker(latlon_fine[0],  radius=8,
                        color="#2ecc71", fill=True, fill_color="#2ecc71",
                        fill_opacity=1.0, tooltip="START").add_to(fmap)
    folium.CircleMarker(latlon_fine[-1], radius=8,
                        color="#e74c3c", fill=True, fill_color="#e74c3c",
                        fill_opacity=1.0, tooltip="END").add_to(fmap)

    # ── Legend ─────────────────────────────────────────────────────────────
    fmap.get_root().html.add_child(folium.Element(f"""
    <div style="position:fixed;bottom:28px;left:28px;z-index:1000;
                background:rgba(255,255,255,0.93);padding:10px 15px;
                border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,0.2);
                font-family:sans-serif;font-size:12px;line-height:1.9;">
        <b style="font-size:13px;">ElephantGraph</b><br>
        <b>Level 1 coarse:</b> {n_cells} H3 waypoints
        <span style="color:#e67e22;">&#9473;&#9473;</span><br>
        <b>Level 2 fine:</b> {T:,} GPS pts @ 10 sec<br>
        <span style="color:#2ecc71;">&#9679;</span> start &nbsp;
        <span style="color:#e74c3c;">&#9679;</span> end
    </div>
    """))

    fmap.fit_bounds([
        [float(lats.min()) - 0.05, float(lons.min()) - 0.05],
        [float(lats.max()) + 0.05, float(lons.max()) + 0.05],
    ])
    folium.LayerControl().add_to(fmap)
    fmap.save(output_html)
    print(f"  Map saved -> {output_html}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ElephantGraph end-to-end: coarse H3 waypoints -> fine GPS path"
    )
    parser.add_argument("--coarse-ckpt",  default=os.path.join(PROJECT_DIR, "checkpoints", "coarse", "best_coarse_model.pt"))
    parser.add_argument("--fine-ckpt",    default=os.path.join(PROJECT_DIR, "checkpoints", "fine",   "best_model.pt"))
    parser.add_argument("--hourly-csv",   default=os.path.join(PROJECT_DIR, "elephantgraph", "data", "processed", "hourly", "hourly.csv"))
    parser.add_argument("--node-emb",     default=os.path.join(PROJECT_DIR, "elephantgraph", "data", "processed", "h3_graph", "node_embeddings.npy"))
    parser.add_argument("--scalers-dir",  default=os.path.join(PROJECT_DIR, "elephantgraph", "data", "scalers"))
    parser.add_argument("--days",         type=float, default=None,
                        help="Generate a trajectory spanning this many days "
                             "(total simulated hours = days*24). Takes precedence "
                             "over --coarse-steps.")
    parser.add_argument("--coarse-steps", type=int,   default=12,
                        help="Number of H3 cell transitions (used only if --days "
                             "is not given)")
    parser.add_argument("--context-len",  type=int,   default=23)
    parser.add_argument("--temperature",  type=float, default=1.0, help="Coarse sampling temperature")
    parser.add_argument("--latent-temperature", type=float, default=1.0,
                        help="Seasonal latent dictionary attention temperature "
                             "(<1 = commit to one seasonal mode, >1 = blend modes)")
    parser.add_argument("--latent-mode", type=int, default=None,
                        help="Force a specific seasonal dictionary entry index k "
                             "(0..K-1) to deterministically replay one movement mode")
    parser.add_argument("--corridor-factor", type=float, default=1.0,
                        help="Corridor half-width as a multiple of the H3 cell radius "
                             "(1.0 = stay within one cell radius of the A->B line; "
                             "lower = tighter, straighter paths)")
    parser.add_argument("--anchor-method", default="medoid",
                        choices=["medoid", "sample", "centroid"],
                        help="How to extract an exact coordinate from each H3 block: "
                             "medoid = central real observed point (recommended), "
                             "sample = random real point, centroid = mean of real points")
    parser.add_argument("--anchor-seed", type=int, default=0,
                        help="RNG seed for anchor sampling (reproducibility)")
    parser.add_argument("--seed-node",    type=int,   default=None)
    parser.add_argument("--behavior",     default="MOVE", choices=["MOVE", "STOP"])
    parser.add_argument("--season",       default="dry",  choices=["dry", "wet"])
    parser.add_argument("--device",       default="cuda")
    parser.add_argument("--output-npy",   default=os.path.join(PROJECT_DIR, "results", "full_trajectory.npy"))
    parser.add_argument("--output-html",  default=os.path.join(PROJECT_DIR, "results", "full_trajectory_map.html"))
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    behavior_code = BEHAVIOR_MAP[args.behavior]
    season_code   = SEASON_MAP[args.season]

    # ── GPS scaler ───────────────────────────────────────────────────────────
    gps_scaler_path = os.path.join(args.scalers_dir, "gps_scaler.pkl")
    if os.path.exists(gps_scaler_path):
        gps_scaler, _, _ = load_scalers(args.scalers_dir)
        print(f"GPS scaler loaded  lon=[{gps_scaler.data_min_[0]:.3f}, {gps_scaler.data_max_[0]:.3f}]"
              f"  lat=[{gps_scaler.data_min_[1]:.3f}, {gps_scaler.data_max_[1]:.3f}]")
    else:
        print("No saved scaler — fitting from training windows ...")
        train_pkl = os.path.join(PROJECT_DIR, "elephantgraph", "data",
                                 "processed", "windows", "train.pkl")
        if not os.path.exists(train_pkl):
            raise FileNotFoundError(
                f"Cannot find {train_pkl}.\n"
                "Run:  python -m elephantgraph.preprocessing.scalers"
            )
        with open(train_pkl, "rb") as f:
            train_windows = pickle.load(f)
        gps_scaler, kin_scaler, env_scaler = fit_scalers(train_windows)
        save_scalers(gps_scaler, kin_scaler, env_scaler, args.scalers_dir)
        print(f"Scaler fitted and saved to {args.scalers_dir}")

    # ── STEP 1: Level 1 — generate all coarse waypoints ─────────────────────
    print("\n── Level 1: Coarse H3 waypoints ──────────────────────────────────")
    h3_graph_dir = os.path.dirname(args.node_emb)
    h3_to_id, id_to_h3, hourly_df = build_h3_mapping(args.hourly_csv, h3_graph_dir)
    num_h3_nodes = len(h3_to_id)

    emb_dict = np.load(args.node_emb, allow_pickle=True).item()
    embed_dim = next(iter(emb_dict.values())).shape[0]
    node_embeddings = torch.zeros(num_h3_nodes, embed_dim)
    for hex_str, emb in emb_dict.items():
        if hex_str in h3_to_id:
            node_embeddings[h3_to_id[hex_str]] = torch.tensor(emb, dtype=torch.float32)

    ckpt_c = torch.load(args.coarse_ckpt, map_location="cpu")
    ms_c   = ckpt_c["model_state"]
    d_model_c    = ms_c["encoder.node_embed.weight"].shape[0]
    num_layers_c = sum(1 for k in ms_c if k.startswith("encoder.encoder.layers.")
                       and k.endswith(".norm1.weight"))
    nhead_c = 8 if d_model_c % 8 == 0 else 4

    # Seasonal latent dictionary dims (auto-detected from checkpoint)
    if "latent_dict.codebook" in ms_c:
        n_seasons_c, n_latent_c, _ = ms_c["latent_dict.codebook"].shape
        has_dict = True
    else:
        n_seasons_c, n_latent_c, has_dict = 2, 16, False

    coarse_model = CoarseGenerator(
        d_model=d_model_c, nhead=nhead_c,
        num_layers=num_layers_c, num_h3_nodes=num_h3_nodes,
        num_latent_entries=n_latent_c, num_seasons=n_seasons_c,
    ).to(device)
    coarse_model.load_state_dict(ms_c)
    coarse_model.eval()
    print(f"Coarse model: d_model={d_model_c}  layers={num_layers_c}  cells={num_h3_nodes}")
    if has_dict:
        print(f"Seasonal latent dictionary: {n_seasons_c} seasons x {n_latent_c} entries"
              f"  (temp={args.latent_temperature}"
              f"{', mode='+str(args.latent_mode) if args.latent_mode is not None else ''})")
    else:
        print("  (checkpoint has no latent dictionary — using legacy season embedding)")

    seed = args.seed_node
    if seed is None:
        seed = h3_to_id[hourly_df["h3_idx"].value_counts().index[0]]
    print(f"Seed node: {seed}  ({id_to_h3[seed]})")

    if args.days is not None:
        target_hours = int(round(args.days * 24))
        print(f"Generating {args.days} days  ({target_hours} hourly ticks) ...")
    else:
        target_hours = None
        print(f"Generating {args.coarse_steps} cell transitions ...")

    coarse_nodes, coarse_dwell = generate_coarse(
        coarse_model, node_embeddings, seed,
        steps=args.coarse_steps,
        target_hours=target_hours,
        behavior_code=behavior_code, season_code=season_code,
        context_len=args.context_len, temperature=args.temperature,
        device=device,
        latent_temperature=args.latent_temperature,
        latent_mode=args.latent_mode,
    )

    # Print the coarse plan
    print(f"\nCoarse plan ({len(coarse_nodes)} waypoints):")
    for i, (nid, dwell) in enumerate(zip(coarse_nodes, coarse_dwell)):
        clat, clon = h3.cell_to_latlng(id_to_h3[nid])
        print(f"  [{i:>2}] node={nid:>4}  dwell={dwell:>3}h  "
              f"centre=({clat:.3f}N, {clon:.3f}E)  {id_to_h3[nid]}")

    del coarse_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── STEP 1b: Extract exact coordinates from each H3 block (WildGraph) ────
    # Each abstract H3 cell is grounded in a concrete REAL observed location
    # inside it, so the fine path is drawn between places elephants actually
    # were — not the geometric hex centre (which can land in a lake).
    print(f"\n── Extracting exact coordinates from H3 blocks "
          f"(method={args.anchor_method}) ──")
    rng = np.random.default_rng(args.anchor_seed)
    cell_points = build_cell_point_lookup(hourly_df)

    anchors = []          # list of (lon, lat) real-grounded anchors
    for i, nid in enumerate(coarse_nodes):
        h3_str = id_to_h3[nid]
        coord  = extract_exact_coord(h3_str, cell_points,
                                     method=args.anchor_method, rng=rng)
        anchors.append(coord)
        npts = len(cell_points.get(h3_str, []))
        clat, clon = h3.cell_to_latlng(h3_str)
        offset_km = np.hypot(coord[1] - clat, coord[0] - clon) * 111
        print(f"  [{i:>2}] {h3_str}  anchor=({coord[1]:.4f}N, {coord[0]:.4f}E)  "
              f"from {npts} real pts  ({offset_km:.1f} km off-centre)")

    # ── STEP 2: Level 2 — fill path between every consecutive waypoint pair ──
    print("\n── Level 2: Fine GPS between each waypoint pair ──────────────────")
    ckpt_f = torch.load(args.fine_ckpt, map_location="cpu")
    ms_f   = ckpt_f["model_state"]
    num_layers_f = sum(1 for k in ms_f if k.startswith("transformer_blocks.")
                       and k.endswith(".norm1.weight"))
    d_model_f = ms_f["gps_kin_embed.lon_embed.weight"].shape[0]
    nhead_f   = ckpt_f.get("nhead", 8)
    if d_model_f % nhead_f != 0:
        nhead_f = 4

    fine_model = ElephantFineDiffusionTransformer(
        d_model=d_model_f, nhead=nhead_f,
        num_layers=num_layers_f, max_seq_len=FINE_MAX_SEQ,
    ).to(device)
    fine_model.load_state_dict(ms_f)
    fine_model.eval()
    diffusion = DDIMDiffusion(T=200, S=40)
    print(f"Fine model:   d_model={d_model_f}  layers={num_layers_f}")

    # Corridor half-width: derive from the actual H3 cell circumradius (centre
    # to vertex) in degrees, scaled by --corridor-factor.  Fine points can never
    # stray further than this perpendicular distance from the A->B spine, so the
    # path stays inside the cells the coarse model chose.
    cell_circumradius_deg = h3_cell_circumradius_deg(id_to_h3[coarse_nodes[0]])
    corridor_halfwidth = cell_circumradius_deg * args.corridor_factor
    print(f"Corridor half-width: {corridor_halfwidth:.4f} deg "
          f"(~{corridor_halfwidth * 111:.1f} km, cell radius x {args.corridor_factor})\n")

    fine_segments = []
    total_pts     = 0

    for i in range(len(coarse_nodes) - 1):
        node_a, node_b = coarse_nodes[i], coarse_nodes[i + 1]
        dwell          = coarse_dwell[i]

        # Use the WildGraph-extracted real anchors, not the hex centres
        start_lonlat = (float(anchors[i][0]),     float(anchors[i][1]))
        end_lonlat   = (float(anchors[i + 1][0]), float(anchors[i + 1][1]))

        n_pts = dwell * FINE_STEPS_PER_HOUR

        segment = generate_fine_between(
            fine_model, diffusion, gps_scaler,
            start_lonlat=start_lonlat,
            end_lonlat  =end_lonlat,
            n_pts_total =n_pts,
            behavior_code=behavior_code,
            season_code  =season_code,
            corridor_halfwidth=corridor_halfwidth,
            device       =device,
        )

        fine_segments.append(segment)
        total_pts += len(segment)
        print(f"  Segment {i+1:>2}/{len(coarse_nodes)-1}  "
              f"({id_to_h3[node_a]} -> {id_to_h3[node_b]})  "
              f"dwell={dwell}h  pts={len(segment)}")

    full_traj = np.concatenate(fine_segments, axis=0)   # (T_total, 2)
    print(f"\nTotal fine GPS points: {total_pts:,}  "
          f"(~{total_pts * FINE_STEP_SEC / 3600:.1f} simulated hours)")

    del fine_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── Save ─────────────────────────────────────────────────────────────────
    print("\n── Saving ────────────────────────────────────────────────────────")
    os.makedirs(os.path.dirname(args.output_npy),  exist_ok=True)
    os.makedirs(os.path.dirname(args.output_html), exist_ok=True)
    np.save(args.output_npy, full_traj[np.newaxis])      # (1, T, 2)
    print(f"  GPS  -> {args.output_npy}  shape={full_traj[np.newaxis].shape}")

    # Rich bundle (fine + coarse layers) for the layered viewer
    coarse_h3   = np.array([id_to_h3[nid] for nid in coarse_nodes])
    coarse_bnds = np.array(
        [np.array(h3.cell_to_boundary(id_to_h3[nid]), dtype=np.float64)  # (V,2) lat,lon
         for nid in coarse_nodes], dtype=object
    )
    output_npz = os.path.splitext(args.output_npy)[0] + ".npz"
    np.savez(
        output_npz,
        fine=full_traj.astype(np.float32),                  # (T,2) lon,lat
        anchors=np.array(anchors, dtype=np.float64),        # (M,2) lon,lat
        coarse_h3=coarse_h3,                                # (M,) str
        coarse_dwell=np.array(coarse_dwell, dtype=np.int32),# (M,)
        coarse_boundaries=coarse_bnds,                      # (M,) object of (V,2) lat,lon
    )
    print(f"  Bundle -> {output_npz}  (fine + coarse layers for the viewer)")

    build_full_map(full_traj, coarse_nodes, coarse_dwell, id_to_h3,
                   args.output_html, anchors=anchors)


if __name__ == "__main__":
    main()
