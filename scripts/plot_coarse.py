"""
Plot a coarse H3 trajectory (output of generate_coarse.py) on an interactive map.

Each H3 cell is drawn as a filled hexagon. The path between consecutive cell
centres is drawn as an animated ant-path so direction is visible. Visit order
is encoded by color (early = blue, late = red).

Usage:
    python scripts/plot_coarse.py --input results/coarse_trajectory.npz
    python scripts/plot_coarse.py --input results/coarse_trajectory.npz --output results/coarse_map.html
"""

import argparse
import os
import numpy as np
import folium
from folium.plugins import AntPath


def step_color(i, total):
    """Blue -> green -> red gradient encoding temporal order."""
    t = i / max(total - 1, 1)
    if t < 0.5:
        r, g, b = int(t * 2 * 100), int(100 + t * 2 * 55), 200
    else:
        r, g, b = int(100 + (t - 0.5) * 2 * 155), int(155 - (t - 0.5) * 2 * 155), int(200 - (t - 0.5) * 2 * 200)
    return "#{:02x}{:02x}{:02x}".format(r, g, b)


def plot_coarse(input_path, output_path):
    data = np.load(input_path, allow_pickle=True)
    centers     = data["centers"]        # (N, 2) lon/lat
    boundaries  = data["boundaries"]     # (N,) object array of polygon coords
    node_ids    = data["node_ids"]       # (N,) int
    h3_strings  = data["h3_strings"]     # (N,) str
    dwell_hours = data["dwell_hours"] if "dwell_hours" in data else np.ones(len(centers), dtype=int)

    N = len(centers)
    centre_latlon = [float(centers[:, 1].mean()), float(centers[:, 0].mean())]

    fmap = folium.Map(
        location=centre_latlon,
        zoom_start=7,
        tiles="CartoDB positron",
        prefer_canvas=True,
    )

    # Satellite layer toggle
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite",
        overlay=False,
        control=True,
    ).add_to(fmap)

    # ── Per-step hexagons ──────────────────────────────────────────────────
    # Unique cells get a single drawn hexagon; repeated visits shown via path
    drawn_cells = {}

    for i in range(N):
        h3_str  = str(h3_strings[i])
        color   = step_color(i, N)
        poly    = [(float(lat), float(lon)) for lat, lon in boundaries[i]]

        dwell = int(dwell_hours[i]) if i < len(dwell_hours) else 1
        tip   = f"Step {i+1} | {dwell}h dwell | H3: {h3_str}"

        if h3_str not in drawn_cells:
            folium.Polygon(
                locations=poly,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.25,
                weight=2,
                tooltip=tip,
            ).add_to(fmap)
            drawn_cells[h3_str] = color
        else:
            folium.Polygon(
                locations=poly,
                color=color,
                fill=False,
                weight=1.5,
                opacity=0.5,
                tooltip=tip + " (revisit)",
            ).add_to(fmap)

    # ── Animated path through cell centres ────────────────────────────────
    path_latlon = [(float(centers[i, 1]), float(centers[i, 0])) for i in range(N)]

    AntPath(
        locations=path_latlon,
        color="#333333",
        weight=2,
        opacity=0.7,
        delay=500,
        tooltip="Coarse trajectory path",
    ).add_to(fmap)

    # ── Start / end markers ────────────────────────────────────────────────
    folium.CircleMarker(
        location=path_latlon[0],
        radius=8,
        color="#2ecc71",
        fill=True,
        fill_color="#2ecc71",
        fill_opacity=1.0,
        tooltip=f"START — step 1 | {h3_strings[0]}",
    ).add_to(fmap)

    folium.CircleMarker(
        location=path_latlon[-1],
        radius=8,
        color="#e74c3c",
        fill=True,
        fill_color="#e74c3c",
        fill_opacity=1.0,
        tooltip=f"END — step {N} | {h3_strings[-1]}",
    ).add_to(fmap)

    # Step number labels for short sequences
    if N <= 30:
        for i in range(N):
            folium.Marker(
                location=path_latlon[i],
                icon=folium.DivIcon(
                    html=f'<div style="font-size:10px;font-weight:bold;'
                         f'color:#333;text-shadow:0 0 3px white;">{i+1}</div>',
                    icon_size=(20, 20),
                    icon_anchor=(10, 10),
                ),
            ).add_to(fmap)

    # ── Colour legend (temporal gradient) ────────────────────────────────
    gradient_css = ", ".join(
        step_color(i, 10) for i in range(10)
    )
    fmap.get_root().html.add_child(folium.Element(f"""
    <div style="position:fixed;bottom:28px;left:28px;z-index:1000;
                background:rgba(255,255,255,0.92);padding:10px 14px;
                border-radius:7px;box-shadow:0 2px 8px rgba(0,0,0,0.25);
                font-family:sans-serif;font-size:12px;line-height:1.8;">
        <b style="font-size:13px;">Coarse H3 trajectory</b><br>
        {N} steps &nbsp;|&nbsp; {len(set(str(s) for s in h3_strings))} unique cells<br>
        <div style="display:flex;align-items:center;gap:6px;margin-top:4px;">
            <span>early</span>
            <div style="width:120px;height:10px;border-radius:4px;
                        background:linear-gradient(to right,{gradient_css});"></div>
            <span>late</span>
        </div>
        <span style="color:#2ecc71;">&#9679;</span> start &nbsp;
        <span style="color:#e74c3c;">&#9679;</span> end
    </div>
    """))

    # Fit bounds
    fmap.fit_bounds([
        [float(centers[:, 1].min()), float(centers[:, 0].min())],
        [float(centers[:, 1].max()), float(centers[:, 0].max())],
    ])
    folium.LayerControl().add_to(fmap)

    fmap.save(output_path)
    print(f"Saved coarse map ({N} steps) -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True,
                        help="Path to .npz output of generate_coarse.py")
    parser.add_argument("--output", default=None,
                        help="Output HTML path (default: <input>_map.html)")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.splitext(args.input)[0] + "_map.html"

    plot_coarse(args.input, args.output)
