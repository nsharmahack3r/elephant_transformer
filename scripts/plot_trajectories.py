"""
Plot generated elephant trajectories on an interactive Folium map.

Usage:
    python scripts/plot_trajectories.py --input results/trajectories.npy
    python scripts/plot_trajectories.py --input results/trajectories.npy --n 50 --output results/map.html
"""

import argparse
import colorsys
import os
import numpy as np
import folium
from folium.plugins import AntPath


def make_colors(n):
    """Generate n visually distinct hex colors via HSV."""
    return [
        "#{:02x}{:02x}{:02x}".format(
            *[int(c * 255) for c in colorsys.hsv_to_rgb(i / max(n, 1), 0.75, 0.88)]
        )
        for i in range(n)
    ]


def plot_trajectories(input_path, output_path, n_trajs=20):
    data = np.load(input_path)
    assert data.ndim == 3 and data.shape[2] == 2, \
        f"Expected shape (N, T, 2), got {data.shape}"

    n_total, seq_len, _ = data.shape
    n_show = min(n_trajs, n_total)
    trajs = data[:n_show]

    all_lats = trajs[:, :, 1]
    all_lons = trajs[:, :, 0]
    centre = [float(all_lats.mean()), float(all_lons.mean())]

    fmap = folium.Map(
        location=centre,
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

    colors = make_colors(n_show)

    for i, traj in enumerate(trajs):
        valid = ~np.isnan(traj).any(axis=1)
        if valid.sum() < 2:
            continue

        latlon = [(float(lat), float(lon)) for lon, lat in traj[valid]]
        color  = colors[i]
        label  = f"Trajectory {i + 1}"

        # Animated moving-ant path
        AntPath(
            locations=latlon,
            color=color,
            weight=2.5,
            opacity=0.85,
            delay=600,
            tooltip=label,
        ).add_to(fmap)

        # Start dot (filled circle)
        folium.CircleMarker(
            location=latlon[0],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=1.0,
            tooltip=f"{label} — start",
        ).add_to(fmap)

        # End dot (white-filled = visually distinct)
        folium.CircleMarker(
            location=latlon[-1],
            radius=6,
            color=color,
            fill=True,
            fill_color="white",
            fill_opacity=0.9,
            tooltip=f"{label} — end",
        ).add_to(fmap)

    # Fit map to data extent
    fmap.fit_bounds([
        [float(all_lats.min()), float(all_lons.min())],
        [float(all_lats.max()), float(all_lons.max())],
    ])

    folium.LayerControl().add_to(fmap)

    # Info overlay
    info_html = f"""
    <div style="position:fixed;bottom:28px;left:28px;z-index:1000;
                background:rgba(255,255,255,0.92);padding:10px 14px;
                border-radius:7px;box-shadow:0 2px 8px rgba(0,0,0,0.25);
                font-family:sans-serif;font-size:12px;line-height:1.6;">
        <b style="font-size:13px;">ElephantGraph</b><br>
        Showing {n_show} of {n_total} trajectories &nbsp;|&nbsp; {seq_len} pts each<br>
        &#9679; = start &nbsp; &#9675; = end
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(info_html))

    fmap.save(output_path)
    print(f"Saved map ({n_show}/{n_total} trajectories) -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot generated GPS trajectories on a map")
    parser.add_argument("--input",  required=True,
                        help="Path to .npy trajectories file  (N, T, 2)")
    parser.add_argument("--output", default=None,
                        help="Output HTML path (default: <input>_map.html)")
    parser.add_argument("--n",      type=int, default=20,
                        help="Number of trajectories to plot (default: 20)")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.splitext(args.input)[0] + "_map.html"

    plot_trajectories(args.input, args.output, n_trajs=args.n)
