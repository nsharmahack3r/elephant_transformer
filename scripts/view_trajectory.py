"""
Layered trajectory viewer for ElephantGraph output.

Reads the bundle written by generate_full.py and produces a self-contained
interactive HTML map with independently toggleable layers:

    [x] Coarse hexagons      — the H3 cells the Level-1 model chose
    [x] Coarse path          — line through the extracted anchors
    [x] Anchor points        — the WildGraph-extracted real coordinates
    [x] Fine trajectory      — the Level-2 GPS path
    [x] Start / End markers

Plus: satellite / map basemap toggle, fit-to-bounds, temporal colour gradient
on the coarse cells, hover tooltips, and a legend.

Input:
    - results/full_trajectory.npz  (preferred — has coarse + fine layers)
    - results/full_trajectory.npy  (fine path only; coarse layers omitted)

Usage:
    python scripts/view_trajectory.py
    python scripts/view_trajectory.py --input results/full_trajectory.npz
    python scripts/view_trajectory.py --input results/full_trajectory.npy --output results/view.html
"""

import argparse
import json
import os
import numpy as np

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")


def _coarse_color(i, n):
    """Blue -> green -> red gradient by visit order."""
    t = i / max(n - 1, 1)
    if t < 0.5:
        r, g, b = int(t*2*60), int(120+t*2*40), 200
    else:
        r, g, b = int(60+(t-0.5)*2*195), int(160-(t-0.5)*2*160), int(200-(t-0.5)*2*200)
    return "#{:02x}{:02x}{:02x}".format(r, g, b)


def load_bundle(path, max_fine_pts):
    """Return a dict of JS-ready layers from an .npz bundle or bare .npy."""
    bundle = {"hexes": [], "coarse_path": [], "anchors": [], "fine": []}

    if path.endswith(".npz"):
        data = np.load(path, allow_pickle=True)
        fine = data["fine"]                                  # (T,2) lon,lat
        anchors = data["anchors"]                            # (M,2) lon,lat
        h3s    = data["coarse_h3"]
        dwell  = data["coarse_dwell"]
        bnds   = data["coarse_boundaries"]                   # object (M,) of (V,2) lat,lon

        n = len(h3s)
        for i in range(n):
            poly = [[round(float(la), 6), round(float(lo), 6)] for la, lo in bnds[i]]
            bundle["hexes"].append({
                "poly": poly,
                "color": _coarse_color(i, n),
                "tip": f"Coarse {i+1}/{n} | {int(dwell[i])}h dwell | {h3s[i]}",
            })
        # anchors + coarse path (lat,lon for Leaflet)
        for i in range(len(anchors)):
            lo, la = float(anchors[i][0]), float(anchors[i][1])
            bundle["anchors"].append({
                "ll": [round(la, 6), round(lo, 6)],
                "tip": f"Anchor {i+1} | {h3s[i]}",
            })
        bundle["coarse_path"] = [a["ll"] for a in bundle["anchors"]]
    else:
        arr = np.load(path)
        fine = arr[0] if arr.ndim == 3 else arr

    # Fine path (drop NaNs, subsample for the browser)
    fine = fine[~np.isnan(fine).any(axis=1)]
    T = len(fine)
    if T > max_fine_pts:
        idx = np.unique(np.linspace(0, T - 1, max_fine_pts).round().astype(int))
        fine_r = fine[idx]
    else:
        fine_r = fine
    bundle["fine"] = [[round(float(la), 6), round(float(lo), 6)] for lo, la in fine_r]
    bundle["_fine_total"] = T
    return bundle


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ElephantGraph — trajectory viewer</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html,body{margin:0;height:100%;font-family:-apple-system,Segoe UI,Roboto,sans-serif;}
  #map{position:absolute;top:0;bottom:0;left:0;right:0;}
  #legend{position:absolute;bottom:22px;left:18px;z-index:1000;
          background:rgba(255,255,255,.93);border-radius:9px;padding:10px 14px;
          box-shadow:0 2px 10px rgba(0,0,0,.25);font-size:12px;line-height:1.8;}
  #legend b{font-size:13px;}
  .grad{width:120px;height:9px;border-radius:4px;display:inline-block;vertical-align:middle;
        background:linear-gradient(to right,#3c78c8,#2aa05a,#c83c28);}
  .dot{display:inline-block;width:10px;height:10px;border-radius:50%;vertical-align:middle;}
</style>
</head>
<body>
<div id="map"></div>
<div id="legend">
  <b>&#128024; ElephantGraph</b><br>
  <span id="l_fine"></span> fine GPS pts<br>
  <span id="l_coarse"></span> coarse cells &nbsp;
  <span class="grad"></span> early&#8594;late<br>
  <span class="dot" style="background:#2c7;"></span> start &nbsp;
  <span class="dot" style="background:#e03131;"></span> end<br>
  <span style="color:#888;">Use the layer control (top-right) to show/hide.</span>
</div>

<script>
const D = %%DATA%%;

const map = L.map('map', { zoomControl:true });
const base = L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
  {attribution:'&copy; OpenStreetMap, &copy; CARTO', maxZoom:19}).addTo(map);
const sat = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
  {attribution:'Esri World Imagery', maxZoom:19});

// ── Layer: coarse hexagons ─────────────────────────────────────────────────
const hexLayer = L.layerGroup();
D.hexes.forEach(h => {
  L.polygon(h.poly, {color:h.color, weight:2, fill:true, fillColor:h.color,
                     fillOpacity:0.15}).bindTooltip(h.tip).addTo(hexLayer);
});

// ── Layer: coarse path (through anchors) ───────────────────────────────────
const coarsePathLayer = L.layerGroup();
if (D.coarse_path.length >= 2) {
  L.polyline(D.coarse_path, {color:'#e67e22', weight:3, opacity:0.9,
                             dashArray:'8 6'}).bindTooltip('Coarse path').addTo(coarsePathLayer);
}

// ── Layer: anchor points ───────────────────────────────────────────────────
const anchorLayer = L.layerGroup();
D.anchors.forEach(a => {
  L.circleMarker(a.ll, {radius:4, color:'#e67e22', fill:true, fillColor:'#e67e22',
                        fillOpacity:1}).bindTooltip(a.tip).addTo(anchorLayer);
});

// ── Layer: fine trajectory ─────────────────────────────────────────────────
const fineLayer = L.layerGroup();
L.polyline(D.fine, {color:'#1a1a2e', weight:1.6, opacity:0.75})
  .bindTooltip('Fine GPS trajectory').addTo(fineLayer);

// ── Start / end markers ────────────────────────────────────────────────────
const markerLayer = L.layerGroup();
if (D.fine.length) {
  L.circleMarker(D.fine[0], {radius:7, color:'#2c7', fillColor:'#2c7', fillOpacity:1})
    .bindTooltip('Start').addTo(markerLayer);
  L.circleMarker(D.fine[D.fine.length-1], {radius:7, color:'#e03131', fillColor:'#e03131', fillOpacity:1})
    .bindTooltip('End').addTo(markerLayer);
}

// Add layers (all visible by default)
hexLayer.addTo(map); coarsePathLayer.addTo(map); anchorLayer.addTo(map);
fineLayer.addTo(map); markerLayer.addTo(map);

// Layer control — base + toggleable overlays
const overlays = {
  "&#11041; Coarse hexagons": hexLayer,
  "&#8211; Coarse path":      coarsePathLayer,
  "&#9679; Anchor points":    anchorLayer,
  "&#10141; Fine trajectory": fineLayer,
  "Start / End":              markerLayer,
};
L.control.layers({ "Map": base, "Satellite": sat }, overlays,
                 { collapsed:false }).addTo(map);

// Fit to everything we have
let pts = D.fine.slice();
D.hexes.forEach(h => pts = pts.concat(h.poly));
if (pts.length) map.fitBounds(L.latLngBounds(pts), {padding:[40,40]});

document.getElementById('l_fine').textContent = D._fine_total.toLocaleString();
document.getElementById('l_coarse').textContent = D.hexes.length;
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description="Layered ElephantGraph trajectory viewer")
    default_npz = os.path.join(PROJECT_DIR, "results", "full_trajectory.npz")
    default_npy = os.path.join(PROJECT_DIR, "results", "full_trajectory.npy")
    ap.add_argument("--input", default=None,
                    help="Path to .npz (preferred) or .npy from generate_full.py")
    ap.add_argument("--output", default=None, help="Output HTML (default: <input>_view.html)")
    ap.add_argument("--max-fine-pts", type=int, default=4000,
                    help="Subsample the fine path to this many points for browser speed")
    args = ap.parse_args()

    if args.input is None:
        args.input = default_npz if os.path.exists(default_npz) else default_npy
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"No trajectory found at {args.input}. Run generate_full.py first.")
    if args.output is None:
        args.output = os.path.splitext(args.input)[0] + "_view.html"

    bundle = load_bundle(args.input, args.max_fine_pts)
    html = HTML_TEMPLATE.replace("%%DATA%%", json.dumps(bundle))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)

    src = "npz (coarse + fine)" if args.input.endswith(".npz") else "npy (fine only)"
    print(f"Input: {args.input}  [{src}]")
    print(f"  fine pts: {bundle['_fine_total']:,}  (rendered {len(bundle['fine']):,})")
    print(f"  coarse cells: {len(bundle['hexes'])}  anchors: {len(bundle['anchors'])}")
    print(f"Saved -> {args.output}")
    print("Open it in a browser; use the top-right layer control to show/hide layers.")


if __name__ == "__main__":
    main()
