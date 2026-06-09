"""
Animated map player: watch an elephant walk a generated trajectory.

Reads the GPS path produced by generate_full.py (results/full_trajectory.npy,
shape (1, T, 2) or (T, 2), columns = lon, lat) and writes a self-contained
interactive HTML file.  An elephant emoji moves along the path in real time,
leaving a growing trail, with play / pause / speed / scrub controls and a
satellite basemap toggle.

No web server needed — just open the HTML in a browser.

Usage:
    python scripts/animate_trajectory.py
    python scripts/animate_trajectory.py --input results/full_trajectory.npy
    python scripts/animate_trajectory.py --frames 2000 --output results/walk.html
"""

import argparse
import json
import os
import numpy as np

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")

FINE_STEP_SEC = 10   # seconds represented by each original GPS point


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ElephantGraph — trajectory walk</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html, body { margin:0; height:100%; font-family:-apple-system,Segoe UI,Roboto,sans-serif; }
  #map { position:absolute; top:0; bottom:0; left:0; right:0; }
  .elephant-icon { font-size:26px; line-height:26px; text-align:center;
                   filter:drop-shadow(0 0 3px rgba(0,0,0,.6)); }
  #panel { position:absolute; bottom:18px; left:50%; transform:translateX(-50%);
           z-index:1000; background:rgba(255,255,255,.95); border-radius:12px;
           box-shadow:0 4px 18px rgba(0,0,0,.3); padding:12px 16px;
           display:flex; flex-direction:column; gap:8px; width:min(560px,92vw); }
  #row1 { display:flex; align-items:center; gap:12px; }
  #playBtn { font-size:18px; width:42px; height:38px; border:none; border-radius:8px;
             background:#2c7; color:white; cursor:pointer; }
  #playBtn:hover { background:#2b6; }
  #scrub { flex:1; }
  .meta { font-size:12px; color:#333; white-space:nowrap; }
  #row2 { display:flex; align-items:center; gap:10px; font-size:12px; color:#444; }
  #speed { width:160px; }
  #info { position:absolute; top:14px; left:14px; z-index:1000;
          background:rgba(255,255,255,.92); border-radius:9px; padding:9px 13px;
          box-shadow:0 2px 9px rgba(0,0,0,.25); font-size:12px; line-height:1.6; }
  #info b { font-size:13px; }
</style>
</head>
<body>
<div id="map"></div>

<div id="info">
  <b>&#128024; ElephantGraph walk</b><br>
  <span id="i_pts"></span> GPS points &nbsp;|&nbsp; <span id="i_dur"></span><br>
  <span id="i_clock"></span>
</div>

<div id="panel">
  <div id="row1">
    <button id="playBtn">&#9658;</button>
    <input id="scrub" type="range" min="0" max="100" value="0" step="0.01">
    <span class="meta"><span id="pct">0</span>%</span>
  </div>
  <div id="row2">
    <span>Speed</span>
    <input id="speed" type="range" min="1" max="400" value="60">
    <span class="meta"><span id="spd">60</span>x</span>
    <span style="flex:1"></span>
    <label><input id="follow" type="checkbox" checked> follow</label>
  </div>
</div>

<script>
const PATH      = %%COORDS%%;        // [[lat,lon], ...]
const STEP_SEC  = %%STEP_SEC%%;      // simulated seconds between consecutive points
const N         = PATH.length;
const TOTAL_SEC = (N - 1) * STEP_SEC;

// ── Map ────────────────────────────────────────────────────────────────────
const map = L.map('map', { zoomControl:true });
const base = L.tileLayer(
  'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
  { attribution:'&copy; OpenStreetMap, &copy; CARTO', maxZoom:19 }).addTo(map);
const sat = L.tileLayer(
  'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
  { attribution:'Esri World Imagery', maxZoom:19 });
L.control.layers({ "Map": base, "Satellite": sat }).addTo(map);

const bounds = L.latLngBounds(PATH);
map.fitBounds(bounds, { padding:[40,40] });

// Faint full route, growing trail, start/end markers
L.polyline(PATH, { color:'#888', weight:1.5, opacity:0.5, dashArray:'4 5' }).addTo(map);
const trail = L.polyline([], { color:'#e8590c', weight:3, opacity:0.9 }).addTo(map);

L.circleMarker(PATH[0],   { radius:7, color:'#2c7', fillColor:'#2c7', fillOpacity:1 })
  .addTo(map).bindTooltip('Start');
L.circleMarker(PATH[N-1], { radius:7, color:'#e03131', fillColor:'#e03131', fillOpacity:1 })
  .addTo(map).bindTooltip('End');

const elephant = L.marker(PATH[0], {
  icon: L.divIcon({ className:'', html:'<div class="elephant-icon">&#128024;</div>',
                    iconSize:[26,26], iconAnchor:[13,13] })
}).addTo(map);

// ── Interpolation along the path by fractional index ───────────────────────
function lerp(a, b, f) { return [a[0]+(b[0]-a[0])*f, a[1]+(b[1]-a[1])*f]; }
function posAt(idxFloat) {
  const i = Math.floor(idxFloat);
  if (i >= N-1) return PATH[N-1];
  return lerp(PATH[i], PATH[i+1], idxFloat - i);
}
function fmtClock(sec) {
  sec = Math.round(sec);
  const d = Math.floor(sec/86400); sec -= d*86400;
  const h = Math.floor(sec/3600);  sec -= h*3600;
  const m = Math.floor(sec/60);
  return (d>0? d+'d ':'') + String(h).padStart(2,'0')+'h '+String(m).padStart(2,'0')+'m';
}

// ── Playback state ─────────────────────────────────────────────────────────
let cursor   = 0;        // fractional index into PATH
let playing  = false;
let speed    = 60;       // simulated seconds per real second base; * slider
let lastTs   = null;

const playBtn = document.getElementById('playBtn');
const scrub   = document.getElementById('scrub');
const pctEl   = document.getElementById('pct');
const speedEl = document.getElementById('speed');
const spdEl   = document.getElementById('spd');
const followEl= document.getElementById('follow');
const clockEl = document.getElementById('i_clock');

document.getElementById('i_pts').textContent = N.toLocaleString();
document.getElementById('i_dur').textContent = fmtClock(TOTAL_SEC) + ' simulated';

function render() {
  const p = posAt(cursor);
  elephant.setLatLng(p);
  const upTo = PATH.slice(0, Math.floor(cursor)+1);
  upTo.push(p);
  trail.setLatLngs(upTo);
  const frac = cursor/(N-1);
  scrub.value = (frac*100).toFixed(2);
  pctEl.textContent = Math.round(frac*100);
  clockEl.textContent = 'Elapsed: ' + fmtClock(frac*TOTAL_SEC);
  if (followEl.checked && playing && !map.getBounds().contains(p)) map.panTo(p);
}

function tick(ts) {
  if (!playing) return;
  if (lastTs === null) lastTs = ts;
  const dtReal = (ts - lastTs)/1000;     // real seconds since last frame
  lastTs = ts;
  // advance: speed (sim sec / real sec) -> fractional indices
  const dIdx = (dtReal * speed) / STEP_SEC;
  cursor = Math.min(N-1, cursor + dIdx);
  render();
  if (cursor >= N-1) { stop(); return; }
  requestAnimationFrame(tick);
}

function play() { if (cursor>=N-1) cursor=0; playing=true; lastTs=null;
                  playBtn.innerHTML='&#10074;&#10074;'; requestAnimationFrame(tick); }
function stop() { playing=false; playBtn.innerHTML='&#9658;'; }

playBtn.onclick = () => playing ? stop() : play();
scrub.oninput   = () => { cursor = (scrub.value/100)*(N-1); render(); };
speedEl.oninput = () => { speed = +speedEl.value; spdEl.textContent = speed; };

render();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description="Animate an elephant walking a generated trajectory")
    ap.add_argument("--input",  default=os.path.join(PROJECT_DIR, "results", "full_trajectory.npy"),
                    help="Path to .npy from generate_full.py  ((1,T,2) or (T,2), lon/lat)")
    ap.add_argument("--output", default=None, help="Output HTML (default: <input>_walk.html)")
    ap.add_argument("--frames", type=int, default=1500,
                    help="Max animation frames (path is subsampled to this for smoothness)")
    ap.add_argument("--step-sec", type=float, default=FINE_STEP_SEC,
                    help="Simulated seconds between consecutive original GPS points")
    args = ap.parse_args()

    if args.output is None:
        args.output = os.path.splitext(args.input)[0] + "_walk.html"

    data = np.load(args.input)
    if data.ndim == 3:          # (N_traj, T, 2) -> take the first trajectory
        data = data[0]
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(f"Expected (T,2) or (1,T,2) lon/lat array, got {data.shape}")

    # Drop any NaN rows
    data = data[~np.isnan(data).any(axis=1)]
    T_full = len(data)
    if T_full < 2:
        raise ValueError("Trajectory has fewer than 2 valid points.")

    # Subsample for smooth playback; scale step-sec so total duration is preserved
    if T_full > args.frames:
        idx = np.linspace(0, T_full - 1, args.frames).round().astype(int)
        idx = np.unique(idx)
        path = data[idx]
        effective_step = args.step_sec * (T_full - 1) / (len(path) - 1)
    else:
        path = data
        effective_step = args.step_sec

    # npy columns are (lon, lat); Leaflet wants [lat, lon]
    latlon = [[round(float(lat), 6), round(float(lon), 6)] for lon, lat in path]

    total_hours = (T_full - 1) * args.step_sec / 3600.0
    html = (HTML_TEMPLATE
            .replace("%%COORDS%%",   json.dumps(latlon))
            .replace("%%STEP_SEC%%", repr(float(effective_step))))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Loaded {T_full:,} GPS points  (~{total_hours/24:.1f} simulated days)")
    print(f"Animation frames: {len(path):,}  (step {effective_step:.1f}s each)")
    print(f"Saved -> {args.output}")
    print("Open it in a browser and press play.")


if __name__ == "__main__":
    main()
