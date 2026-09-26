"""
Phase 8 Interactive Offline Cartographic Research Replay Visualizer
Renders an offline cartographic OSM basemap (road hierarchy, casings, asphalt fills,
dashed lane lines, metric coordinate grid, scale bar, compass rose)
along with all research navigation overlays (GT, Phase 6 DR, Phase 7 Map DR, Phase 8 Fused,
raw GNSS scatter & covariance, live HUD gauges, and 3-DOF Chi-Square NIS monitor).
100% offline, self-contained HTML5/Canvas application.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase8Replay")


def generate_phase8_research_replay(
    output_html_path: Path,
    road_segments: List[Dict[str, Any]],
    frames_data: List[Dict[str, Any]],
    title: str = "Phase 8 GNSS/INS Fusion & Recovery Simulation (60s Blackout)",
    waypoints_data: Optional[List[Dict[str, Any]]] = None,
):
    """
    Generates a high-fidelity offline cartographic research replay HTML tool
    with Topological Route-Aware Dead Reckoning (RADR) Maneuver Guidance.
    """
    output_html_path = Path(output_html_path)
    output_html_path.parent.mkdir(parents=True, exist_ok=True)

    if waypoints_data is None:
        # Default topological waypoints along the Tile Hill Lane / A45 junction
        waypoints_data = [
            {
                "maneuver_id": "WP_01_LANE_SELECT",
                "type": "LANE_CHANGE_LEFT",
                "pos_enu": [-4462.0, -170.0],
                "turn_angle_deg": -5.0,
                "road_name": "A45 Underpass Approach",
                "action_instruction": "Keep Left (Lane 1) for Tile Hill Underpass",
                "recommended_lanes": [1],
                "total_lanes": 2,
                "trigger_radius_m": 16.0,
            },
            {
                "maneuver_id": "WP_02_TURN_90",
                "type": "TURN_RIGHT_90",
                "pos_enu": [-4479.5, -92.0],
                "turn_angle_deg": 88.5,
                "road_name": "Tile Hill Lane (Eastbound)",
                "action_instruction": "Take Right 90° Turn onto Tile Hill Lane",
                "recommended_lanes": [2],
                "total_lanes": 2,
                "trigger_radius_m": 15.0,
            },
            {
                "maneuver_id": "WP_03_RECOVERY_ZONE",
                "type": "STRAIGHT_CONTINUE",
                "pos_enu": [-4487.4, -38.0],
                "turn_angle_deg": 0.0,
                "road_name": "A45 Underpass Exit",
                "action_instruction": "Underpass Exit: NavIC Re-acquisition Zone",
                "recommended_lanes": [1, 2],
                "total_lanes": 2,
                "trigger_radius_m": 22.0,
            },
        ]

    roads_json = json.dumps(road_segments)
    frames_json = json.dumps(frames_data)
    waypoints_json = json.dumps(waypoints_data)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  :root {{
    --bg-base: #080d1a;
    --bg-surface: #0f172a;
    --bg-card: #1e293b;
    --border-color: #334155;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --accent-cyan: #06b6d4;
    --accent-emerald: #10b981;
    --accent-amber: #f59e0b;
    --accent-rose: #f43f5e;
    --accent-indigo: #6366f1;
  }}

  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 0; background: var(--bg-base); color: var(--text-primary);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    display: flex; flex-direction: column; height: 100vh; overflow: hidden;
  }}

  header {{
    background: var(--bg-surface); padding: 10px 20px; display: flex; justify-content: space-between; align-items: center;
    border-bottom: 1px solid var(--border-color); flex-shrink: 0;
  }}
  .header-left h1 {{ margin: 0; font-size: 1.15rem; color: var(--accent-cyan); font-weight: 700; letter-spacing: -0.01em; }}
  .header-left .subtitle {{ font-size: 0.8rem; color: var(--text-secondary); margin-top: 2px; }}

  .header-badges {{ display: flex; gap: 8px; align-items: center; }}
  .badge {{
    padding: 4px 10px; border-radius: 9999px; font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;
  }}
  .badge-healthy {{ background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid #059669; }}
  .badge-suspect {{ background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid #d97706; }}
  .badge-outage {{ background: rgba(244, 63, 94, 0.2); color: #fb7185; border: 1px solid #e11d48; animation: pulseOutage 1.5s infinite; }}
  .badge-recovering {{ background: rgba(6, 182, 212, 0.2); color: #38bdf8; border: 1px solid #0284c7; animation: pulseRecover 1s infinite; }}

  @keyframes pulseOutage {{
    0%, 100% {{ box-shadow: 0 0 0 0 rgba(244, 63, 94, 0.4); }}
    50% {{ box-shadow: 0 0 0 8px rgba(244, 63, 94, 0); }}
  }}
  @keyframes pulseRecover {{
    0%, 100% {{ box-shadow: 0 0 0 0 rgba(6, 182, 212, 0.4); }}
    50% {{ box-shadow: 0 0 0 8px rgba(6, 182, 212, 0); }}
  }}
  @keyframes pulseManeuver {{
    0%, 100% {{ box-shadow: 0 0 0 0 rgba(6, 182, 212, 0.5); }}
    50% {{ box-shadow: 0 0 0 10px rgba(6, 182, 212, 0); }}
  }}
  @keyframes pulseTurnAlert {{
    0%, 100% {{ box-shadow: 0 0 0 0 rgba(244, 63, 94, 0.6); }}
    50% {{ box-shadow: 0 0 0 12px rgba(244, 63, 94, 0); }}
  }}
  .card-maneuver {{
    border-left: 3px solid var(--accent-cyan) !important;
    background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
  }}

  .main-container {{
    display: flex; flex: 1; position: relative; overflow: hidden;
  }}

  #canvas-container {{
    flex: 1; position: relative; background: #060913; cursor: grab; overflow: hidden;
  }}
  #canvas-container:active {{ cursor: grabbing; }}
  canvas {{ width: 100%; height: 100%; display: block; }}

  /* Sidebar HUD */
  .sidebar {{
    width: 360px; background: var(--bg-surface); border-left: 1px solid var(--border-color);
    padding: 16px; display: flex; flex-direction: column; gap: 14px; font-size: 0.85rem;
    overflow-y: auto; flex-shrink: 0;
  }}

  .card {{
    background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 8px; padding: 12px;
  }}
  .card-title {{
    font-size: 0.75rem; text-transform: uppercase; color: var(--text-muted); font-weight: 700;
    margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center; letter-spacing: 0.05em;
  }}

  .metric-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
  .metric-item {{ display: flex; flex-direction: column; }}
  .metric-label {{ font-size: 0.72rem; color: var(--text-secondary); }}
  .metric-val {{ font-size: 1.05rem; font-weight: 700; font-family: monospace; color: var(--text-primary); }}
  .metric-val.highlight {{ color: var(--accent-cyan); }}
  .metric-val.alert {{ color: var(--accent-rose); }}
  .metric-val.warn {{ color: var(--accent-amber); }}

  /* NIS Bar */
  .nis-bar-container {{
    position: relative; height: 12px; background: #0b1329; border-radius: 6px; overflow: hidden; margin-top: 6px; border: 1px solid #1e293b;
  }}
  .nis-bar-fill {{
    height: 100%; width: 0%; background: linear-gradient(90deg, #10b981 0%, #f59e0b 60%, #ef4444 100%);
    transition: width 0.1s linear;
  }}
  .nis-threshold-mark {{
    position: absolute; top: 0; bottom: 0; width: 2px; background: #ffffff; left: 45.4%; /* 11.345 / 25.0 */
    box-shadow: 0 0 4px #fff;
  }}

  /* Layer Controls */
  .layer-toggles {{ display: flex; flex-direction: column; gap: 6px; font-size: 0.8rem; }}
  .layer-toggle {{ display: flex; align-items: center; gap: 8px; cursor: pointer; user-select: none; }}
  .layer-toggle input {{ cursor: pointer; accent-color: var(--accent-cyan); }}
  .layer-color {{ width: 12px; height: 12px; border-radius: 2px; flex-shrink: 0; }}

  /* Transport Controls */
  .transport {{
    background: var(--bg-surface); border-top: 1px solid var(--border-color); padding: 10px 20px;
    display: flex; flex-direction: column; gap: 8px; flex-shrink: 0;
  }}
  .slider-row {{ display: flex; align-items: center; gap: 12px; }}
  .slider-row input[type=range] {{
    flex: 1; accent-color: var(--accent-cyan); cursor: pointer; height: 5px;
  }}
  .time-display {{ font-family: monospace; font-size: 0.85rem; color: var(--accent-cyan); min-width: 65px; font-weight: 600; }}

  .btn-row {{ display: flex; justify-content: space-between; align-items: center; }}
  .btn-group {{ display: flex; gap: 6px; align-items: center; }}
  button {{
    background: var(--bg-card); border: 1px solid var(--border-color); color: var(--text-primary);
    padding: 6px 12px; border-radius: 6px; font-size: 0.8rem; cursor: pointer; font-weight: 600;
    transition: background 0.15s, border-color 0.15s;
  }}
  button:hover {{ background: #334155; border-color: #475569; }}
  button.active {{ background: var(--accent-cyan); color: #080d1a; border-color: var(--accent-cyan); }}

  /* Canvas Cartographic Overlays */
  .canvas-overlay-scale {{
    position: absolute; bottom: 16px; left: 16px; pointer-events: none;
    background: rgba(15, 23, 42, 0.85); backdrop-filter: blur(4px);
    padding: 6px 12px; border-radius: 6px; border: 1px solid #334155;
    font-family: monospace; font-size: 0.75rem; color: var(--text-secondary);
  }}
  .scale-bar-line {{
    height: 3px; background: #38bdf8; margin-top: 3px; position: relative; border-left: 2px solid #fff; border-right: 2px solid #fff;
  }}

  .canvas-overlay-compass {{
    position: absolute; top: 16px; right: 16px; pointer-events: none;
    width: 44px; height: 44px; background: rgba(15, 23, 42, 0.85); backdrop-filter: blur(4px);
    border-radius: 50%; border: 1px solid #334155; display: flex; flex-direction: column;
    align-items: center; justify-content: center; font-size: 0.65rem; font-weight: 800; color: #ef4444;
  }}
</style>
</head>
<body>

<header>
  <div class="header-left">
    <h1>ISRO SIH26168: Phase 8 Hybrid Navigation Replay</h1>
    <div class="subtitle">Offline Cartographic OSM Basemap • 60s GNSS Outage & Seamless Soft Recovery</div>
  </div>
  <div class="header-badges">
    <div id="badge-gnss" class="badge badge-healthy">GNSS_HEALTHY</div>
    <div id="badge-recovery" class="badge" style="background: rgba(99, 102, 241, 0.15); color: #a5b4fc; border: 1px solid #4f46e5;">SOFT RECOVERY</div>
  </div>
</header>

<div class="main-container">
  <div id="canvas-container">
    <canvas id="map-canvas"></canvas>
    <div class="canvas-overlay-scale">
      <div id="scale-text">100 m</div>
      <div id="scale-bar" class="scale-bar-line" style="width: 80px;"></div>
    </div>
    <div class="canvas-overlay-compass">
      ▲<span style="color:#94a3b8; font-size:0.6rem;">N</span>
    </div>
  </div>

  <div class="sidebar">
    <!-- State Card -->
    <div class="card">
      <div class="card-title">Estimator State & Recovery Mode</div>
      <div class="metric-grid">
        <div class="metric-item">
          <span class="metric-label">GNSS Automaton</span>
          <span id="txt-state" class="metric-val highlight">HEALTHY</span>
        </div>
        <div class="metric-item">
          <span class="metric-label">Recovery Alpha (α)</span>
          <span id="txt-alpha" class="metric-val">1.000</span>
        </div>
        <div class="metric-item">
          <span class="metric-label">Pos Step (Δp)</span>
          <span id="txt-step" class="metric-val">0.00 m</span>
        </div>
        <div class="metric-item">
          <span class="metric-label">Teleport Bound</span>
          <span class="metric-val" style="color:#10b981;">≤ 3.5 m</span>
        </div>
      </div>
    </div>

    <!-- Topological Route Guidance Card (RADR) -->
    <div class="card card-maneuver" id="card-maneuver">
      <div class="card-title" style="color: var(--accent-cyan);">
        <span>Topological Route Guidance (RADR)</span>
        <span id="badge-maneuver-phase" class="badge" style="background: rgba(6, 182, 212, 0.2); color: #38bdf8; border: 1px solid #0284c7;">EN ROUTE</span>
      </div>
      
      <!-- Big Maneuver Visual Banner -->
      <div style="display: flex; align-items: center; gap: 12px; margin: 8px 0; padding: 10px; background: rgba(15, 23, 42, 0.7); border-radius: 8px; border: 1px solid #334155;">
        <div id="maneuver-icon-container" style="width: 48px; height: 48px; border-radius: 8px; background: #080d1a; border: 1px solid #06b6d4; display: flex; align-items: center; justify-content: center; font-size: 1.8rem; color: #38bdf8; flex-shrink: 0;">
          <span id="maneuver-icon-svg">⮱</span>
        </div>
        <div style="flex: 1; display: flex; flex-direction: column; overflow: hidden;">
          <div style="display: flex; align-items: baseline; gap: 6px;">
            <span id="txt-maneuver-dist" style="font-size: 1.5rem; font-weight: 800; font-family: monospace; color: #38bdf8;">85 m</span>
            <span id="txt-maneuver-ttm" style="font-size: 0.78rem; color: #94a3b8;">(~6.2s)</span>
          </div>
          <div id="txt-maneuver-prompt" style="font-size: 0.8rem; font-weight: 600; color: #f8fafc; line-height: 1.2; word-break: break-word;">In 85m: Take Right 90° Turn onto Tile Hill Lane</div>
        </div>
      </div>

      <!-- Lane Indicator Visualizer -->
      <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 6px; padding: 6px 10px; background: #0b1329; border-radius: 6px; border: 1px solid #1e293b;">
        <span style="font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase; font-weight: 600;">Lane Guidance</span>
        <span id="txt-lane-display" style="font-family: monospace; font-size: 0.95rem; font-weight: 700; color: #10b981; letter-spacing: 2px;">[ ⮱ ]  [   ]</span>
      </div>

      <!-- Heading Zero-Drift Calibration Pill -->
      <div id="maneuver-calib-pill" style="display: none; margin-top: 8px; padding: 6px 10px; background: rgba(16, 185, 129, 0.15); border: 1px solid #059669; border-radius: 6px; font-size: 0.74rem; color: #34d399; align-items: center; gap: 6px;">
        <span>✓</span>
        <span id="txt-maneuver-calib">Maneuver Verified via Gyro: Heading Bias Reset (Drift Bounded)</span>
      </div>
    </div>

    <!-- Error Decomposition Card -->
    <div class="card">
      <div class="card-title">Real-Time Trajectory Errors (vs GT)</div>
      <div class="metric-grid">
        <div class="metric-item">
          <span class="metric-label">Along-Track Error (e∥)</span>
          <span id="txt-along" class="metric-val highlight">0.00 m</span>
        </div>
        <div class="metric-item">
          <span class="metric-label">Cross-Track Error (e⊥)</span>
          <span id="txt-cross" class="metric-val">0.00 m</span>
        </div>
        <div class="metric-item">
          <span class="metric-label">Total 2D Error (e2D)</span>
          <span id="txt-e2d" class="metric-val">0.00 m</span>
        </div>
        <div class="metric-item">
          <span class="metric-label">Phase 7 Map Limit</span>
          <span class="metric-val" style="color:#f59e0b;">0.48 m</span>
        </div>
      </div>
    </div>

    <!-- Chi-Square Innovation Card -->
    <div class="card">
      <div class="card-title">
        <span>3-DOF Chi-Square Gating (NIS)</span>
        <span id="txt-nis-val" style="color:#06b6d4; font-family:monospace;">0.00</span>
      </div>
      <div class="nis-bar-container">
        <div id="nis-bar" class="nis-bar-fill"></div>
        <div class="nis-threshold-mark" title="99% Threshold (11.345)"></div>
      </div>
      <div style="display:flex; justify-content:space-between; font-size:0.68rem; color:var(--text-muted); margin-top:4px;">
        <span>0</span>
        <span style="color:#f8fafc;">Threshold: 11.35 (99%)</span>
        <span>25.0 (Cutoff)</span>
      </div>
    </div>

    <!-- Speed Telemetry Card -->
    <div class="card">
      <div class="card-title">Speed Telemetry (m/s)</div>
      <div class="metric-grid">
        <div class="metric-item">
          <span class="metric-label">Reference (GT)</span>
          <span id="txt-speed-gt" class="metric-val">0.0 m/s</span>
        </div>
        <div class="metric-item">
          <span class="metric-label">AI Inference</span>
          <span id="txt-speed-ai" class="metric-val">0.0 m/s</span>
        </div>
        <div class="metric-item">
          <span class="metric-label">Phase 8 Fused Speed</span>
          <span id="txt-speed-fused" class="metric-val highlight">0.0 m/s</span>
        </div>
        <div class="metric-item">
          <span class="metric-label">Raw GNSS Speed</span>
          <span id="txt-speed-gnss" class="metric-val">0.0 m/s</span>
        </div>
      </div>
    </div>

    <!-- Map Matching Card -->
    <div class="card">
      <div class="card-title">OSM Road Topology Constraints</div>
      <div style="font-size:0.8rem; margin-bottom:4px;">
        <span style="color:var(--text-muted);">Active Road:</span>
        <span id="txt-road-name" style="font-weight:600; color:#f8fafc;">A45 Dunchurch Hwy</span>
      </div>
      <div class="metric-grid" style="margin-top:6px;">
        <div class="metric-item">
          <span class="metric-label">Segment ID</span>
          <span id="txt-seg-id" class="metric-val" style="font-size:0.85rem;">seg_412</span>
        </div>
        <div class="metric-item">
          <span class="metric-label">Belief Confidence</span>
          <span id="txt-map-conf" class="metric-val" style="color:#10b981;">0.98</span>
        </div>
      </div>
    </div>

    <!-- Layer Visibility Toggles -->
    <div class="card">
      <div class="card-title">Cartographic & Research Layers</div>
      <div class="layer-toggles">
        <label class="layer-toggle">
          <input type="checkbox" id="chk-roads" checked>
          <div class="layer-color" style="background:#475569;"></div>
          <span>Offline OSM Road Hierarchy</span>
        </label>
        <label class="layer-toggle">
          <input type="checkbox" id="chk-grid" checked>
          <div class="layer-color" style="background:#1e293b;"></div>
          <span>Metric Coordinate Grid (50m)</span>
        </label>
        <label class="layer-toggle">
          <input type="checkbox" id="chk-gt" checked>
          <div class="layer-color" style="background:#f8fafc;"></div>
          <span>Ground Truth (VBOX RTK)</span>
        </label>
        <label class="layer-toggle">
          <input type="checkbox" id="chk-p6" checked>
          <div class="layer-color" style="background:#ef4444;"></div>
          <span>Phase 6 Dead Reckoning</span>
        </label>
        <label class="layer-toggle">
          <input type="checkbox" id="chk-p7" checked>
          <div class="layer-color" style="background:#f59e0b;"></div>
          <span>Phase 7 Map-Constrained DR</span>
        </label>
        <label class="layer-toggle">
          <input type="checkbox" id="chk-p8" checked>
          <div class="layer-color" style="background:#06b6d4;"></div>
          <span>Phase 8 Fused Hybrid Trajectory</span>
        </label>
        <label class="layer-toggle">
          <input type="checkbox" id="chk-gnss" checked>
          <div class="layer-color" style="background:#3b82f6;"></div>
          <span>Raw GNSS Fix Scatter & Uncertainty</span>
        </label>
        <label class="layer-toggle">
          <input type="checkbox" id="chk-maneuvers" checked>
          <div class="layer-color" style="background:#38bdf8;"></div>
          <span>Topological Maneuver Waypoints (RADR)</span>
        </label>
      </div>
    </div>
  </div>
</div>

<div class="transport">
  <div class="slider-row">
    <div class="time-display" id="time-display">00:00.0</div>
    <input type="range" id="time-slider" min="0" max="100" value="0" step="1">
    <div class="time-display" id="time-total" style="text-align:right; color:var(--text-muted);">01:00.0</div>
  </div>
  <div class="btn-row">
    <div class="btn-group">
      <button id="btn-play">▶ Play</button>
      <button id="btn-restart">↺ Restart</button>
      <button id="btn-prev">◀ Step</button>
      <button id="btn-next">Step ▶</button>
    </div>
    <div class="btn-group">
      <span style="font-size:0.75rem; color:var(--text-muted); margin-right:4px;">Speed:</span>
      <button class="btn-speed" data-speed="0.5">0.5x</button>
      <button class="btn-speed active" data-speed="1.0">1x</button>
      <button class="btn-speed" data-speed="2.0">2x</button>
      <button class="btn-speed" data-speed="5.0">5x</button>
      <button class="btn-speed" data-speed="10.0">10x</button>
    </div>
  </div>
</div>

<script>
// Embedded Datasets (100% Offline)
const roadNetwork = {roads_json};
const frames = {frames_json};
const waypoints = {waypoints_json};

const canvas = document.getElementById("map-canvas");
const ctx = canvas.getContext("2d");
const container = document.getElementById("canvas-container");

let currentFrame = 0;
let isPlaying = false;
let playSpeed = 1.0;
let playInterval = null;

// Camera / Viewport State (Meters to Canvas)
let view = {{
  x: 0,        // Center East in meters
  y: 0,        // Center North in meters
  scale: 1.5,  // Pixels per meter
  isDragging: false,
  dragStartX: 0,
  dragStartY: 0,
  autoFollow: true
}};

// Compute bounding box and initial center
function initViewport() {{
  if (frames.length > 0) {{
    view.x = frames[0].ref[0];
    view.y = frames[0].ref[1];
  }}
  resizeCanvas();
}}

function resizeCanvas() {{
  canvas.width = container.clientWidth * window.devicePixelRatio;
  canvas.height = container.clientHeight * window.devicePixelRatio;
  render();
}}
window.addEventListener("resize", resizeCanvas);

// Coordinate transforms: East-North (m) to Canvas (px)
function worldToCanvas(x, y) {{
  const cx = canvas.width / 2;
  const cy = canvas.height / 2;
  // ENU: East is +X, North is +Y. Canvas: +Y is down.
  const px = cx + (x - view.x) * view.scale;
  const py = cy - (y - view.y) * view.scale;
  return [px, py];
}}

function canvasToWorld(px, py) {{
  const cx = canvas.width / 2;
  const cy = canvas.height / 2;
  const x = view.x + (px - cx) / view.scale;
  const y = view.y - (py - cy) / view.scale;
  return [x, y];
}}

// Mouse Interaction: Pan and Zoom
container.addEventListener("mousedown", (e) => {{
  view.isDragging = true;
  view.dragStartX = e.clientX;
  view.dragStartY = e.clientY;
}});

window.addEventListener("mousemove", (e) => {{
  if (!view.isDragging) return;
  const dx = e.clientX - view.dragStartX;
  const dy = e.clientY - view.dragStartY;
  view.dragStartX = e.clientX;
  view.dragStartY = e.clientY;

  view.x -= dx / view.scale;
  view.y += dy / view.scale;
  view.autoFollow = false; // Disable auto-follow when user manually pans
  render();
}});

window.addEventListener("mouseup", () => {{
  view.isDragging = false;
}});

container.addEventListener("wheel", (e) => {{
  e.preventDefault();
  const zoomFactor = e.deltaY < 0 ? 1.15 : 0.85;
  view.scale = Math.max(0.2, Math.min(25.0, view.scale * zoomFactor));
  render();
}});

// Dual-click restores auto-follow
container.addEventListener("dblclick", () => {{
  view.autoFollow = true;
  if (frames[currentFrame]) {{
    view.x = frames[currentFrame].p8[0];
    view.y = frames[currentFrame].p8[1];
  }}
  render();
}});

// Road Hierarchy Styling
function getRoadStyle(type) {{
  switch(type) {{
    case 'motorway':
    case 'trunk':
      return {{ width: 14 * view.scale, casingWidth: 18 * view.scale, fill: '#334155', casing: '#0f172a', dash: true }};
    case 'primary':
    case 'secondary':
      return {{ width: 10 * view.scale, casingWidth: 14 * view.scale, fill: '#283548', casing: '#0f172a', dash: true }};
    case 'tertiary':
    case 'residential':
      return {{ width: 6 * view.scale, casingWidth: 9 * view.scale, fill: '#1e293b', casing: '#080d1a', dash: false }};
    default:
      return {{ width: 4 * view.scale, casingWidth: 6 * view.scale, fill: '#162033', casing: '#080d1a', dash: false }};
  }}
}}

// Main Render Loop
function render() {{
  ctx.save();
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const showRoads = document.getElementById("chk-roads").checked;
  const showGrid = document.getElementById("chk-grid").checked;
  const showGT = document.getElementById("chk-gt").checked;
  const showP6 = document.getElementById("chk-p6").checked;
  const showP7 = document.getElementById("chk-p7").checked;
  const showP8 = document.getElementById("chk-p8").checked;
  const showGNSS = document.getElementById("chk-gnss").checked;
  const showManeuvers = document.getElementById("chk-maneuvers") ? document.getElementById("chk-maneuvers").checked : true;

  const f = frames[currentFrame] || frames[0];

  if (view.autoFollow && f) {{
    view.x = f.p8[0];
    view.y = f.p8[1];
  }}

  // 1. Metric Coordinate Grid
  if (showGrid) {{
    drawMetricGrid();
  }}

  // 2. Offline Cartographic OSM Road Network
  if (showRoads && roadNetwork.length > 0) {{
    drawCartographicRoads();
  }}

  // 2.5 Topological Maneuver Waypoints & Decision Nodes (RADR)
  if (showManeuvers && f) {{
    drawManeuverWaypoints(f);
  }}

  // 3. Trajectory Trails up to currentFrame
  if (showGT) drawTrajectory(frames.map(fr => fr.ref), '#94a3b8', 2.0, [4, 4], currentFrame);
  if (showP6) drawTrajectory(frames.map(fr => fr.p6), '#ef4444', 2.0, null, currentFrame);
  if (showP7) drawTrajectory(frames.map(fr => fr.p7), '#f59e0b', 2.5, null, currentFrame);
  if (showP8) drawTrajectory(frames.map(fr => fr.p8), '#06b6d4', 3.0, null, currentFrame, true);

  // 4. Raw GNSS Fix Scatter & Uncertainty Ellipse
  if (showGNSS && f.gnss_raw) {{
    drawGNSSFix(f);
  }}

  // 5. Vehicle Cursor & Orientation Cone
  if (f) {{
    drawVehicle(f);
  }}

  ctx.restore();
  updateScaleBar();
}}

// Draw Metric Grid
function drawMetricGrid() {{
  const gridSize = 50.0; // 50 meters
  const minWorld = canvasToWorld(0, canvas.height);
  const maxWorld = canvasToWorld(canvas.width, 0);

  const startX = Math.floor(minWorld[0] / gridSize) * gridSize;
  const endX = Math.ceil(maxWorld[0] / gridSize) * gridSize;
  const startY = Math.floor(minWorld[1] / gridSize) * gridSize;
  const endY = Math.ceil(maxWorld[1] / gridSize) * gridSize;

  ctx.lineWidth = 1;
  ctx.strokeStyle = 'rgba(30, 41, 59, 0.4)';
  ctx.fillStyle = 'rgba(71, 85, 105, 0.5)';
  ctx.font = `${{Math.max(10, 8 * window.devicePixelRatio)}}px monospace`;

  for (let x = startX; x <= endX; x += gridSize) {{
    const [px, _] = worldToCanvas(x, 0);
    ctx.beginPath();
    ctx.moveTo(px, 0);
    ctx.lineTo(px, canvas.height);
    ctx.stroke();
    ctx.fillText(`${{Math.round(x)}}m`, px + 4, canvas.height - 8);
  }}

  for (let y = startY; y <= endY; y += gridSize) {{
    const [_, py] = worldToCanvas(0, y);
    ctx.beginPath();
    ctx.moveTo(0, py);
    ctx.lineTo(canvas.width, py);
    ctx.stroke();
    ctx.fillText(`${{Math.round(y)}}m`, 8, py - 4);
  }}
}}

// Draw Cartographic Roads with Casings and Lane Dividers
function drawCartographicRoads() {{
  // Pass 1: Outer Casings
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  for (const seg of roadNetwork) {{
    const style = getRoadStyle(seg.type);
    const [p1x, p1y] = worldToCanvas(seg.p1[0], seg.p1[1]);
    const [p2x, p2y] = worldToCanvas(seg.p2[0], seg.p2[1]);

    ctx.beginPath();
    ctx.lineWidth = style.casingWidth;
    ctx.strokeStyle = style.casing;
    ctx.moveTo(p1x, p1y);
    ctx.lineTo(p2x, p2y);
    ctx.stroke();
  }}

  // Pass 2: Asphalt Fills
  for (const seg of roadNetwork) {{
    const style = getRoadStyle(seg.type);
    const [p1x, p1y] = worldToCanvas(seg.p1[0], seg.p1[1]);
    const [p2x, p2y] = worldToCanvas(seg.p2[0], seg.p2[1]);

    ctx.beginPath();
    ctx.lineWidth = style.width;
    ctx.strokeStyle = style.fill;
    ctx.moveTo(p1x, p1y);
    ctx.lineTo(p2x, p2y);
    ctx.stroke();
  }}

  // Pass 3: Dashed Lane Divider Markings on major roads
  ctx.setLineDash([8, 8]);
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = 'rgba(248, 250, 252, 0.25)';

  for (const seg of roadNetwork) {{
    const style = getRoadStyle(seg.type);
    if (!style.dash) continue;
    const [p1x, p1y] = worldToCanvas(seg.p1[0], seg.p1[1]);
    const [p2x, p2y] = worldToCanvas(seg.p2[0], seg.p2[1]);

    ctx.beginPath();
    ctx.moveTo(p1x, p1y);
    ctx.lineTo(p2x, p2y);
    ctx.stroke();
  }}
  ctx.setLineDash([]);
}}

// Draw Trajectory Line
function drawTrajectory(pts, color, lineWidth, dash, endIdx, glow = false) {{
  if (pts.length < 2) return;
  ctx.save();
  ctx.lineWidth = lineWidth * window.devicePixelRatio;
  ctx.strokeStyle = color;
  if (dash) ctx.setLineDash(dash);

  if (glow) {{
    ctx.shadowColor = color;
    ctx.shadowBlur = 10;
  }}

  ctx.beginPath();
  const maxK = Math.min(pts.length, endIdx + 1);
  for (let k = 0; k < maxK; k++) {{
    const [px, py] = worldToCanvas(pts[k][0], pts[k][1]);
    if (k === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }}
  ctx.stroke();
  ctx.restore();
}}

// Draw Raw GNSS Fix & Uncertainty Ellipse
function drawGNSSFix(f) {{
  if (!f.gnss_raw || isNaN(f.gnss_raw[0])) return;
  const [gx, gy] = worldToCanvas(f.gnss_raw[0], f.gnss_raw[1]);
  const radius = Math.max(4, (f.acc_m || 2.5) * view.scale);

  ctx.save();
  // Uncertainty circle
  ctx.beginPath();
  ctx.arc(gx, gy, radius, 0, Math.PI * 2);
  ctx.fillStyle = f.gnss_accepted ? 'rgba(6, 182, 212, 0.12)' : 'rgba(244, 63, 94, 0.15)';
  ctx.fill();
  ctx.strokeStyle = f.gnss_accepted ? 'rgba(6, 182, 212, 0.4)' : 'rgba(244, 63, 94, 0.5)';
  ctx.setLineDash([3, 3]);
  ctx.stroke();

  // Fix Dot
  ctx.beginPath();
  ctx.arc(gx, gy, 4, 0, Math.PI * 2);
  ctx.fillStyle = f.gnss_accepted ? '#06b6d4' : '#ef4444';
  ctx.fill();
  ctx.restore();
}}

// Draw Dynamic Vehicle Cursor
function drawVehicle(f) {{
  const [vx, vy] = worldToCanvas(f.p8[0], f.p8[1]);
  const yaw = f.p8_yaw; // ENU yaw in radians (0 is East, CCW)

  ctx.save();
  ctx.translate(vx, vy);
  ctx.rotate(-yaw); // Canvas rotation (clockwise)

  // Radar FOV sweep cone
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.arc(0, 0, 35 * window.devicePixelRatio, -Math.PI / 6, Math.PI / 6);
  ctx.closePath();
  ctx.fillStyle = 'rgba(6, 182, 212, 0.08)';
  ctx.fill();

  // Outer vehicle ring
  ctx.beginPath();
  ctx.arc(0, 0, 10 * window.devicePixelRatio, 0, Math.PI * 2);
  ctx.fillStyle = 'rgba(15, 23, 42, 0.9)';
  ctx.fill();
  ctx.lineWidth = 2 * window.devicePixelRatio;
  ctx.strokeStyle = f.state === 'GNSS_OUTAGE' ? '#ef4444' : '#06b6d4';
  ctx.stroke();

  // Heading pointer arrow
  ctx.beginPath();
  ctx.moveTo(14 * window.devicePixelRatio, 0);
  ctx.lineTo(2 * window.devicePixelRatio, -6 * window.devicePixelRatio);
  ctx.lineTo(4 * window.devicePixelRatio, 0);
  ctx.lineTo(2 * window.devicePixelRatio, 6 * window.devicePixelRatio);
  ctx.closePath();
  ctx.fillStyle = '#ffffff';
  ctx.fill();

  ctx.restore();
}}

// Draw Topological Maneuver Waypoints & Decision Nodes (RADR)
function drawManeuverWaypoints(f) {{
  if (!waypoints || waypoints.length === 0) return;

  for (let i = 0; i < waypoints.length; i++) {{
    const wp = waypoints[i];
    const [wpx, wpy] = worldToCanvas(wp.pos_enu[0], wp.pos_enu[1]);

    // Skip if far outside viewport
    if (wpx < -80 || wpx > canvas.width + 80 || wpy < -80 || wpy > canvas.height + 80) continue;

    // Distance from current vehicle position
    const dx = wp.pos_enu[0] - f.p8[0];
    const dy = wp.pos_enu[1] - f.p8[1];
    const dist = Math.sqrt(dx * dx + dy * dy);

    ctx.save();

    // 1. Trigger Radius Circle
    const trigPx = wp.trigger_radius_m * view.scale;
    ctx.beginPath();
    ctx.arc(wpx, wpy, trigPx, 0, Math.PI * 2);
    if (dist <= wp.trigger_radius_m) {{
      ctx.fillStyle = 'rgba(16, 185, 129, 0.22)';
      ctx.strokeStyle = '#10b981';
      ctx.lineWidth = 2 * window.devicePixelRatio;
    }} else if (dist < wp.trigger_radius_m * 3.5) {{
      ctx.fillStyle = 'rgba(6, 182, 212, 0.12)';
      ctx.strokeStyle = '#06b6d4';
      ctx.lineWidth = 1.5 * window.devicePixelRatio;
      ctx.setLineDash([4, 4]);
    }} else {{
      ctx.fillStyle = 'rgba(56, 189, 248, 0.05)';
      ctx.strokeStyle = 'rgba(56, 189, 248, 0.35)';
      ctx.lineWidth = 1 * window.devicePixelRatio;
      ctx.setLineDash([2, 4]);
    }}
    ctx.fill();
    ctx.stroke();
    ctx.setLineDash([]);

    // 2. Center Waypoint Pin / Badge
    const pinRadius = Math.max(7, 9 * window.devicePixelRatio);
    ctx.beginPath();
    ctx.arc(wpx, wpy, pinRadius, 0, Math.PI * 2);
    ctx.fillStyle = dist <= wp.trigger_radius_m ? '#10b981' : '#0284c7';
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2 * window.devicePixelRatio;
    ctx.stroke();

    // 3. Maneuver Symbol inside Pin
    ctx.fillStyle = '#ffffff';
    ctx.font = `bold ${{Math.max(9, 10 * window.devicePixelRatio)}}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    let sym = "•";
    if (wp.type.includes("RIGHT")) sym = "⮱";
    else if (wp.type.includes("LEFT")) sym = "⮰";
    else if (wp.type.includes("UNDERPASS") || wp.type.includes("TUNNEL")) sym = "⮷";
    else if (wp.type.includes("FLYOVER")) sym = "⮵";
    ctx.fillText(sym, wpx, wpy);

    // 4. Floating Waypoint Label Pill
    const labelText = `${{sym}} ${{wp.road_name}} (${{Math.round(dist)}}m)`;
    ctx.font = `bold ${{Math.max(10, 11 * window.devicePixelRatio)}}px -apple-system, BlinkMacSystemFont, sans-serif`;
    const textMetrics = ctx.measureText(labelText);
    const boxW = textMetrics.width + 16;
    const boxH = 22 * window.devicePixelRatio;
    const boxX = wpx - boxW / 2;
    const boxY = wpy - pinRadius - boxH - 6;

    // Pill background
    ctx.beginPath();
    if (ctx.roundRect) {{
      ctx.roundRect(boxX, boxY, boxW, boxH, 4);
    }} else {{
      ctx.rect(boxX, boxY, boxW, boxH);
    }}
    ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
    ctx.fill();
    ctx.strokeStyle = dist <= wp.trigger_radius_m ? '#10b981' : '#334155';
    ctx.lineWidth = 1 * window.devicePixelRatio;
    ctx.stroke();

    // Pill text
    ctx.fillStyle = dist <= wp.trigger_radius_m ? '#34d399' : '#f8fafc';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(labelText, wpx, boxY + boxH / 2);

    ctx.restore();
  }}
}}

// Fallback Maneuver Calculator when raw frames lack explicit maneuver objects
function computeManeuverFallback(f) {{
  if (!waypoints || waypoints.length === 0) return null;
  let activeWp = waypoints[0];
  let minDist = 99999;
  for (const wp of waypoints) {{
    const dx = wp.pos_enu[0] - f.p8[0];
    const dy = wp.pos_enu[1] - f.p8[1];
    const dist = Math.sqrt(dx * dx + dy * dy);
    // In this slice, movement is primarily northward (+y)
    if (dy > -wp.trigger_radius_m) {{
      activeWp = wp;
      minDist = Math.max(0, dist);
      break;
    }}
  }}

  const speed = Math.max(1.5, f.speed_fused || 6.5);
  const ttm = minDist / speed;
  let phase = "EN_ROUTE";
  if (minDist <= activeWp.trigger_radius_m) phase = "EXECUTE_NOW";
  else if (minDist <= 50.0) phase = "PREPARE_MANEUVER";
  else if (minDist <= 150.0) phase = "ADVANCE_ALERT";

  let laneDisp = "[ ⮱ ]  [   ]";
  if (activeWp.type.includes("LEFT")) laneDisp = "[ ⮰ ]  [   ]";
  else if (activeWp.type.includes("UNDERPASS")) laneDisp = "[ ⮷ ]  [   ]";
  else if (activeWp.type.includes("STRAIGHT")) laneDisp = "[ ↑ ]  [ ↑ ]";

  let icon = "turn_right_90";
  if (activeWp.type.includes("LEFT")) icon = "turn_left_90";
  else if (activeWp.type.includes("UNDERPASS")) icon = "underpass_tunnel";
  else if (activeWp.type.includes("STRAIGHT")) icon = "straight";

  return {{
    dist_m: minDist,
    ttm_s: ttm,
    phase: phase,
    prompt: minDist <= activeWp.trigger_radius_m ? `${{activeWp.action_instruction.toUpperCase()}} NOW` : `In ${{Math.round(minDist)}}m: ${{activeWp.action_instruction}}`,
    lane_display: laneDisp,
    icon: icon,
    turn_executed: minDist <= activeWp.trigger_radius_m,
    missed_turn: false,
    divergence_m: 0.0,
  }};
}}

// Update Cartographic Scale Bar
function updateScaleBar() {{
  const targetPx = 80;
  const meters = targetPx / view.scale;
  let niceMeters = 50;
  if (meters > 200) niceMeters = 500;
  else if (meters > 100) niceMeters = 200;
  else if (meters > 40) niceMeters = 100;
  else if (meters > 20) niceMeters = 50;
  else if (meters > 10) niceMeters = 20;
  else niceMeters = 10;

  const actualPx = niceMeters * view.scale;
  document.getElementById("scale-bar").style.width = `${{actualPx}}px`;
  document.getElementById("scale-text").textContent = `${{niceMeters}} m`;
}}

// Update Sidebar HUD & Live Metrics
function updateHUD() {{
  const f = frames[currentFrame] || frames[0];
  if (!f) return;

  // Badges & States
  const stateStr = f.state || f.gnss_state || "GNSS_HEALTHY";
  const badge = document.getElementById("badge-gnss");
  badge.textContent = stateStr;
  badge.className = "badge";
  if (stateStr === "GNSS_HEALTHY") badge.classList.add("badge-healthy");
  else if (stateStr === "GNSS_SUSPECT") badge.classList.add("badge-suspect");
  else if (stateStr === "GNSS_OUTAGE") badge.classList.add("badge-outage");
  else if (stateStr === "GNSS_RECOVERING") badge.classList.add("badge-recovering");

  document.getElementById("txt-state").textContent = stateStr.replace("GNSS_", "");
  const alphaVal = f.alpha !== undefined ? f.alpha : (f.recovery_alpha !== undefined ? f.recovery_alpha : 1.0);
  document.getElementById("txt-alpha").textContent = Number(alphaVal).toFixed(3);
  document.getElementById("txt-step").textContent = `${{(f.p_step || 0).toFixed(2)}} m`;

  // Topological Maneuver Guidance (RADR) Update
  const m = f.maneuver || (computeManeuverFallback ? computeManeuverFallback(f) : null);
  if (m) {{
    const badgeM = document.getElementById("badge-maneuver-phase");
    if (badgeM) {{
      badgeM.textContent = (m.phase || "EN_ROUTE").replace(/_/g, " ");
      badgeM.className = "badge";
      if (m.phase === "EXECUTE_NOW") {{
        badgeM.style.background = "rgba(244, 63, 94, 0.25)";
        badgeM.style.color = "#fb7185";
        badgeM.style.border = "1px solid #e11d48";
        badgeM.style.animation = "pulseTurnAlert 1s infinite";
      }} else if (m.phase === "PREPARE_MANEUVER") {{
        badgeM.style.background = "rgba(245, 158, 11, 0.25)";
        badgeM.style.color = "#fbbf24";
        badgeM.style.border = "1px solid #d97706";
        badgeM.style.animation = "pulseManeuver 1.5s infinite";
      }} else if (m.phase === "MANEUVER_COMPLETED") {{
        badgeM.style.background = "rgba(16, 185, 129, 0.25)";
        badgeM.style.color = "#34d399";
        badgeM.style.border = "1px solid #059669";
        badgeM.style.animation = "none";
      }} else if (m.phase === "MISSED_MANEUVER") {{
        badgeM.style.background = "rgba(239, 68, 68, 0.3)";
        badgeM.style.color = "#f87171";
        badgeM.style.border = "1px solid #dc2626";
        badgeM.style.animation = "pulseTurnAlert 0.8s infinite";
      }} else {{
        badgeM.style.background = "rgba(6, 182, 212, 0.15)";
        badgeM.style.color = "#38bdf8";
        badgeM.style.border = "1px solid #0284c7";
        badgeM.style.animation = "none";
      }}
    }}

    const distElem = document.getElementById("txt-maneuver-dist");
    if (distElem) distElem.textContent = m.dist_m > 900 ? "--" : `${{Math.round(m.dist_m)}} m`;
    const ttmElem = document.getElementById("txt-maneuver-ttm");
    if (ttmElem) ttmElem.textContent = m.ttm_s > 0 && m.ttm_s < 300 ? `(~${{m.ttm_s.toFixed(1)}}s)` : "";
    const promptElem = document.getElementById("txt-maneuver-prompt");
    if (promptElem) promptElem.textContent = m.prompt || "Continue on route";
    const laneElem = document.getElementById("txt-lane-display");
    if (laneElem) laneElem.textContent = m.lane_display || "[ ↑ ]  [   ]";

    const iconElem = document.getElementById("maneuver-icon-svg");
    if (iconElem) {{
      if (m.icon === "turn_right_90" || m.icon === "turn_right_slight") iconElem.textContent = "⮱";
      else if (m.icon === "turn_left_90" || m.icon === "turn_left_slight") iconElem.textContent = "⮰";
      else if (m.icon === "underpass_tunnel") iconElem.textContent = "⮷";
      else if (m.icon === "flyover_ramp") iconElem.textContent = "⮵";
      else iconElem.textContent = "↑";
    }}

    const calibPill = document.getElementById("maneuver-calib-pill");
    if (calibPill) {{
      if (m.turn_executed) {{
        calibPill.style.display = "flex";
        calibPill.style.background = "rgba(16, 185, 129, 0.15)";
        calibPill.style.borderColor = "#059669";
        calibPill.style.color = "#34d399";
        document.getElementById("txt-maneuver-calib").textContent = "Maneuver Verified via Gyro: Heading Bias Reset (Drift Clamped)";
      }} else if (m.missed_turn) {{
        calibPill.style.display = "flex";
        calibPill.style.background = "rgba(239, 68, 68, 0.2)";
        calibPill.style.borderColor = "#dc2626";
        calibPill.style.color = "#f87171";
        document.getElementById("txt-maneuver-calib").textContent = `DIVERGENCE WARNING: Missed Turn (+${{(m.divergence_m || 0).toFixed(1)}}m cross-track)`;
      }} else {{
        calibPill.style.display = "none";
      }}
    }}
  }}

  // Errors
  document.getElementById("txt-along").textContent = `${{(f.along_err || 0).toFixed(2)}} m`;
  document.getElementById("txt-cross").textContent = `${{(f.cross_err || 0).toFixed(2)}} m`;
  const totErr = f.total_err !== undefined ? f.total_err : (f.e2d || 0);
  document.getElementById("txt-e2d").textContent = `${{Number(totErr).toFixed(2)}} m`;

  // NIS
  const nisVal = f.gnss_nis || 0.0;
  document.getElementById("txt-nis-val").textContent = nisVal.toFixed(2);
  const nisPct = Math.min(100, (nisVal / 25.0) * 100);
  document.getElementById("nis-bar").style.width = `${{nisPct}}%`;

  // Speeds
  document.getElementById("txt-speed-gt").textContent = `${{(f.speed_gt || 0).toFixed(1)}} m/s`;
  document.getElementById("txt-speed-ai").textContent = `${{(f.speed_ai || 0).toFixed(1)}} m/s`;
  document.getElementById("txt-speed-fused").textContent = `${{(f.speed_fused || 0).toFixed(1)}} m/s`;
  document.getElementById("txt-speed-gnss").textContent = `${{(f.speed_gnss || 0).toFixed(1)}} m/s`;

  // Road Info
  document.getElementById("txt-road-name").textContent = f.road_name || "A45 Dunchurch Hwy";
  document.getElementById("txt-seg-id").textContent = f.road_id || "seg_412";
  document.getElementById("txt-map-conf").textContent = (f.map_conf !== undefined ? f.map_conf : 0.95).toFixed(2);

  // Time slider & display
  document.getElementById("time-slider").value = currentFrame;
  const mm = Math.floor(f.t / 60).toString().padStart(2, '0');
  const ss = (f.t % 60).toFixed(1).padStart(4, '0');
  document.getElementById("time-display").textContent = `${{mm}}:${{ss}}`;
}}

// Transport Controls & Playback
const btnPlay = document.getElementById("btn-play");
const timeSlider = document.getElementById("time-slider");

function togglePlay() {{
  isPlaying = !isPlaying;
  btnPlay.textContent = isPlaying ? "⏸ Pause" : "▶ Play";
  if (isPlaying) {{
    startPlayback();
  }} else {{
    clearInterval(playInterval);
  }}
}}

function startPlayback() {{
  clearInterval(playInterval);
  const intervalMs = 100 / playSpeed; // 10 Hz nominal
  playInterval = setInterval(() => {{
    if (currentFrame < frames.length - 1) {{
      currentFrame++;
      updateHUD();
      render();
    }} else {{
      isPlaying = false;
      btnPlay.textContent = "▶ Play";
      clearInterval(playInterval);
    }}
  }}, intervalMs);
}}

btnPlay.addEventListener("click", togglePlay);
document.getElementById("btn-restart").addEventListener("click", () => {{
  currentFrame = 0;
  updateHUD();
  render();
}});
document.getElementById("btn-prev").addEventListener("click", () => {{
  if (currentFrame > 0) {{
    currentFrame--;
    updateHUD();
    render();
  }}
}});
document.getElementById("btn-next").addEventListener("click", () => {{
  if (currentFrame < frames.length - 1) {{
    currentFrame++;
    updateHUD();
    render();
  }}
}});

timeSlider.max = frames.length - 1;
timeSlider.addEventListener("input", (e) => {{
  currentFrame = parseInt(e.target.value);
  updateHUD();
  render();
}});

// Speed buttons
document.querySelectorAll(".btn-speed").forEach(btn => {{
  btn.addEventListener("click", (e) => {{
    document.querySelectorAll(".btn-speed").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    playSpeed = parseFloat(btn.getAttribute("data-speed"));
    if (isPlaying) startPlayback();
  }});
}});

// Layer checkbox change triggers render
document.querySelectorAll(".layer-toggle input").forEach(inp => {{
  inp.addEventListener("change", render);
}});

// Initialize
initViewport();
updateHUD();
render();
</script>

</body>
</html>
"""
    output_html_path.write_text(html_content, encoding="utf-8")
    logger.info(f"Generated standalone cartographic research replay HTML: {output_html_path}")
