# Phase 7 System Architecture: Offline OpenStreetMap Map Matching & Road-Constrained Navigation

**Project:** Smart India Hackathon 2026 — Problem Statement SIH26168  
**Sponsoring Agency:** Indian Space Research Organisation (ISRO)  
**System:** AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation  

---

## 1. Architectural Mission & Phase Boundary

Phase 7 introduces an offline spatial prior—the OpenStreetMap (OSM) road network—to bound long-term dead-reckoning cross-track position and heading drift during GNSS-denied navigation. 

### Core Design Axioms
1. **Preserve Phase 6 Integrity:** Phase 7 does not discard or replace the validated Phase 6 vehicle-physics ESKF. Instead, the raw Phase 6 state is continuously propagated and maintained, with map matching operating either as an observer or as an uncertainty-weighted soft measurement source.
2. **100% Offline Runtime Guarantee:** Zero live internet calls, zero Overpass queries, and zero remote tile downloads are executed at runtime. All road vectors and topological relationships are stored in a compact, deterministic local database.
3. **Strict Causality (No Future Information):** Runtime matching is purely causal. The candidate state at time $t$ depends exclusively on the current navigation estimate, local map geometry, and the prior candidate state $c_{t-1}$. Trajectory smoothing, future GNSS lookahead, and whole-trip post-optimization are strictly forbidden at runtime.
4. **No Hard Snapping:** The navigation filter never overrides its position by forcing $p \leftarrow p_{\text{road}}$. Map information enters strictly as soft Kalman observations with adaptive covariance and Chi-square innovation gating.

---

## 2. End-to-End System Block Diagram

```text
       ┌────────────────────────────────────────────────────────┐
       │             OFFLINE PREPROCESSING PIPELINE             │
       │  OSM Source JSON -> Highway Extraction -> ENU Transform │
       │         -> Directed Segments -> cKDTree Index          │
       └───────────────────────────┬────────────────────────────┘
                                   │ (Local Database: 58,334 segments)
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   STREAMING RUNTIME ENGINE (10 Hz)                     │
│                                                                        │
│  [12-ch IMU] ──► IMU Strapdown ──► [Phase 4 AI Speed Inference]        │
│                        │                            │                  │
│                        ▼                            ▼                  │
│             State Prediction (15-State)   Forward Speed Update         │
│                        │                            │                  │
│                        ▼                            ▼                  │
│             Disturbance Detection ──► [Non-Holonomic Constraints (NHC)]│
│                        │                            │                  │
│                        ▼                            ▼                  │
│             Stationary Detection ──►  [ZUPT / ZARU Updates]            │
│                                                     │                  │
│                                                     ▼                  │
│                                           Phase 6 Raw State            │
│                                            (p, v, q, ba, bg, P)        │
│                                                     │                  │
│                        ┌────────────────────────────┴─────────────┐    │
│                        │                                          │    │
│                        ▼                                          ▼    │
│             [Spatial Candidate Query]                     [Phase 6 Log]│
│             d_max = 3*sigma_pos + d_min                                │
│                        │                                               │
│                        ▼                                               │
│             [Candidate Likelihood Scoring]                             │
│             Distance + Heading + Speed                                 │
│                        │                                               │
│                        ▼                                               │
│             [Causal Temporal Tracker]                                  │
│             B_t(c_t) ∝ P(z_t|c_t) * Σ P(c_t|c_{t-1}) B_{t-1}           │
│                        │                                               │
│                        ▼                                               │
│             [Map Confidence Gate]                                      │
│             c_map in [0, 1]                                            │
│                        │                                               │
│                        ▼                                               │
│             [Soft Kalman Updates (M5)]                                 │
│             - Cross-Track: z=0, H=[n_E, n_N, 0, ...]                   │
│             - Road Heading: z=psi_road, H=[0, 0, 1(up), ...]           │
│             - Adaptive Noise: R = R_base / c_map^2                     │
│             - Innovation Gating: NIS <= chi2_1(0.99)                   │
│                        │                                               │
│                        ▼                                               │
│             [Phase 7 Corrected Navigation Output]                      │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Modular Subsystem Decomposition

All Phase 7 modules reside within `Data_details/src/phase7/`:

### 3.1 Map Engine (`map/`)
- **`osm_loader.py`**: Parses offline OSM JSON, extracts drivable highway classes (`motorway`, `trunk`, `primary`, `secondary`, `tertiary`, `residential`, `unclassified`, `service`), interprets one-way rules, parses speed limits, and builds directed segments.
- **`map_database.py`**: Encapsulates `RoadSegment` and `RoadMapDatabase`. For bidirectional roads, both forward and reverse directed segments are instantiated to ensure unambiguous topological transitions.
- **`coordinate_mapper.py`**: Reuses `Data_details.src.coordinate_transform` to convert WGS84 Geodetic coordinates into local metric East-North-Up (ENU) coordinates relative to the S1 origin ($52.401660^\circ\text{ N}, -1.505290^\circ\text{ E}, 147.5\text{ m}$).
- **`geometry.py`**: High-performance 2D projection engine computing orthogonal point-to-segment projection, signed cross-track distance $d_{\text{cross}}$, along-track parameter $s$, tangent azimuth $\psi_{\text{road}}$, and left-pointing normal vector $\mathbf{n}$.
- **`spatial_index.py`**: KD-Tree spatial index implemented via `scipy.spatial.cKDTree` sampling polyline road segments at $\le 10\text{ m}$ intervals. Evaluates uncertainty-adaptive query radii:
  $$d_{\text{max}} = \min(k_{\sigma} \sigma_{\text{pos}} + d_{\text{min}}, d_{\text{clamp}})$$

### 3.2 Probabilistic Matching Engine (`matching/`)
- **`candidate_generator.py`**: Queries the spatial index and computes structured `RoadCandidate` objects containing projections, signed cross-track residuals, along-track fractions, and wrapped heading differences.
- **`candidate_scoring.py`**: Evaluates Gaussian emission likelihoods combining spatial distance, heading compatibility, and dynamic speed constraints.
- **`topology.py`**: Directed graph representation using `networkx.DiGraph`. Computes Markovian transition probabilities $P(c_t | c_{t-1})$ favoring along-road continuation and connected junctions while exponentially penalizing disconnected topological jumps.
- **`temporal_matcher.py`**: Maintains causal recursive candidate belief distribution $B_t(c_t)$. Computes continuous map confidence $c_{\text{map}} \in [0, 1]$ based on candidate dominance, spatial proximity, and heading alignment.
- **`map_measurement.py`**: Formulates scalar cross-track distance ($z_p = 0$) and road tangent heading ($z_\psi = \psi_{\text{road}}$) innovations, derives analytical 15-state ESKF Jacobians ($H_p, H_\psi$), computes adaptive covariance $R = R_0 / c_{\text{map}}^2$, and checks Chi-square innovation gating ($\text{NIS} \le 6.635$).
- **`map_update.py`**: Executes Joseph-stabilized indirect Kalman updates and injects indirect error state $\delta\mathbf{x}$ into the nominal ESKF state.

### 3.3 Core Navigation & Streaming (`core/`, `streaming.py`)
- **`phase7_navigator.py`**: Coordinates the Phase 6 ESKF and Phase 7 Map Matcher. Maintains raw Phase 6 and map-constrained estimates in parallel.
- **`streaming.py`**: Implements `Phase7StreamingEngine` processing incoming 10 Hz IMU samples sequentially.

### 3.4 Evaluation Suite (`evaluation/`, `visualization/`)
- **`map_matching_metrics.py`**: Evaluates cross-track RMSE, along-track RMSE, yaw RMSE, road switch count, map confidence, and innovation gating percentages.
- **`blackout_benchmarks.py`**: Evaluates multi-duration outages (10s, 30s, 60s, 120s) on held-out test splits.
- **`ablation.py`**: Runs M0 through M5 component ablations.
- **`pipeline.py`**: Master batch execution script generating tables and publication-grade plots.
- **`research_replay.py`**: Generates standalone interactive HTML5 research replays with vehicle icons, heading indicators, and live telemetry HUDs.

---

## 4. Architectural Invariants

1. **State Preservation:** The nominal state vector is $[\mathbf{p}, \mathbf{v}, \mathbf{q}, \mathbf{b}_a, \mathbf{b}_g]^T \in \mathbb{R}^{16}$ with indirect error $\delta\mathbf{x} \in \mathbb{R}^{15}$. No state dimension is altered or removed.
2. **Coordinate Standard:** All map geometries, filter positions, and velocities reside in local ENU (East-North-Up). Heading angles represent Cartesian ENU yaw (counter-clockwise from East).
3. **Execution Latency:** Spatial candidate query + temporal matching + soft Kalman update requires $< 0.25\text{ ms}$, ensuring complete cycle latency remains $\approx 2.4\text{ ms}$ (well below the 100 ms 10 Hz budget).
