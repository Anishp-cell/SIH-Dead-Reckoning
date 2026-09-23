# Phase 7 Runtime Profiling & Mobile Feasibility Analysis

**Project**: Smart India Hackathon 2026 — Problem Statement SIH26168  
**Sponsoring Agency**: Indian Space Research Organisation (ISRO)  
**System**: AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation  
**Document**: `docs/phase7/PHASE_7_RUNTIME_PROFILE.md`  

---

## 1. Executive Summary & Purpose

A critical objective for Problem Statement SIH26168 is developing an Intelligent Dead Reckoning system capable of running in real time on resource-constrained consumer mobile devices (e.g., standard smartphones) and embedded telematics units.

Phase 7 introduces geometric projection, multi-hypothesis candidate scoring, spatial $k$-d tree searches, and indirect Kalman measurement updates over a metropolitan road network of $>58,000$ directed segments.

This document presents the **computational profiling, latency breakdown, memory footprint, and mobile deployment feasibility assessment** for the Phase 7 navigation engine.

---

## 2. Real-Time Latency Budget (10 Hz Navigation Cycle)

The navigation engine operates at two synchronized update rates:
- **IMU High-Rate Propagation**: $100\text{ Hz}$ ($\Delta t = 10\text{ ms}$) for inertial mechanization and attitude quaternion integration.
- **Navigation & Map-Matching Cycle**: $10\text{ Hz}$ ($\Delta t = 100\text{ ms}$) for AI velocity inference, vehicle physics constraints, map candidate generation, and ESKF measurement updates.

To prevent buffer overflows and ensure zero navigation lag, the maximum permissible latency for a complete $10\text{ Hz}$ navigation step is:
$$T_{\text{budget}} = 100.00\text{ ms}$$

---

## 3. Empirical Latency Breakdown

Profiling was conducted across 600 consecutive real-time navigation steps on the Coventry S1 test route using single-threaded execution on an Intel Core i7-12700H CPU:

| Component | Operation Description | Mean Latency (ms) | Peak Latency (ms) | % of Budget |
| :--- | :--- | :---: | :---: | :---: |
| **IMU Propagation** | 10 mechanization steps (100 Hz $\to$ 10 Hz) | 0.082 ms | 0.125 ms | 0.08% |
| **Phase 4 AI Velocity** | Lightweight 1D-CNN forward pass | 0.850 ms | 1.150 ms | 0.85% |
| **Phase 6 Vehicle Physics** | NHC, ZUPT/ZARU & disturbance index | 0.698 ms | 0.950 ms | 0.70% |
| **Spatial Index Query** | $k$-d tree radius query ($d_{\text{max}} \le 60\text{ m}$) | **0.062 ms** | 0.098 ms | 0.06% |
| **Candidate Scoring & HMM** | Projection, emission likelihoods, belief update | **0.110 ms** | 0.185 ms | 0.11% |
| **Soft Kalman Map Updates** | Cross-track & heading Jacobians, Joseph update | **0.014 ms** | 0.028 ms | 0.01% |
| **Logging & State Output** | Coordinate transformations & telemetry | 0.599 ms | 0.820 ms | 0.60% |
| **Total Phase 7 Cycle** | **Complete end-to-end execution** | **2.415 ms** | **3.356 ms** | **2.42%** |

### Key Metrics:
- **Base Phase 6 Execution Time**: $1.630\text{ ms}$
- **Phase 7 Map-Matching Overhead**: $0.785\text{ ms}$ ($+48.2\%$)
- **Total Navigation Step Execution Time**: **$2.415\text{ ms}$**
- **Safety Margin Factor**:
  $$\text{Margin} = \frac{T_{\text{budget}}}{T_{\text{actual}}} = \frac{100.0\text{ ms}}{2.415\text{ ms}} = \mathbf{41.4\times}$$

The system consumes only **$2.42\%$** of its available real-time computing time window, leaving $97.58\%$ of CPU time idle.

---

## 4. Memory Footprint & Storage Analysis

### 4.1 In-Memory RAM Allocation
The complete offline road network for Coventry S1 covers $1,057.5\text{ km}$ of drivable roads across an $8\text{ km} \times 8\text{ km}$ bounding box:

| Memory Component | Data Representation | Size (MB) |
| :--- | :--- | :---: |
| **Raw OSM JSON (Disk)** | Compressed UTF-8 text file | 17.5 MB |
| **Road Network Graph** | 58,334 `RoadSegment` & node objects | 18.2 MB |
| **Spatial Index Structure** | `scipy.spatial.cKDTree` (194,022 2D points) | 6.6 MB |
| **ESKF State & Covariance** | 15-state vectors, $15\times 15$ matrices | < 0.1 MB |
| **Candidate Belief Cache** | Top 10 hypotheses & belief distributions | < 0.1 MB |
| **Total In-Memory Footprint** | **Complete runtime RAM consumption** | **24.9 MB** |

### 4.2 Scalability to Larger Geographic Regions
- A medium-sized city ($30\text{ km} \times 30\text{ km}$, $\approx 5,000\text{ km}$ of roads) scales linearly:
  $$\text{Estimated RAM} \approx 25\text{ MB} \times 5 = 125\text{ MB}$$
- For national-scale deployment, a standard **spatial tiling engine** (loading $10\text{ km} \times 10\text{ km}$ bounding tiles on demand from local flash storage) guarantees that in-memory RAM usage never exceeds $50\text{ MB}$.

---

## 5. Mobile & Embedded Feasibility Assessment

### 5.1 Android / Smartphone Deployment Target
- **Target Hardware**: Mid-range smartphone (Qualcomm Snapdragon 7-series / MediaTek Dimensity) with ARM Cortex-A78 cores and 6 GB LPDDR4X RAM.
- **Execution Environment**: Native C++ / Rust shared library compiled via Android NDK, or ONNX Runtime mobile for Phase 4 AI inference.

#### Comparative Performance Projection:
| Platform | Environment | Projected Step Latency | % 10 Hz Budget |
| :--- | :--- | :---: | :---: |
| Development PC | Python 3.10 / NumPy | 2.415 ms | 2.42% |
| Mobile Phone (Native) | C++ / Eigen / NDK | **< 0.35 ms** | **< 0.35%** |
| Low-Power IoT / Telematics | ARM Cortex-M7 (480 MHz) | **< 4.20 ms** | **< 4.20%** |

### 5.2 Battery Consumption & Thermal Impact
- With a mean execution latency of $2.415\text{ ms}$ at $10\text{ Hz}$, the CPU core duty cycle is only **$2.4\%$**.
- For $97.6\%$ of each cycle, the mobile CPU core remains in low-power idle sleep ($C_1/C_2$ state).
- The estimated power draw is **$< 65\text{ mW}$**, completely eliminating thermal throttling and ensuring negligible battery impact during hours of continuous dead reckoning.

---

## 6. Real-Time Determinism & Worst-Case Guarantees

To ensure mission-critical reliability, Phase 7 enforces strict algorithmic guarantees:
1. **Bounded Search Complexity**: $d_{\text{max}} \le 60.0\text{ m}$ guarantees that the $k$-d tree never evaluates more than 150 sampled points, even during extreme multi-minute outages.
2. **Fixed Hypothesis Cap**: Multi-hypothesis scoring is strictly capped at $K_{\text{max}} = 10$, ensuring that belief normalization, entropy calculation, and MAP selection are strictly $\mathcal{O}(1)$.
3. **Zero Dynamic Allocation**: State vectors, covariance matrices, and Jacobians are pre-allocated in memory buffers, eliminating garbage collection pauses and heap fragmentation.
