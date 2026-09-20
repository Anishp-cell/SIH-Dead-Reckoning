# Phase 5 System Audit: Upstream Conventions, Interfaces, and Reusability

**Project**: Smart India Hackathon 2026 — Problem Statement SIH26168  
**Sponsoring Agency**: Indian Space Research Organisation (ISRO)  
**System**: AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation  
**Phase**: Phase 5 — 15-State Error-State Kalman Filter Navigation Core  
**Document Code**: `SIH26168-AUDIT-PHASE5-01`  
**Classification**: MANDATORY SYSTEM AUDIT & CONVENTION RECONCILIATION  

---

## 1. Upstream Components & Reusability Matrix

| Component | Source File | Convention | Units | Frame | Input | Output | Reusable in Phase 5? |
|:----------|:------------|:-----------|:------|:------|:------|:-------|:---------------------|
| **Coordinate Transform** | `Data_details/src/coordinate_transform.py` | Local ENU (WGS84 $\to$ ECEF $\to$ ENU tangent plane) | meters ($m$) | Navigation (ENU: $x$=East, $y$=North, $z$=Up) | Geodetic $(\phi, \lambda, h)$ | $(x_{\text{east}}, y_{\text{north}}, z_{\text{up}})$ | **YES** (Ground truth reference for post-hoc evaluation) |
| **Attitude & Quaternions** | `Data_details/src/attitude_estimation.py` | Scalar-first $[q_w, q_x, q_y, q_z]$, Hamiltonian product, ZYX Euler | radians / degrees | Body $\to$ Nav ($\mathbf{v}_n = \mathbf{R}_{nb} \mathbf{v}_b$) | Angular rate $\boldsymbol{\omega}$, Accelerometer $\mathbf{a}$ | Quaternion $\mathbf{q}$, DCM $\mathbf{R}_{nb}$ | **YES** (Quaternion math, small-angle error injection) |
| **Phone-Vehicle Alignment**| `Data_details/src/phone_vehicle_alignment.py` | ISO 8855 adapted: $x_v$=Fwd, $y_v$=Right, $z_v$=Up | $\text{m/s}^2, \text{rad/s}$ | Phone Body $\to$ Vehicle Frame | Phone IMU $(\mathbf{a}_p, \boldsymbol{\omega}_p)$ | Vehicle IMU $(\mathbf{a}_v, \boldsymbol{\omega}_v)$ | **YES** (Pre-aligned in Phase 3) |
| **Gravity Removal** | `Data_details/src/gravity_compensation.py` | Earth gravity $\mathbf{g}_n = [0, 0, -9.80665]^T$ | $\text{m/s}^2$ | Navigation (ENU) | Specific force $\mathbf{f}_n$ | Dynamic acceleration $\mathbf{a}_n = \mathbf{f}_n + \mathbf{g}_n$ | **YES** (INS nominal acceleration dynamics) |
| **Inertial Baseline** | `Data_details/src/inertial_baseline.py` | Naive trapezoidal double integration | $m, \text{m/s}$ | 2D horizontal plane | Unfiltered IMU | Baseline trajectory | **YES** (E0 reference baseline) |
| **Causal Signal Conditioning** | `Data_details/outputs/phase3/processed/s1_filtered_causal_imu.csv` | 2nd-order Causal Butterworth low-pass, strictly backward | $\text{m/s}^2, \text{rad/s}$ | Vehicle Frame ($x$=Fwd, $y$=Right, $z$=Up) | Raw smartphone IMU | Preconditioned 12-channel telemetry | **YES** (Primary input stream A for Phase 5) |
| **Phase 4 Motion Engine** | `Data_details/src/phase4/streaming.py` | Causal rolling-buffer inference (`MotionEstimate`) | $\text{m/s}$ | Vehicle longitudinal axis ($x_v$) | 12-channel feature window (30x12) | $\hat{v}_{\text{fwd}}, \sigma_v, \text{state}$ | **YES** (Primary input stream B for Phase 5) |
| **Phase 4 Checkpoint** | `Data_details/outputs/phase4/models/phase4_uncertainty.pt` | Heteroscedastic Dilated TCN | $\text{m/s}$ | Forward speed $\hat{v}$, variance $\sigma_v^2$ | Normalized 12-ch features | Speed & variance predictions | **YES** (Authoritative Phase 4 AI model) |

---

## 2. Exact Physical Units & Constants

1. **Inertial Kinematics**:
   - Accelerometer specific force $\mathbf{f}_b$: Meters per second squared ($\text{m/s}^2$).
   - Gyroscope angular rate $\boldsymbol{\omega}_{ib}^b$: Radians per second ($\text{rad/s}$).
   - Standard Earth gravity magnitude: $g_0 = 9.80665\text{ m/s}^2$.
   - Navigation frame gravity vector: $\mathbf{g}_n = [0.0, 0.0, -9.80665]^T\text{ m/s}^2$.
2. **Timestamps**:
   - Sensor time $t$: Continuous seconds ($s$) from run epoch.
   - Integration interval $\Delta t = t_k - t_{k-1}$: Nominal $0.100\text{ s}$ ($10.0\text{ Hz}$).
3. **AI Speed & Uncertainty**:
   - Estimated forward speed $\hat{v}_{\text{fwd}}$: Meters per second ($\text{m/s}$). Non-negative ($\hat{v} \ge 0$).
   - Speed uncertainty $\sigma_v$: Standard deviation in meters per second ($\text{m/s}$).
   - Scalar measurement variance $R_k$: $(\text{m/s})^2 = \text{m}^2/\text{s}^2$.

---

## 3. Coordinate Frame Specifications

### 3.1 Navigation Frame ($n$-frame): Local ENU
- **Origin**: Tangent point at initial GNSS fix: $(\phi_0, \lambda_0, h_0) = (52.401660^\circ, -1.505290^\circ, 147.5\text{ m})$.
- **Axes**:
  - $x_n$: Local **East**
  - $y_n$: Local **North**
  - $z_n$: Local **Up** (normal to WGS84 ellipsoid)
- **Gravity Direction**: Directed toward Earth center, along $-z_n$:
  $$\mathbf{g}_n = \begin{bmatrix} 0 \\ 0 \\ -9.80665 \end{bmatrix}\text{ m/s}^2$$

### 3.2 Vehicle Body Frame ($b$-frame): ISO 8855 Adapted
- **Origin**: Vehicle center of gravity / rigid smartphone mount location.
- **Axes**:
  - $x_b$: **Forward** along longitudinal vehicle centerline (direction of forward driving).
  - $y_b$: **Lateral Right** toward passenger door.
  - $z_b$: **Vertical Up** orthogonal to road plane (pointing up through roof).
- **Measurement Mapping**:
  Phase 3 preconditioned signals are already aligned with this frame:
  $$\mathbf{f}_b = \begin{bmatrix} a_{\text{fwd}} \\ a_{\text{lat}} \\ a_{\text{up}} \end{bmatrix}, \quad \boldsymbol{\omega}_{ib}^b = \begin{bmatrix} \omega_{\text{roll}} \\ \omega_{\text{pitch}} \\ \omega_{\text{yaw}} \end{bmatrix}$$

---

## 4. Quaternion & Rotation Formalism

1. **Representation**: Unit quaternion with scalar-first convention:
   $$\mathbf{q} = \begin{bmatrix} q_w \\ q_x \\ q_y \\ q_z \end{bmatrix} = \begin{bmatrix} \cos(\theta / 2) \\ \mathbf{u} \sin(\theta / 2) \end{bmatrix}, \quad \|\mathbf{q}\| = 1$$
2. **Direction Cosine Matrix $\mathbf{R}_{nb}$**:
   Transforms vectors from Vehicle Body frame ($b$) to Navigation frame ($n$):
   $$\mathbf{v}_n = \mathbf{R}_{nb}(\mathbf{q}) \mathbf{v}_b$$
   $$\mathbf{R}_{nb} = \begin{bmatrix}
   1 - 2(q_y^2 + q_z^2) & 2(q_x q_y - q_w q_z) & 2(q_x q_z + q_w q_y) \\
   2(q_x q_y + q_w q_z) & 1 - 2(q_x^2 + q_z^2) & 2(q_y q_z - q_w q_x) \\
   2(q_x q_z - q_w q_y) & 2(q_y q_z + q_w q_x) & 1 - 2(q_x^2 + q_y^2)
   \end{bmatrix}$$
3. **Inverse Transformation**:
   $$\mathbf{R}_{bn} = \mathbf{R}_{nb}^T$$
   $$\mathbf{v}_b = \mathbf{R}_{bn} \mathbf{v}_n = \mathbf{R}_{nb}^T \mathbf{v}_n$$

---

## 5. Critical Convention Reconciliation: Cartesian Yaw vs Geographic Heading

### The Mathematical Ambiguity
In navigation datasets (including IO-VNBD GPS and VBOX telemetry), geographic heading $\psi_{\text{GPS}}$ is defined as:
- $0^\circ = \text{North}$
- $90^\circ = \text{East}$
- $180^\circ = \text{South}$
- $270^\circ = \text{West}$
- Measured **clockwise** from true North.

However, in standard 3D Cartesian coordinates on an ENU plane ($x_n = \text{East}$, $y_n = \text{North}$), standard Euler yaw $\psi_{\text{ENU}}$ is defined as:
- $0^\circ = \text{East}$ ($+x_n$)
- $+90^\circ = \text{North}$ ($+y_n$)
- Measured **counter-clockwise** from East.

### Rigorous Mathematical Mapping
When the vehicle is heading along geographic azimuth $\psi_{\text{GPS}}$, its unit direction vector in ENU coordinates is:
$$\mathbf{u}_{\text{veh}} = \begin{bmatrix} \sin \psi_{\text{GPS}} \\ \cos \psi_{\text{GPS}} \\ 0 \end{bmatrix}$$
Under Cartesian rotation by $\psi_{\text{ENU}}$ about $+z_n$, body forward unit vector $[1, 0, 0]^T$ transforms to:
$$\mathbf{R}_{nb} \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix} = \begin{bmatrix} \cos \psi_{\text{ENU}} \\ \sin \psi_{\text{ENU}} \\ 0 \end{bmatrix}$$
Equating the two yields the exact bijection:
$$\cos \psi_{\text{ENU}} = \sin \psi_{\text{GPS}}, \quad \sin \psi_{\text{ENU}} = \cos \psi_{\text{GPS}}$$
$$\psi_{\text{ENU}} = \frac{\pi}{2} - \psi_{\text{GPS}} = 90^\circ - \psi_{\text{GPS}}$$
$$\psi_{\text{GPS}} = \frac{\pi}{2} - \psi_{\text{ENU}} = 90^\circ - \psi_{\text{ENU}}$$

**Implementation Rule**: When initializing attitude from GPS heading at blackout entry ($t_0$), the initial Euler yaw must be converted via $\psi_{\text{ENU}} = 90^\circ - \psi_{\text{GPS}}$.

---

## 6. Phase 4 AI Speed Measurement Interface

The Phase 4 module exports a streaming causal inference engine:
```python
@dataclass
class MotionEstimate:
    timestamp: float          # Epoch timestamp (s)
    forward_speed_mps: float  # Estimated forward velocity v_fwd (m/s)
    speed_uncertainty: float  # Standard deviation sigma_v (m/s)
    motion_state: str         # STANDSTILL, CRUISING, ACCELERATING, BRAKING, TURNING
    motion_confidence: float  # Normalized score in [0, 1]
```

### Measurement Model Formulation
Forward vehicle speed $z_{v, k} = \hat{v}_{\text{fwd}, k}$ observes the longitudinal component ($x_b$) of navigation velocity $\mathbf{v}_n$:
$$\mathbf{v}_b = \mathbf{R}_{bn} \mathbf{v}_n = \mathbf{R}_{nb}^T \mathbf{v}_n$$
$$h(\mathbf{x}) = \mathbf{e}_1^T \mathbf{R}_{nb}^T \mathbf{v}_n = \begin{bmatrix} 1 & 0 & 0 \end{bmatrix} \mathbf{R}_{nb}^T \mathbf{v}_n$$
$$z_{v, k} = h(\mathbf{x}_k) + r_k, \quad r_k \sim \mathcal{N}(0, R_k)$$
$$R_k = \sigma_{v, k}^2$$

---

## 7. Available Sequences & Ground Truth Isolation

1. **Moving Test Sequence**: IO-VNBD Sequence S1 ($51,746\text{ samples}$, $86.24\text{ min}$, $37.25\text{ km}$).
   - Training block: Samples $0 \to 36,220$ ($0 \to 3,622.0\text{ s}$).
   - Validation block: Samples $36,271 \to 43,980$ ($3,627.1 \to 4,398.0\text{ s}$).
   - **Test block (Held-Out)**: Samples $44,031 \to 51,745$ ($4,403.1 \to 5,174.5\text{ s}$, $741.5\text{ s}$ duration).
   - **Blackout Evaluation Region**: Simulated GNSS outages will be placed **strictly inside the held-out test block ($t \ge 4,403.1\text{ s}$)**.
2. **Ground Truth Strict Isolation**:
   - Runtime filter inputs: **ONLY** preconditioned IMU $(\mathbf{f}_b, \boldsymbol{\omega}_b)$ + Phase 4 AI estimate $(\hat{v}, \sigma_v)$.
   - Vehicle CAN speed, GPS coordinates, and VBOX telemetry are **NEVER** provided to the ESKF at runtime. They are used exclusively post-hoc to compute trajectory error metrics.

---

## 8. Proposed Phase 5 Directory & File Structure

```text
Data_details/src/phase5/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── state.py              # 15-state nominal and error state definitions
│   ├── quaternion.py         # Quaternion math, SO(3) mappings, small-angle error injection
│   ├── frames.py             # ENU and ISO 8855 frame transformations
│   ├── propagation.py        # 0.1s nominal state & 15x15 covariance propagation
│   ├── measurement.py        # AI forward speed measurement model, Jacobian H, NIS gating
│   └── eskf.py               # Complete 15-state Error-State Kalman Filter core
├── models/
│   ├── __init__.py
│   └── imu_noise.py          # Process noise Q construction, discrete IMU noise modeling
├── evaluation/
│   ├── __init__.py
│   ├── trajectory_metrics.py # Endpoint error, drift %, RMSE, velocity/heading metrics
│   ├── consistency.py        # NIS and NEES filter consistency analysis
│   └── blackout_benchmarks.py# E0 (Raw), E1 (ESKF-no-AI), E2 (ESKF+AI) benchmarks
├── streaming.py              # Causal sequential real-time navigation engine
└── pipeline.py               # End-to-end master evaluation script

Data_details/tests/phase5/
├── __init__.py
├── test_quaternion.py        # Identity, inversion, composition, small-angle error tests
├── test_frames.py            # ENU <-> Body roundtrip, gravity vector tests
├── test_propagation.py       # Nominal INS constant acceleration/velocity tests
├── test_jacobians.py         # Analytical F and H vs numerical finite-difference checks
├── test_covariance.py        # Covariance positive definiteness, Joseph form symmetry
├── test_measurement.py       # Speed update, NIS gating, outlier rejection tests
└── test_streaming.py         # Real-time sample-by-sample causal streaming tests

docs/phase5/
├── PHASE_5_SYSTEM_AUDIT.md
├── PHASE_5_MATHEMATICAL_FOUNDATION.md
├── PHASE_5_FRAME_CONVENTION.md
├── PHASE_5_NOISE_MODEL.md
├── PHASE_5_INITIALIZATION.md
├── PHASE_5_OBSERVABILITY.md
├── PHASE_5_VALIDATION.md
├── PHASE_5_FAILURE_ANALYSIS.md
└── PHASE_5_FINAL_REPORT.md
```
