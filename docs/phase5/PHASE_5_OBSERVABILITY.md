# Phase 5 Observability & Information Analysis

**Project**: Smart India Hackathon 2026 — Problem Statement SIH26168  
**Sponsoring Agency**: Indian Space Research Organisation (ISRO)  
**System**: AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation  
**Phase**: Phase 5 — 15-State Error-State Kalman Filter Navigation Core  
**Document Code**: `SIH26168-OBSERV-PHASE5-01`  
**Classification**: CONTROL THEORY & OBSERVABILITY SPECIFICATION  

---

## 1. Observability Matrix & Subspace Decomposition

### Technical Derivation
A continuous or discrete linear time-varying system $(\mathbf{F}_k, \mathbf{H}_k)$ is locally observable over $N$ steps if the discrete observability Gramian $\mathcal{O}_N$ has full rank ($\text{rank}(\mathcal{O}_N) = n = 15$):
$$\mathcal{O}_N = \begin{bmatrix} \mathbf{H}_0 \\ \mathbf{H}_1 \mathbf{F}_0 \\ \mathbf{H}_2 \mathbf{F}_1 \mathbf{F}_0 \\ \vdots \\ \mathbf{H}_{N-1} \mathbf{F}_{N-2} \dots \mathbf{F}_0 \end{bmatrix} \in \mathbb{R}^{N \times 15}$$

When the system is excited only by inertial propagation and a scalar forward speed measurement $z_k = \mathbf{e}_1^T \mathbf{R}_{nb}^T \mathbf{v}_n$, the rank of $\mathcal{O}$ is **strictly less than 15**.

---

## 2. State-by-State Observability Breakdown

### 2.1 Position States ($\delta \mathbf{p}_n \in \mathbb{R}^3$): Completely Unobservable
- **Mathematical Cause**: The measurement equation $h(\mathbf{x}) = \mathbf{e}_1^T \mathbf{R}_{nb}^T \mathbf{v}_n$ has zero partial derivatives with respect to position:
  $$\frac{\partial h}{\partial \delta \mathbf{p}_n} = \mathbf{0}_{1 \times 3}$$
  In the state transition matrix $\mathbf{F}_k$, position never feeds back into velocity, attitude, or biases (rows 1–4 of column 0 are zero).
- **Physical Meaning**: Speed tells you how fast you are moving, but never where on Earth you started. The translation origin remains fundamentally unobservable without GNSS, map anchors, or visual landmarks.

### 2.2 Velocity States ($\delta \mathbf{v}_n \in \mathbb{R}^3$): Highly Observable
- **Mathematical Cause**: $\mathbf{H}[0, 3:6] = \mathbf{R}_{nb}[:, 0]^T$. The forward speed update projects directly onto the navigation velocity vector.
- **Physical Meaning**: Longitudinal vehicle velocity is continuously pinned to the AI forward speed, arresting runaway velocity divergence.

### 2.3 Tilt Attitude States ($\delta \theta_{\text{roll}}, \delta \theta_{\text{pitch}}$): Weakly to Moderately Observable
- **Mathematical Cause**: Pitch tilt couples Earth gravity into longitudinal acceleration:
  $$\dot{v}_{\text{fwd}} \approx a_{\text{meas}} + g \sin\theta$$
  During vehicle acceleration transients ($\dot{v} \ne 0$), discrepancies between predicted speed and measured speed allow the filter to infer road incline (pitch). Roll tilt couples centripetal acceleration during turns:
  $$a_{\text{lat}} \approx v \cdot \omega_{\text{yaw}} + g \sin\phi$$

### 2.4 Yaw Heading State ($\delta \theta_{\text{yaw}}$): Locally Unobservable during Straight Cruising
- **Mathematical Cause**: Rotating the entire ENU coordinate system around the vertical $z_n$ axis leaves forward speed scalar magnitude completely invariant.
- **Physical Meaning**: **The forward speed measurement cannot stabilize gyro heading drift during straight driving.**
- **During Dynamic Turns**: Yaw rate $\omega_{\text{yaw}}$ couples forward speed into centripetal lateral acceleration, providing weak nonlinear heading observability. However, without a magnetometer, Non-Holonomic Constraints (Phase 6), or GNSS (Phase 8), absolute heading will slowly drift.

---

## 3. Simple Explanation (10-Year-Old Level)
Imagine running with your eyes closed on a flat football field:
- A smart watch tells you: *"You are jogging at exactly 5 meters per second."*
- Do you know your speed? **Yes!**
- Do you know where you are on the field? **No!** (You could be near the goal post or near the bleachers).
- Do you know which direction you are facing? **Not really!** You could be running East or North, and your speedometer still says 5 m/s.
This is why a speedometer alone stops you from accidentally running at supersonic speed, but cannot stop you from slowly curving off course!

---

## 4. Why This Matters for Our Project
This mathematical proof establishes the exact boundary of what Phase 5 can achieve and why Phase 6 is required:
- **Phase 5 Achieves**: Velocity stabilization and quadratic error arrest ($\mathcal{O}(t^2) \to \mathcal{O}(t)$).
- **Phase 5 Cannot Solve**: Unconstrained yaw drift during long straight blackouts.
- **Phase 6 Requirement**: Non-Holonomic Constraints ($v_{\text{lateral}} = 0, v_{\text{up}} = 0$) and zero-velocity updates (ZUPT) must be introduced to mechanically bind the vehicle trajectory to the road plane.
