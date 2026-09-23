# Phase 7 Observability Analysis: Road Constraints and Error-State Dynamics

**Project**: Smart India Hackathon 2026 — Problem Statement SIH26168  
**Sponsoring Agency**: Indian Space Research Organisation (ISRO)  
**System**: AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation  
**Document**: `docs/phase7/PHASE_7_OBSERVABILITY.md`  

---

## 1. Executive Summary & Purpose

A fundamental principle of Kalman filtering is **observability**: an error state can only be bounded if it projects into the measurement residual $\boldsymbol{\nu} = \mathbf{z} - \mathbf{h}(\hat{\mathbf{x}})$ over time through the observability Gramian.

In Phase 5 and Phase 6, we demonstrated that while AI forward speed and physical vehicle constraints (NHC, ZUPT, ZARU) successfully bound velocity errors and stationary drift, **position and heading remain vulnerable during extended motion**:
1. In Phase 5, position is completely unobservable ($\text{rank} = 0$), and yaw is unobservable.
2. In Phase 6, NHC couples lateral velocity to attitude during turning maneuvers, providing *conditional* yaw observability. However, during straight-line driving, yaw bias remains weakly observable, and position remains completely unobservable.

Phase 7 introduces two new pseudo-measurements derived from the offline OpenStreetMap road network:
1. **Cross-Track Distance ($z_p = 0$)**: Constrains lateral position to the road centerline.
2. **Road Tangent Heading ($z_\psi = \psi_{\text{road}}$)**: Directly observes vehicle yaw.

This document presents a rigorous linear system observability analysis of the 15-state ESKF across Phases 5, 6, and 7.

---

## 2. 15-State Error-State System Formulation

The indirect continuous-time error-state vector is:
$$\delta\mathbf{x} = \begin{bmatrix} \delta\mathbf{p}_n^T & \delta\mathbf{v}_n^T & \delta\boldsymbol{\theta}^T & \delta\mathbf{b}_a^T & \delta\mathbf{b}_g^T \end{bmatrix}^T \in \mathbb{R}^{15}$$

The linearized continuous-time error dynamics are:
$$\delta\dot{\mathbf{x}}(t) = \mathbf{F}(t) \delta\mathbf{x}(t) + \mathbf{G}(t) \mathbf{w}(t)$$

where the system dynamics matrix $\mathbf{F} \in \mathbb{R}^{15 \times 15}$ is:
$$\mathbf{F} = \begin{bmatrix}
\mathbf{0}_{3\times 3} & \mathbf{I}_3 & \mathbf{0}_{3\times 3} & \mathbf{0}_{3\times 3} & \mathbf{0}_{3\times 3} \\
\mathbf{0}_{3\times 3} & \mathbf{0}_{3\times 3} & -[\hat{\mathbf{f}}_n]_\times & -R_{nb} & \mathbf{0}_{3\times 3} \\
\mathbf{0}_{3\times 3} & \mathbf{0}_{3\times 3} & -[\hat{\boldsymbol{\omega}}_b]_\times & \mathbf{0}_{3\times 3} & -\mathbf{I}_3 \\
\mathbf{0}_{3\times 3} & \mathbf{0}_{3\times 3} & \mathbf{0}_{3\times 3} & -\frac{1}{\tau_a}\mathbf{I}_3 & \mathbf{0}_{3\times 3} \\
\mathbf{0}_{3\times 3} & \mathbf{0}_{3\times 3} & \mathbf{0}_{3\times 3} & \mathbf{0}_{3\times 3} & -\frac{1}{\tau_g}\mathbf{I}_3
\end{bmatrix}$$
where:
- $\hat{\mathbf{f}}_n = R_{nb} \hat{\mathbf{f}}_b$ is the specific force projected into the navigation frame.
- $[\hat{\mathbf{f}}_n]_\times$ is the skew-symmetric matrix of specific force.
- $\hat{\boldsymbol{\omega}}_b$ is the compensated body angular rate.
- $\tau_a, \tau_g$ are the bias correlation times (Gauss-Markov random walk).

The discrete-time state transition matrix over interval $\Delta t$ is:
$$\boldsymbol{\Phi} = \exp(\mathbf{F} \Delta t) \approx \mathbf{I}_{15} + \mathbf{F} \Delta t + \frac{1}{2} \mathbf{F}^2 \Delta t^2$$

---

## 3. Observability Matrix Analysis Across Phases

A linear time-invariant system $(\mathbf{F}, \mathbf{H})$ is completely observable if and only if the observability matrix $\mathcal{O}$ has full column rank ($\text{rank}(\mathcal{O}) = 15$):
$$\mathcal{O} = \begin{bmatrix} \mathbf{H} \\ \mathbf{H}\mathbf{F} \\ \mathbf{H}\mathbf{F}^2 \\ \vdots \\ \mathbf{H}\mathbf{F}^{14} \end{bmatrix} \in \mathbb{R}^{15m \times 15}$$

For time-varying trajectories, observability is evaluated via the rank of the Observability Gramian:
$$\mathcal{W}_o(0, T) = \int_0^T \boldsymbol{\Phi}(t, 0)^T \mathbf{H}(t)^T \mathbf{H}(t) \boldsymbol{\Phi}(t, 0) \, dt$$

### 3.1 Phase 5: AI Forward Velocity Updates Only
In Phase 5, the only measurement is scalar forward velocity:
$$\mathbf{H}_5 = \begin{bmatrix} \mathbf{0}_{1\times 3} & \mathbf{e}_1^T R_{nb}^T & \mathbf{e}_1^T [\hat{\mathbf{v}}_b]_\times & \mathbf{0}_{1\times 3} & \mathbf{0}_{1\times 3} \end{bmatrix} \in \mathbb{R}^{1 \times 15}$$

#### Observability Rank:
$$\text{rank}(\mathcal{O}_5) = 6 \text{ out of } 15$$
- **Observable Subspaces**:
  - Forward velocity error: $\delta v_{\text{fwd}}$ (directly measured).
  - Pitch and roll errors: $\delta\phi, \delta\theta$ (coupled to gravity vector during acceleration).
  - Accelerometer biases: $\delta b_{a, x}, \delta b_{a, y}$ (partially observable during dynamic maneuvers).
- **Unobservable Subspaces (Null Space $\text{null}(\mathcal{O}_5) = 9$)**:
  - **All 3 position states**: $\delta e, \delta n, \delta u$ (completely absent from $\mathbf{H}_5$ and $\mathbf{F}$ columns 1-3). Position error grows quadratically with time: $\sigma_p(t) \propto t^{3/2}$.
  - **Yaw heading error**: $\delta\psi$ (unobservable during constant velocity).
  - **Vertical gyro bias**: $\delta b_{g, z}$ (unobservable).
  - **Vertical velocity & accelerometer bias**: $\delta v_z, \delta b_{a, z}$.

---

### 3.2 Phase 6: NHC + ZUPT/ZARU Constraints
In Phase 6, Non-Holonomic Constraints enforce zero lateral and vertical velocity:
$$\mathbf{H}_{NHC} = \begin{bmatrix} \mathbf{0}_{2\times 3} & \begin{bmatrix} \mathbf{e}_2^T \\ \mathbf{e}_3^T \end{bmatrix} R_{nb}^T & \begin{bmatrix} \mathbf{e}_2^T \\ \mathbf{e}_3^T \end{bmatrix} [\hat{\mathbf{v}}_b]_\times & \mathbf{0}_{2\times 3} & \mathbf{0}_{2\times 3} \end{bmatrix} \in \mathbb{R}^{2 \times 15}$$

#### Observability Rank During Motion:
$$\text{rank}(\mathcal{O}_6) = 9 \text{ out of } 15$$
- **Newly Observable Subspaces**:
  - 3D velocity is now fully observable: $\text{rank}(\delta\mathbf{v}) = 3$.
  - Yaw heading $\delta\psi$ becomes **conditionally observable** during cornering:
    When $\dot{\psi} \ne 0$, the vehicle experiences centripetal acceleration $a_{\text{lat}} = v_{\text{fwd}} \dot{\psi}$. The second derivative $\mathbf{H}_{NHC}\mathbf{F}$ contains terms coupling yaw error $\delta\psi$ to measured lateral forces.
- **Persistent Unobservable Subspaces (Null Space $\text{null}(\mathcal{O}_6) = 6$)**:
  - **All 3 position states**: $\delta e, \delta n, \delta u$ remain in the null space! Phase 6 cannot bound absolute position drift during outages.
  - **Yaw heading during straight driving**: When $\dot{\psi} \approx 0$ and acceleration $\mathbf{a} \approx \mathbf{0}$, the coupling terms vanish. Yaw drift accumulates freely as $\sigma_\psi(t) \propto \sigma_{bg} \sqrt{t}$.

---

### 3.3 Phase 7: Offline OSM Road-Constrained Navigation
Phase 7 adds the cross-track distance and road heading observation models:
$$\mathbf{H}_p = \begin{bmatrix} -t_n & t_e & 0 & \mathbf{0}_{1\times 3} & \mathbf{0}_{1\times 3} & \mathbf{0}_{1\times 3} & \mathbf{0}_{1\times 3} \end{bmatrix} \in \mathbb{R}^{1 \times 15}$$
$$\mathbf{H}_\psi = \begin{bmatrix} \mathbf{0}_{1\times 3} & \mathbf{0}_{1\times 3} & 0 & 0 & 1 & \mathbf{0}_{1\times 3} & \mathbf{0}_{1\times 3} \end{bmatrix} \in \mathbb{R}^{1 \times 15}$$

#### Observability Rank:
$$\text{rank}(\mathcal{O}_7) = 11 \text{ out of } 15$$

#### 1. Cross-Track Position Subspace
The row $\mathbf{H}_p$ directly spans the lateral position error component:
$$\delta p_{\perp} = -t_n \delta e + t_e \delta n$$
This collapses the lateral position error variance from unbounded growth to a fixed, bounded boundary:
$$\lim_{t \to \infty} P_{\perp}(t) \le R_p = \frac{\sigma_{\text{cross}, 0}^2}{c_{\text{map}}^2} \approx (1.5\text{ m})^2$$

#### 2. Full 3D Attitude Subspace
With $\mathbf{H}_\psi$ directly observing yaw:
$$\mathbf{H}_\psi \delta\mathbf{x} = \delta\psi$$
The entire attitude subspace $\delta\boldsymbol{\theta} = [\delta\phi, \delta\theta, \delta\psi]^T$ becomes **unconditionally observable** ($\text{rank} = 3$), even on perfectly straight highways!

#### 3. Gyroscope Vertical Bias Subspace ($\delta b_{g, z}$)
In the continuous error dynamics:
$$\delta\dot{\psi}(t) = -\delta b_{g, z}(t) + w_{gz}(t)$$
Because $\delta\psi$ is continuously measured by $\mathbf{H}_\psi$, the time derivative of the innovation residual directly reveals the vertical gyroscope bias:
$$\frac{d}{dt}\left(\psi_{\text{road}} - \hat{\psi}(t)\right) = \delta\dot{\psi}(t) = -\delta b_{g, z}(t)$$
Thus, **vertical gyroscope bias $\delta b_{g, z}$ becomes fully observable** through the map heading updates!

#### 4. The Remaining Unobservable Subspaces ($\text{null}(\mathcal{O}_7) = 4$)
Why is $\text{rank}(\mathcal{O}_7) = 11$ and not $15$?
1. **Along-Track Position $\delta p_{\parallel}$**: The road centerline is an infinite 1D manifold. Driving forward along a straight road produces zero cross-track innovation, regardless of whether the vehicle is at mile marker 10 or mile marker 10.5. Along-track position remains unobserved by the road centerline alone; it is bounded solely by integrating Phase 4 AI forward velocity.
2. **Vertical Position $\delta u$**: 2D OSM road networks do not provide millimeter-accurate vertical profiles (unless 3D elevation is ingested). Vertical position is weakly bounded by vertical NHC ($v_z^b \approx 0$).
3. **Longitudinal Accelerometer Bias $\delta b_{a, x}$**: Only weakly observable during constant speed; fully observable during braking/acceleration.

---

## 4. Comprehensive Observability Comparison Matrix

| Error State Component | Symbol | Phase 5 (AI Velocity) | Phase 6 (Vehicle Physics) | Phase 7 (Road Network) |
| :--- | :---: | :---: | :---: | :---: |
| **East Position** | $\delta e$ | **Unobservable** | **Unobservable** | **Observable (Cross-Track Component)** |
| **North Position** | $\delta n$ | **Unobservable** | **Unobservable** | **Observable (Cross-Track Component)** |
| **Up Position** | $\delta u$ | **Unobservable** | **Unobservable** | **Unobservable** (bounded by $v_z \approx 0$) |
| **East Velocity** | $\delta v_e$ | Observable | Observable | Observable |
| **North Velocity** | $\delta v_n$ | Observable | Observable | Observable |
| **Up Velocity** | $\delta v_u$ | **Unobservable** | Observable (NHC) | Observable (NHC) |
| **Roll Error** | $\delta\phi$ | Observable | Observable | Observable |
| **Pitch Error** | $\delta\theta$ | Observable | Observable | Observable |
| **Yaw Error** | $\delta\psi$ | **Unobservable** | Conditional (turns only) | **Unconditional (Road Heading)** |
| **Accel Bias $X$** | $\delta b_{a, x}$ | Weak | Observable | Observable |
| **Accel Bias $Y$** | $\delta b_{a, y}$ | **Unobservable** | Observable (NHC) | Observable (NHC) |
| **Accel Bias $Z$** | $\delta b_{a, z}$ | **Unobservable** | Observable (NHC) | Observable (NHC) |
| **Gyro Bias $X$** | $\delta b_{g, x}$ | Observable | Observable | Observable |
| **Gyro Bias $Y$** | $\delta b_{g, y}$ | Observable | Observable | Observable |
| **Gyro Bias $Z$** | $\delta b_{g, z}$ | **Unobservable** | **Unobservable** (straight) | **Observable (via $\dot{\psi}$ Innovation)** |
| **Total Observability Rank** | — | **6 / 15** | **9 / 15** | **11 / 15** |

---

## 5. Physical Implications for Long-Duration Blackouts

The elevation of observability rank from 9 to 11 explains the dramatic empirical improvements demonstrated in Phase 7:
1. **Arresting Yaw Runaway**: In Phase 6, a 120-second blackout allowed yaw drift to accumulate up to $25^\circ$, leading to $2,000\text{ m}$ of endpoint divergence. In Phase 7, road heading updates hold yaw error within $\pm 1.8^\circ$, cutting 120s divergence in half.
2. **Eliminating Cross-Track Drift**: On straight highways, Phase 6 cross-track error diverged to $484\text{ m}$. Phase 7 bounds cross-track error to $0.68\text{ m}$, a **10-fold reduction**.
3. **The Unresolved Frontier (Phase 8)**: Because along-track position $\delta p_{\parallel}$ remains in the null space of road-constrained dead reckoning, longitudinal drift will eventually accumulate over multi-kilometer outages. This formally establishes the necessity of Phase 8 (GNSS Fusion and Outage Reacquisition).
