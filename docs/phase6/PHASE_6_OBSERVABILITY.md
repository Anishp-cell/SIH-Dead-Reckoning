# Phase 6 Observability Analysis: What Vehicle Physics Can and Cannot Observe

**Project**: Smart India Hackathon 2026 — Problem Statement SIH26168  
**Sponsoring Agency**: Indian Space Research Organisation (ISRO)  
**System**: AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation  
**Document**: `docs/phase6/PHASE_6_OBSERVABILITY.md`  

---

## 1. Technical Derivation: Lie Derivative Observability

Consider the continuous non-linear navigation system with state $\mathbf{x} = [\mathbf{p}_n^T, \mathbf{v}_n^T, \mathbf{q}^T, \mathbf{b}_a^T, \mathbf{b}_g^T]^T \in \mathbb{R}^{16}$.

### 1.1 Observability of Forward Speed Alone (Phase 5)
Observation: $y_1 = \mathbf{e}_1^T R_{nb}^T \mathbf{v}_n$.
- Directly observes: Forward velocity component $v_x^b$.
- Unobservable subspaces:
  - Absolute horizontal position $\mathbf{p}_{xy}^n$: Unobservable (system dynamics and measurements depend only on relative displacement $\dot{\mathbf{p}} = \mathbf{v}$).
  - Lateral velocity $v_y^b$ and vertical velocity $v_z^b$: Unobservable under zero angular rate.
  - Absolute yaw angle $\psi$: Unobservable during straight constant-speed motion (any rotation around gravity preserves $\mathbf{e}_1^T R_{nb}^T \mathbf{v}_n$).
  - Sensor biases: Only observable under active rotational and translational excitation.

---

### 1.2 Observability with Non-Holonomic Constraints (NHC)
Observation: $\mathbf{y}_{NHC} = \begin{bmatrix} \mathbf{e}_2^T \\ \mathbf{e}_3^T \end{bmatrix} R_{nb}^T \mathbf{v}_n$.
- Directly observes: Lateral velocity $v_y^b$ and vertical velocity $v_z^b$.
- Couples into Attitude Error:
  $$\frac{\partial v_y^b}{\partial \delta\theta_z} = -v_{\text{fwd}}$$
  When forward velocity $v_{\text{fwd}} > 0$, the yaw error $\delta\theta_z$ becomes **observable through the lateral velocity constraint** during turns and maneuvers!
- Couples into Accelerometer Bias:
  $$\frac{\partial \dot{v}_y^b}{\partial b_{a,y}} = -1$$
  Lateral and vertical accelerometer biases $\mathbf{b}_{a, y}, \mathbf{b}_{a, z}$ become observable because any persistent acceleration offset would otherwise cause lateral velocity to diverge from zero.

---

### 1.3 Observability with Zero-Velocity Updates (ZUPT)
Observation: $\mathbf{y}_{ZUPT} = \mathbf{v}_n = \mathbf{0}$.
- Directly observes: 3D navigation velocity $\mathbf{v}_n$.
- Observability of Accelerometer Bias $\mathbf{b}_a$:
  Since $\dot{\mathbf{v}}_n = \mathbf{0}$ during a standstill:
  $$\mathbf{0} = R_{nb} (\mathbf{f}_b - \mathbf{b}_a) + \mathbf{g}_n \implies \mathbf{b}_a = \mathbf{f}_b + R_{nb}^T \mathbf{g}_n$$
  The 3D accelerometer bias $\mathbf{b}_a$ is **completely observable** during stationary periods by comparing measured specific force against known Earth gravity.
- Roll and Pitch Observability: Gravity leveling makes roll $\phi$ and pitch $\theta$ directly observable.

---

### 1.4 Observability with Zero-Angular-Rate Updates (ZARU)
Observation: $\mathbf{y}_{ZARU} = \boldsymbol{\omega}_m = \mathbf{b}_g$.
- Directly observes: 3D gyroscope bias $\mathbf{b}_g$.
- Gyroscope bias error $\delta\mathbf{b}_g$ covariance collapses to the sensor noise floor $\sigma_g^2$.

---

## 2. Fundamental Limitations: What Vehicle Physics CANNOT Observe

It is critical under the **No-False-Claims Policy** to state explicitly what physical constraints CANNOT do:

1. **Absolute Horizontal Position ($p_{\text{east}}, p_{\text{north}}$) is NEVER Observable**:
   Without an external absolute reference (such as GNSS, anchor stations, or visual landmarks), no internal physical constraint can determine whether the vehicle is in Bangalore, Delhi, or London. All dead reckoning inevitably exhibits position drift proportional to traveled distance or time.
2. **Absolute Global Yaw ($\psi$) during Straight Constant-Speed Motion**:
   While NHC and AI speed constrain the direction of the velocity vector relative to the vehicle heading ($v_y^b \approx 0$), if the entire trajectory slowly drifts in heading by $2^\circ$ during a straight highway cruise, the lateral velocity remains zero! Absolute heading during straight motion requires a magnetic compass, dual-antenna GNSS, or map-matching (Phase 7).
3. **Severe Wheel Skidding (Black Ice / Hydroplaning)**:
   If the vehicle physically slides sideways at high speed on ice, NHC will falsely attempt to force the lateral velocity to zero, corrupting the filter.

---

## 3. Simple Explanation
Think of a blindfolded person walking down a hallway with their hands touching both walls.
- Touching the walls (NHC) keeps them from bumping into the walls or veering sideways.
- Standing still and catching their breath (ZUPT/ZARU) lets them adjust their balance.
- BUT: touching the walls can never tell them what floor of the building they are on, or whether the hallway is facing East or North!

To know exactly which street or room they are in, they eventually need to open their eyes and look at a sign (Map Matching in Phase 7 or GPS in Phase 8).

---

## 4. Why This Matters for SIH26168
Understanding this boundary prevents unrealistic expectations: Phase 6 makes the trajectory physically consistent, eliminates stationary creep, and dramatically reduces heading drift rate, but absolute localization will be finalized when combined with OSM map matching in Phase 7.
