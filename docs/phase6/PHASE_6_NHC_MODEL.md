# Phase 6 Non-Holonomic Constraints (NHC) Specification

**Project**: Smart India Hackathon 2026 — Problem Statement SIH26168  
**Sponsoring Agency**: Indian Space Research Organisation (ISRO)  
**System**: AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation  
**Document**: `docs/phase6/PHASE_6_NHC_MODEL.md`  

---

## 1. Technical Derivation

### 1.1 Physical Basis of Non-Holonomic Constraints
For a land vehicle moving on a roadway, the non-holonomic constraint states that wheel rolling without sideslip prevents lateral movement, and road surface contact prevents vertical penetration or flight:
$$v_{\text{lat}} = v_y^b \approx 0$$
$$v_{\text{up}} = v_z^b \approx 0$$

These constraints act directly on the **vehicle body frame**, NOT in the global ENU frame. In the local ENU navigation frame, a vehicle traveling North at $10\text{ m/s}$ has $v_n = [0, 10, 0]^T$; a vehicle traveling East at $10\text{ m/s}$ has $v_n = [10, 0, 0]^T$. Transforming into the vehicle body frame:
$$\mathbf{v}_b = R_{nb}^T \mathbf{v}_n = \begin{bmatrix} v_x^b \\ v_y^b \\ v_z^b \end{bmatrix} = \begin{bmatrix} v_{\text{fwd}} \\ v_{\text{lat}} \\ v_{\text{up}} \end{bmatrix}$$
both scenarios produce $\mathbf{v}_b = [10, 0, 0]^T$.

### 1.2 Virtual Measurement Model
$$\mathbf{z}_{NHC} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$
$$\mathbf{h}_{NHC}(\mathbf{x}) = \begin{bmatrix} \mathbf{e}_2^T R_{nb}^T \mathbf{v}_n \\ \mathbf{e}_3^T R_{nb}^T \mathbf{v}_n \end{bmatrix} = \begin{bmatrix} v_y^b \\ v_z^b \end{bmatrix}$$
where $\mathbf{e}_2 = [0, 1, 0]^T$ and $\mathbf{e}_3 = [0, 0, 1]^T$.

Innovation residual:
$$\boldsymbol{\nu}_{NHC} = \mathbf{z}_{NHC} - \mathbf{h}_{NHC}(\hat{\mathbf{x}}) = -\begin{bmatrix} \hat{v}_y^b \\ \hat{v}_z^b \end{bmatrix}$$

### 1.3 Analytical Measurement Jacobian $H_{NHC}$
$$\mathbf{H}_{NHC} = \frac{\partial \mathbf{h}_{NHC}}{\partial \delta\mathbf{x}} = \begin{bmatrix} \mathbf{0}_{2\times 3} & \mathbf{H}_v & \mathbf{H}_\theta & \mathbf{0}_{2\times 3} & \mathbf{0}_{2\times 3} \end{bmatrix} \in \mathbb{R}^{2\times 15}$$

1. **Velocity sensitivity block**:
   $$\mathbf{H}_v = \begin{bmatrix} \mathbf{e}_2^T R_{nb}^T \\ \mathbf{e}_3^T R_{nb}^T \end{bmatrix} = \begin{bmatrix} (\text{Col } 1 \text{ of } R_{nb})^T \\ (\text{Col } 2 \text{ of } R_{nb})^T \end{bmatrix} \in \mathbb{R}^{2\times 3}$$
2. **Attitude sensitivity block**:
   $$\mathbf{H}_\theta = \begin{bmatrix} \mathbf{e}_2^T [\hat{\mathbf{v}}_b]_\times \\ \mathbf{e}_3^T [\hat{\mathbf{v}}_b]_\times \end{bmatrix} = \begin{bmatrix} v_{\text{up}} & 0 & -v_{\text{fwd}} \\ -v_{\text{lat}} & v_{\text{fwd}} & 0 \end{bmatrix} \in \mathbb{R}^{2\times 3}$$

Notice the fundamental coupling: $\frac{\partial v_{\text{lat}}}{\partial \delta\theta_z} = -v_{\text{fwd}}$. When the vehicle has forward speed $v_{\text{fwd}} > 0$, any orientation error around the yaw axis $\delta\theta_z$ manifests directly as an apparent lateral velocity. Enforcing $v_{\text{lat}} \approx 0$ provides direct observability on heading error during motion.

---

## 2. Simple Explanation
Think of a train on train tracks. The train can move forward or backward smoothly, but the steel rails prevent it from sliding sideways off the track or jumping into the air. 

While a car doesn't have steel tracks, its rubber tires act like invisible tracks on the road. If our smartphone calculations start guessing that the car is moving sideways at a $45^\circ$ angle down the street, NHC acts like the tracks, pulling the car back into straight-line alignment.

---

## 3. Why This Matters for SIH26168
In Phase 5, unconstrained lateral velocity integrated quadratic position drift. By constraining lateral and vertical velocity to near-zero, NHC stabilizes the trajectory and prevents sideways wandering during outages.

---

## 4. Assumptions & Limitations
- **Assumption**: Vehicle is driving on normal road pavement without prolonged black ice, hydroplaning, or drifting.
- **Limitation**: During aggressive high-speed cornering, tire compliance produces small slip angles ($\alpha \approx 2^\circ - 5^\circ$). To prevent NHC from fighting legitimate turns, the measurement covariance $\mathbf{R}_{NHC}$ must be adaptively expanded during high lateral acceleration.
