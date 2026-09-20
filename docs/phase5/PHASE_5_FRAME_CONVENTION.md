# Phase 5 Coordinate Frames & Rotation Conventions

**Project**: Smart India Hackathon 2026 — Problem Statement SIH26168  
**Sponsoring Agency**: Indian Space Research Organisation (ISRO)  
**System**: AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation  
**Phase**: Phase 5 — 15-State Error-State Kalman Filter Navigation Core  
**Document Code**: `SIH26168-FRAME-PHASE5-01`  
**Classification**: RIGOROUS COORDINATE CONVENTION SPECIFICATION  

---

## 1. Primary Coordinate Frames

In inertial navigation, mixing coordinate frame definitions causes catastrophic navigation divergence. Phase 5 strictly formalizes two reference frames:

```text
       Navigation Frame (n-frame)              Vehicle Body Frame (b-frame)
                  +Z (Up)                                  +Z (Up)
                     ▲                                        ▲
                     │                                        │
                     │                                        │
                     └────────► +Y (North)                    └────────► +X (Forward)
                    /                                        /
                   /                                        /
                  ▼                                        ▼
               +X (East)                                +Y (Lateral Right)
```

---

### 1.1 Navigation Frame ($n$-frame): Local East-North-Up (ENU)
- **Origin**: Geodetic point of initial GNSS fix $(\phi_0, \lambda_0, h_0) = (52.401660^\circ, -1.505290^\circ, 147.5\text{ m})$.
- **Axes**:
  - $+x_n$: Local **East** (orthogonal to meridian plane, tangent to parallel).
  - $+y_n$: Local **North** (along true meridian, tangent to ellipsoid pointing toward North Pole).
  - $+z_n$: Local **Up** (normal to WGS84 reference ellipsoid pointing upward).
- **Gravity Direction**:
  $$\mathbf{g}_n = \begin{bmatrix} 0 \\ 0 \\ -g_0 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ -9.80665 \end{bmatrix}\text{ m/s}^2$$

---

### 1.2 Vehicle Body Frame ($b$-frame): ISO 8855 Adapted
- **Origin**: Center of mass of vehicle / smartphone rigidly mounted in cradle.
- **Axes**:
  - $+x_b$: **Forward** longitudinal driving direction along vehicle centerline.
  - $+y_b$: **Lateral Right** toward passenger door (forming right-handed system).
  - $+z_b$: **Vertical Up** orthogonal to road bed, pointing out through car roof.
- **IMU Specific Force Vector in Body Frame**:
  $$\mathbf{f}_b = \begin{bmatrix} a_{\text{fwd}} \\ a_{\text{lat}} \\ a_{\text{up}} \end{bmatrix}$$
- **IMU Angular Velocity Vector in Body Frame**:
  $$\boldsymbol{\omega}_{ib}^b = \begin{bmatrix} \omega_{\text{roll}} \\ \omega_{\text{pitch}} \\ \omega_{\text{yaw}} \end{bmatrix}$$

---

## 2. Direction Cosine Matrix (DCM) Notation

1. $\mathbf{R}_{nb} \in \text{SO}(3)$:
   - Definition: Transforms a vector expressed in the Body frame ($b$) to the Navigation frame ($n$).
   - Formula:
     $$\mathbf{v}_n = \mathbf{R}_{nb} \mathbf{v}_b$$
   - Columns of $\mathbf{R}_{nb}$: The unit axes of the body frame expressed in navigation coordinates:
     $$\mathbf{R}_{nb} = \begin{bmatrix} \mathbf{x}_{b}^n & \mathbf{y}_{b}^n & \mathbf{z}_{b}^n \end{bmatrix}$$

2. $\mathbf{R}_{bn} \in \text{SO}(3)$:
   - Definition: Transforms a vector expressed in Navigation frame ($n$) to Body frame ($b$).
   - Orthonormal Property:
     $$\mathbf{R}_{bn} = \mathbf{R}_{nb}^{-1} = \mathbf{R}_{nb}^T$$
     $$\mathbf{v}_b = \mathbf{R}_{bn} \mathbf{v}_n = \mathbf{R}_{nb}^T \mathbf{v}_n$$

---

## 3. Quaternion Mathematics & Conventions

Phase 5 adopts the standard scalar-first unit quaternion convention:

$$\mathbf{q} = \begin{bmatrix} q_w \\ \mathbf{q}_v \end{bmatrix} = \begin{bmatrix} q_w \\ q_x \\ q_y \\ q_z \end{bmatrix} \in \mathbb{H}, \quad \|\mathbf{q}\| = \sqrt{q_w^2 + q_x^2 + q_y^2 + q_z^2} = 1$$

### 3.1 Hamilton Product ($\otimes$)
For two quaternions $\mathbf{p} = [p_w, \mathbf{p}_v]^T$ and $\mathbf{q} = [q_w, \mathbf{q}_v]^T$:
$$\mathbf{p} \otimes \mathbf{q} = \begin{bmatrix} p_w q_w - \mathbf{p}_v \cdot \mathbf{q}_v \\ p_w \mathbf{q}_v + q_w \mathbf{p}_v + \mathbf{p}_v \times \mathbf{q}_v \end{bmatrix}$$

In component form:
$$\mathbf{p} \otimes \mathbf{q} = \begin{bmatrix}
p_w q_w - p_x q_x - p_y q_y - p_z q_z \\
p_w q_x + p_x q_w + p_y q_z - p_z q_y \\
p_w q_y - p_x q_z + p_y q_w + p_z q_x \\
p_w q_z + p_x q_y - p_y q_x + p_z q_w
\end{bmatrix}$$

### 3.2 Vector Rotation via Quaternion
A vector $\mathbf{v}_b$ in body frame is rotated to navigation frame by:
$$\begin{bmatrix} 0 \\ \mathbf{v}_n \end{bmatrix} = \mathbf{q} \otimes \begin{bmatrix} 0 \\ \mathbf{v}_b \end{bmatrix} \otimes \mathbf{q}^*$$
where $\mathbf{q}^* = [q_w, -\mathbf{q}_v]^T$ is the quaternion conjugate.

### 3.3 Quaternion to Direction Cosine Matrix
$$\mathbf{R}(\mathbf{q}) = \mathbf{R}_{nb} = \begin{bmatrix}
1 - 2(q_y^2 + q_z^2) & 2(q_x q_y - q_w q_z) & 2(q_x q_z + q_w q_y) \\
2(q_x q_y + q_w q_z) & 1 - 2(q_x^2 + q_z^2) & 2(q_y q_z - q_w q_x) \\
2(q_x q_z - q_w q_y) & 2(q_y q_z + q_w q_x) & 1 - 2(q_x^2 + q_y^2)
\end{bmatrix}$$

---

## 4. Euler Angle Mapping & Geographic Azimuth Reconciliation

### 4.1 Euler Angles (ZYX Yaw-Pitch-Roll)
Let $\psi$ (yaw), $\theta$ (pitch), and $\phi$ (roll) be rotations about $z$, $y$, and $x$ respectively:
$$\mathbf{R}_{nb} = \mathbf{R}_z(\psi) \mathbf{R}_y(\theta) \mathbf{R}_x(\phi)$$

### 4.2 Reconciliation of Cartesian ENU Yaw vs Geographic Azimuth
- **Geographic GPS Heading $\psi_{\text{GPS}}$**:
  Measured clockwise from true North:
  $$\text{North} = 0^\circ, \quad \text{East} = 90^\circ, \quad \text{South} = 180^\circ, \quad \text{West} = 270^\circ$$
- **Cartesian ENU Yaw $\psi_{\text{ENU}}$**:
  Measured counter-clockwise from East:
  $$\text{East} = 0^\circ, \quad \text{North} = +90^\circ, \quad \text{West} = 180^\circ, \quad \text{South} = -90^\circ$$

### Rigorous Conversion Identity
When the vehicle has geographic azimuth $\psi_{\text{GPS}}$, its unit direction in East-North coordinates is:
$$\mathbf{u}_{\text{veh}} = \begin{bmatrix} \sin \psi_{\text{GPS}} \\ \cos \psi_{\text{GPS}} \\ 0 \end{bmatrix}$$

Under Cartesian rotation by $\psi_{\text{ENU}}$, the body forward vector $[1, 0, 0]^T$ transforms to:
$$\mathbf{R}_{nb} \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix} = \begin{bmatrix} \cos \psi_{\text{ENU}} \\ \sin \psi_{\text{ENU}} \\ 0 \end{bmatrix}$$

Equating components:
$$\cos \psi_{\text{ENU}} = \sin \psi_{\text{GPS}}$$
$$\sin \psi_{\text{ENU}} = \cos \psi_{\text{GPS}}$$

Therefore:
$$\boxed{\psi_{\text{ENU}} = 90^\circ - \psi_{\text{GPS}} = \frac{\pi}{2} - \psi_{\text{GPS}}}$$
$$\boxed{\psi_{\text{GPS}} = 90^\circ - \psi_{\text{ENU}} = \frac{\pi}{2} - \psi_{\text{ENU}}}$$

---

## 5. Small-Angle Error Representation & Quaternion Injection

In the Error-State Kalman Filter, orientation error is parameterized as a 3-dimensional vector $\delta \boldsymbol{\theta} \in \mathbb{R}^3$ (minimal parameterization, avoiding 4-state quaternion singularity).

### 5.1 True vs Nominal Attitude
The true attitude quaternion $\mathbf{q}$ is related to the nominal attitude $\hat{\mathbf{q}}$ and the small-angle error quaternion $\delta \mathbf{q}$ by:
$$\mathbf{q} = \hat{\mathbf{q}} \otimes \delta \mathbf{q}$$
where the error $\delta \mathbf{q}$ is defined in the **Body frame**:
$$\delta \mathbf{q} \approx \begin{bmatrix} 1 \\ \frac{1}{2} \delta \boldsymbol{\theta} \end{bmatrix} + \mathcal{O}(\|\delta \boldsymbol{\theta}\|^2)$$

### 5.2 Error State Injection & Reset
When the Kalman update produces an error correction $\delta \hat{\boldsymbol{\theta}}$, the nominal quaternion is updated by:
$$\hat{\mathbf{q}}^+ = \hat{\mathbf{q}}^- \otimes \begin{bmatrix} 1 \\ \frac{1}{2} \delta \hat{\boldsymbol{\theta}} \end{bmatrix}$$
followed immediately by normalization:
$$\hat{\mathbf{q}}^+ \leftarrow \frac{\hat{\mathbf{q}}^+}{\|\hat{\mathbf{q}}^+\|}$$
The error state is then reset to zero:
$$\delta \hat{\mathbf{x}} \leftarrow \mathbf{0}_{15 \times 1}$$
