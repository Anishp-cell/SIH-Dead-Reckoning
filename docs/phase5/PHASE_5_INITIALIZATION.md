# Phase 5 Navigation State & Covariance Initialization Protocol

**Project**: Smart India Hackathon 2026 — Problem Statement SIH26168  
**Sponsoring Agency**: Indian Space Research Organisation (ISRO)  
**System**: AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation  
**Phase**: Phase 5 — 15-State Error-State Kalman Filter Navigation Core  
**Document Code**: `SIH26168-INIT-PHASE5-01`  
**Classification**: INITIALIZATION SPECIFICATION  

---

## 1. State Vector Initialization Architecture

When entering a GNSS outage or commencing a navigation session at epoch $t_0$, the 16-dimensional nominal state $\hat{\mathbf{x}}_0$ must be initialized from available physical cues without introducing unmodeled initial transient offsets.

```text
               Entry Accelerometer Vector [ax, ay, az]^T
                                   │
                                   ▼
                       Gravity Leveling Roll & Pitch
                                   │
    Entry GPS Heading ────────────┼────────────► Initial Quaternion q_0
                                   │
    Entry GPS Position ────────────┼────────────► Initial Position p_0
                                   │
    Entry Velocity / CAN Speed ────┴────────────► Initial Velocity v_0
```

---

## 2. Mathematical Initialization Equations

### 2.1 Position ($\mathbf{p}_0 \in \mathbb{R}^3$)
When benchmarking relative blackout dead reckoning from local origin:
$$\mathbf{p}_0 = \begin{bmatrix} 0.0 \\ 0.0 \\ 0.0 \end{bmatrix}\text{ meters}$$
When initialized at absolute geodetic entry point $(\phi_0, \lambda_0, h_0)$:
$$\mathbf{p}_0 = \text{geodetic\_to\_enu}(\phi_0, \lambda_0, h_0)$$

### 2.2 Velocity ($\mathbf{v}_0 \in \mathbb{R}^3$)
If stationary at initialization:
$$\mathbf{v}_0 = \begin{bmatrix} 0.0 \\ 0.0 \\ 0.0 \end{bmatrix}\text{ m/s}$$
If in motion at blackout entry, initial velocity in Navigation (ENU) frame is formed from vehicle speed $v_0$ and geographic heading $\psi_{\text{GPS}, 0}$:
$$\psi_{\text{ENU}, 0} = \frac{\pi}{2} - \psi_{\text{GPS}, 0}$$
$$\mathbf{v}_0 = \begin{bmatrix} v_0 \cos \psi_{\text{ENU}, 0} \\ v_0 \sin \psi_{\text{ENU}, 0} \\ 0.0 \end{bmatrix} = \begin{bmatrix} v_0 \sin \psi_{\text{GPS}, 0} \\ v_0 \cos \psi_{\text{GPS}, 0} \\ 0.0 \end{bmatrix}\text{ m/s}$$

### 2.3 Attitude Quaternion ($\mathbf{q}_0 \in \mathbb{H}$)
Attitude is split into gravity tilt (roll, pitch) and horizontal heading (yaw):
1. **Roll ($\phi_0$) and Pitch ($\theta_0$) Leveling**:
   Estimated from stationary / smoothed body acceleration $\mathbf{f}_b = [a_x, a_y, a_z]^T$:
   $$\phi_0 = \text{atan2}(-a_y, a_z)$$
   $$\theta_0 = \text{atan2}(a_x, \sqrt{a_y^2 + a_z^2})$$
2. **Yaw ($\psi_0$)**:
   $$\psi_{\text{ENU}, 0} = 90^\circ - \psi_{\text{GPS}, 0}$$
3. **Quaternion Synthesis**:
   $$\mathbf{q}_0 = \text{euler\_to\_quaternion}(\phi_0, \theta_0, \psi_{\text{ENU}, 0})$$

### 2.4 Sensor Biases ($\mathbf{b}_{a, 0}, \mathbf{b}_{g, 0}$)
Initialized to Phase 2 static calibration estimates or zero if pre-compensated:
$$\mathbf{b}_{a, 0} = \mathbf{0}_{3 \times 1}\text{ m/s}^2, \quad \mathbf{b}_{g, 0} = \mathbf{0}_{3 \times 1}\text{ rad/s}$$

---

## 3. Initial Covariance Matrix ($\mathbf{P}_0 \in \mathbb{R}^{15 \times 15}$)

The initial error covariance matrix $\mathbf{P}_0$ defines the filter's initial confidence. Setting $\mathbf{P}_0 = \mathbf{I}_{15}$ is mathematically incorrect because position, velocity, attitude, and biases have completely different units and scales:

$$\mathbf{P}_0 = \text{diag}\left( \sigma_{p0}^2 \mathbf{I}_3, \, \sigma_{v0}^2 \mathbf{I}_3, \, \sigma_{\theta0}^2, \sigma_{\theta0}^2, \sigma_{\psi0}^2, \, \sigma_{ba0}^2 \mathbf{I}_3, \, \sigma_{bg0}^2 \mathbf{I}_3 \right)$$

### Physically Justified Values:
- Position Uncertainty: $\sigma_{p0} = 1.0\text{ m} \implies \sigma_{p0}^2 = 1.0\text{ m}^2$ (standard GNSS fix accuracy).
- Velocity Uncertainty: $\sigma_{v0} = 0.5\text{ m/s} \implies \sigma_{v0}^2 = 0.25\text{ m}^2/\text{s}^2$.
- Roll/Pitch Tilt Uncertainty: $\sigma_{\theta0} = 2.0^\circ = 0.0349\text{ rad} \implies \sigma_{\theta0}^2 = 1.22 \times 10^{-3}\text{ rad}^2$.
- Yaw Heading Uncertainty: $\sigma_{\psi0} = 5.0^\circ = 0.0873\text{ rad} \implies \sigma_{\psi0}^2 = 7.62 \times 10^{-3}\text{ rad}^2$.
- Accelerometer Bias Uncertainty: $\sigma_{ba0} = 0.05\text{ m/s}^2 \implies \sigma_{ba0}^2 = 2.5 \times 10^{-3}\text{ m}^2/\text{s}^4$.
- Gyroscope Bias Uncertainty: $\sigma_{bg0} = 1.0 \times 10^{-3}\text{ rad/s} \implies \sigma_{bg0}^2 = 1.0 \times 10^{-6}\text{ rad}^2/\text{s}^2$.
