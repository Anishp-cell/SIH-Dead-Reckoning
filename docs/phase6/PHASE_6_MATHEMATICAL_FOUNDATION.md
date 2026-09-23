# Phase 6 Mathematical Foundations: Vehicle Physics Constraints, ZUPT/ZARU, and Disturbance-Aware ESKF

**Project**: Smart India Hackathon 2026 — Problem Statement SIH26168  
**Sponsoring Agency**: Indian Space Research Organisation (ISRO)  
**System**: AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation  
**Document**: `docs/phase6/PHASE_6_MATHEMATICAL_FOUNDATION.md`  

---

## 1. Introduction & Motivation

Phase 5 established the foundational 15-state Error-State Kalman Filter (ESKF) using calibrated smartphone IMU propagation and forward velocity updates from Phase 4 AI motion intelligence. While Phase 5 demonstrated that forward velocity updates reduce longitudinal divergence, it revealed three critical navigation failure modes:
1. **Lateral and Vertical Drift**: Forward velocity $v_x^b$ updates only constrain the longitudinal component. Lateral slip velocity ($v_y^b$) and vertical velocity ($v_z^b$) accumulate uncontrolled quadratic drift from integrated accelerometer biases.
2. **Stationary Position Creep**: During vehicle stops (e.g., at red lights or traffic jams), small residual sensor biases and velocity noise integrate into artificial positional creep (over $500\text{ m}$ of drift during a 50-second standstill).
3. **Unconstrained Gyroscope Yaw Drift**: Over long outages ($>60\text{ s}$), unobserved gyroscope bias drift causes heading errors ($>70^\circ$). Because the forward velocity vector is rotated by the estimated attitude, any heading error projects forward speed into cross-track position errors.

Phase 6 addresses these physical realities without relying on maps, GNSS, or learned heading models, by formalizing:
- **Non-Holonomic Constraints (NHC)**: Land vehicles typically roll without significant sideslip or vertical flight ($v_y^b \approx 0, v_z^b \approx 0$).
- **Zero-Velocity Updates (ZUPT)**: Physical standstill clamps velocity to zero ($\mathbf{v}_n \approx \mathbf{0}$) and isolates accelerometer biases.
- **Zero-Angular-Rate Updates (ZARU)**: Physical standstill isolates gyroscope biases ($\boldsymbol{\omega}_m \approx \mathbf{b}_g$), arresting yaw drift rate.
- **Disturbance-Aware Adaptive Covariance**: Dynamic weighting preventing virtual constraints from corrupting legitimate maneuvers during sharp turns or potholes.

---

## 2. Non-Holonomic Constraints (NHC)

### 2.1 Technical Derivation
A wheeled road vehicle travels primarily along its longitudinal axis. Under normal adhesion conditions without severe skidding or flight, the velocities perpendicular to the forward direction in the vehicle body frame are approximately zero:
$$v_{\text{lat}} = v_y^b \approx 0$$
$$v_{\text{up}} = v_z^b \approx 0$$

Let $\mathbf{v}_n = [v_{\text{east}}, v_{\text{north}}, v_{\text{up}}]^T$ be the velocity in the local ENU navigation frame. The velocity in the vehicle body frame is:
$$\mathbf{v}_b = R_{nb}^T \mathbf{v}_n = \begin{bmatrix} v_x^b \\ v_y^b \\ v_z^b \end{bmatrix} = \begin{bmatrix} v_{\text{fwd}} \\ v_{\text{lat}} \\ v_{\text{up}} \end{bmatrix}$$

The virtual NHC measurement vector is defined as:
$$\mathbf{z}_{NHC} = \begin{bmatrix} 0 \\ 0 \end{bmatrix} \in \mathbb{R}^2$$

The non-linear observation model $\mathbf{h}_{NHC}(\mathbf{x})$ extracts the lateral and vertical components:
$$\mathbf{h}_{NHC}(\mathbf{x}) = \begin{bmatrix} \mathbf{e}_2^T \\ \mathbf{e}_3^T \end{bmatrix} \mathbf{v}_b = \begin{bmatrix} \mathbf{e}_2^T R_{nb}^T \mathbf{v}_n \\ \mathbf{e}_3^T R_{nb}^T \mathbf{v}_n \end{bmatrix} \in \mathbb{R}^2$$
where $\mathbf{e}_2 = [0, 1, 0]^T$ and $\mathbf{e}_3 = [0, 0, 1]^T$.

The innovation residual $\boldsymbol{\nu}_{NHC}$ is:
$$\boldsymbol{\nu}_{NHC} = \mathbf{z}_{NHC} - \mathbf{h}_{NHC}(\hat{\mathbf{x}}) = \begin{bmatrix} 0 - \hat{v}_y^b \\ 0 - \hat{v}_z^b \end{bmatrix} = -\begin{bmatrix} \hat{v}_{\text{lat}} \\ \hat{v}_{\text{up}} \end{bmatrix}$$

#### Analytical Measurement Jacobian $H_{NHC}$
We linearize $\mathbf{h}_{NHC}(\mathbf{x})$ with respect to the 15-state indirect error state:
$$\delta\mathbf{x} = \begin{bmatrix} \delta\mathbf{p}_n^T & \delta\mathbf{v}_n^T & \delta\boldsymbol{\theta}^T & \delta\mathbf{b}_a^T & \delta\mathbf{b}_g^T \end{bmatrix}^T \in \mathbb{R}^{15}$$

Recalling the right-multiplicative body attitude error convention:
$$R_{nb}(\mathbf{q}) = R_{nb}(\hat{\mathbf{q}}) R(\delta\mathbf{q}) \approx R_{nb} (I_3 + [\delta\boldsymbol{\theta}]_\times)$$
$$R_{bn}(\mathbf{q}) = R_{nb}^T = (I_3 - [\delta\boldsymbol{\theta}]_\times) R_{nb}^T$$

Perturbing velocity $\mathbf{v}_n = \hat{\mathbf{v}}_n + \delta\mathbf{v}_n$:
$$\mathbf{v}_b = (I_3 - [\delta\boldsymbol{\theta}]_\times) R_{nb}^T (\hat{\mathbf{v}}_n + \delta\mathbf{v}_n) = \hat{\mathbf{v}}_b + R_{nb}^T \delta\mathbf{v}_n - [\delta\boldsymbol{\theta}]_\times \hat{\mathbf{v}}_b + \mathcal{O}(\|\delta\mathbf{x}\|^2)$$

Using the skew-symmetric identity $-[\delta\boldsymbol{\theta}]_\times \hat{\mathbf{v}}_b = \hat{\mathbf{v}}_b \times \delta\boldsymbol{\theta} = [\hat{\mathbf{v}}_b]_\times \delta\boldsymbol{\theta}$:
$$\mathbf{v}_b \approx \hat{\mathbf{v}}_b + R_{nb}^T \delta\mathbf{v}_n + [\hat{\mathbf{v}}_b]_\times \delta\boldsymbol{\theta}$$

Applying selection matrices $\begin{bmatrix} \mathbf{e}_2^T \\ \mathbf{e}_3^T \end{bmatrix}$:
$$\mathbf{H}_{NHC} = \frac{\partial \mathbf{h}_{NHC}}{\partial \delta\mathbf{x}} = \begin{bmatrix} \mathbf{0}_{2\times 3} & \mathbf{H}_v & \mathbf{H}_\theta & \mathbf{0}_{2\times 3} & \mathbf{0}_{2\times 3} \end{bmatrix} \in \mathbb{R}^{2\times 15}$$

1. **Velocity Jacobian Block $\mathbf{H}_v \in \mathbb{R}^{2\times 3}$**:
   $$\mathbf{H}_v = \begin{bmatrix} \mathbf{e}_2^T R_{nb}^T \\ \mathbf{e}_3^T R_{nb}^T \end{bmatrix} = \begin{bmatrix} (\text{Col } 1 \text{ of } R_{nb})^T \\ (\text{Col } 2 \text{ of } R_{nb})^T \end{bmatrix}$$
2. **Attitude Jacobian Block $\mathbf{H}_\theta \in \mathbb{R}^{2\times 3}$**:
   Since:
   $$[\hat{\mathbf{v}}_b]_\times = \begin{bmatrix} 0 & -v_z^b & v_y^b \\ v_z^b & 0 & -v_x^b \\ -v_y^b & v_x^b & 0 \end{bmatrix} = \begin{bmatrix} 0 & -v_{\text{up}} & v_{\text{lat}} \\ v_{\text{up}} & 0 & -v_{\text{fwd}} \\ -v_{\text{lat}} & v_{\text{fwd}} & 0 \end{bmatrix}$$
   We compute:
   $$\mathbf{e}_2^T [\hat{\mathbf{v}}_b]_\times = \begin{bmatrix} v_{\text{up}} & 0 & -v_{\text{fwd}} \end{bmatrix}$$
   $$\mathbf{e}_3^T [\hat{\mathbf{v}}_b]_\times = \begin{bmatrix} -v_{\text{lat}} & v_{\text{fwd}} & 0 \end{bmatrix}$$
   Therefore:
   $$\mathbf{H}_\theta = \begin{bmatrix} v_{\text{up}} & 0 & -v_{\text{fwd}} \\ -v_{\text{lat}} & v_{\text{fwd}} & 0 \end{bmatrix} \in \mathbb{R}^{2\times 3}$$

### 2.2 Simple Explanation
Imagine driving a car down the road. Cars have rubber tires that roll forward, but they cannot move sideways through solid asphalt like a hovercraft, nor can they float upwards into the sky. 

If our smartphone math starts claiming that the car is sliding sideways at $15\text{ km/h}$ or flying up $3\text{ meters}$, we know that is physically impossible. Non-Holonomic Constraints act as a mathematical track: every fraction of a second, the filter checks "Is the car drifting sideways or up?" and gently nudges those velocities back to zero.

Crucially, because the side-wheels only stay locked if the car points in the direction it is traveling, whenever the car drives forward, measuring zero sideways velocity immediately tells the filter if its compass heading was pointing the wrong way!

### 2.3 Why This Matters for Our Project
Forward AI speed alone only controls speed in the direction the phone *thinks* it is pointing. If heading slips by $10^\circ$, a pure AI speed model keeps barreling forward along that wrong angle. NHC couples lateral velocity back into attitude $\delta\boldsymbol{\theta}$, providing the essential missing observability link that stabilizes vehicle heading during motion.

---

## 3. Disturbance-Aware Adaptive Covariance

### 3.1 Technical Derivation
Real vehicles do experience legitimate non-zero lateral velocity during maneuvers (centripetal tire slip angle $\alpha$) and vertical velocity over bumps:
$$v_{\text{lat}} \approx -v_{\text{fwd}} \left(\frac{m v_{\text{fwd}}}{C_{\alpha} L}\right) \dot{\psi}$$

If NHC is enforced with an infinitely rigid covariance ($\mathbf{R}_{NHC} \to \mathbf{0}$), the filter would treat legitimate cornering slip as an error, corrupting attitude estimates.

We formulate an adaptive measurement noise covariance:
$$\mathbf{R}_{NHC} = \begin{bmatrix} \sigma_{\text{lat}}^2(t) & 0 \\ 0 & \sigma_{\text{up}}^2(t) \end{bmatrix}$$
where the standard deviations are dynamically modulated by a causal disturbance index $D(t) \in [0, \infty)$:
$$\sigma_{\text{lat}}(t) = \sigma_{\text{lat}, 0} \cdot \left(1 + \kappa_{\text{dyn}} D_{\text{lat}}(t)\right)$$
$$\sigma_{\text{up}}(t) = \sigma_{\text{up}, 0} \cdot \left(1 + \kappa_{\text{dyn}} D_{\text{up}}(t)\right)$$

The causal disturbance metrics are computed from streaming IMU quantities:
1. **Lateral Maneuver Disturbance**:
   $$D_{\text{lat}}(t) = \frac{|a_{\text{lat}}(t)|}{a_{\text{lat, max}}} + \frac{|\omega_{\text{yaw}}(t)|}{\omega_{\text{yaw, max}}}$$
2. **Vertical Shock / Pothole Disturbance**:
   $$D_{\text{up}}(t) = \frac{|a_{\text{up}}(t) - g|}{g_{\text{thresh}}} + \frac{\text{VibrationEnergy}(t)}{E_{\text{thresh}}}$$

When $D(t)$ spikes, $\mathbf{R}_{NHC}$ expands, gracefully down-weighting the constraint so the filter does not fight physical maneuvers.

### 3.2 Simple Explanation
If a car turns sharply around a street corner, the tires squeak and slide a tiny bit sideways. If the car hits a speed bump, it bounces up for a split second.

If our filter rigidly shouted "You can NEVER move sideways or up, even for a millisecond!", it would panic during turns and bumps. Instead, our disturbance detector detects when the car is cornering or hitting a bump and says: "Relax the rule for a moment—the car is just turning!" When the car goes straight again, it tightens the rule back up.

### 3.3 Why This Matters for Our Project
Fixed NHC causes filter instability and NIS rejections during active urban driving. Adaptive covariance guarantees numerical stability and prevents divergence across potholes and aggressive steering maneuvers.

---

## 4. Zero-Velocity Update (ZUPT)

### 4.1 Technical Derivation
When the vehicle is completely stopped at an intersection or in traffic:
$$\mathbf{v}_{\text{true}}^n = \mathbf{0}_{3\times 1}$$

The ZUPT virtual measurement is:
$$\mathbf{z}_{ZUPT} = \mathbf{0}_{3\times 1} \in \mathbb{R}^3$$
$$\mathbf{h}_{ZUPT}(\mathbf{x}) = \mathbf{v}_n$$
$$\boldsymbol{\nu}_{ZUPT} = \mathbf{0} - \hat{\mathbf{v}}_n = -\hat{\mathbf{v}}_n$$

The measurement Jacobian with respect to $\delta\mathbf{x}$ is constant and exact:
$$\mathbf{H}_{ZUPT} = \begin{bmatrix} \mathbf{0}_{3\times 3} & I_3 & \mathbf{0}_{3\times 3} & \mathbf{0}_{3\times 3} & \mathbf{0}_{3\times 3} \end{bmatrix} \in \mathbb{R}^{3\times 15}$$

Covariance:
$$\mathbf{R}_{ZUPT} = \sigma_{\text{zupt}}^2 I_3, \quad \sigma_{\text{zupt}} = 0.05\text{ m/s}$$

#### Observability Impact of ZUPT
During stationary periods, the state transition equation for velocity is:
$$\dot{\mathbf{v}}_n = R_{nb} (\mathbf{f}_b - \mathbf{b}_a) + \mathbf{g}_n$$
When stationary, $\mathbf{v}_n = \mathbf{0}$ and $\dot{\mathbf{v}}_n = \mathbf{0}$. Therefore:
$$\mathbf{0} = R_{nb} (\mathbf{f}_b - \mathbf{b}_a) + \mathbf{g}_n \implies R_{nb} \mathbf{b}_a = R_{nb} \mathbf{f}_b + \mathbf{g}_n$$
By locking $\mathbf{v}_n = \mathbf{0}$, the accelerometer bias $\mathbf{b}_a$ becomes **directly observable** from the gravity leveling residual!

### 4.2 Simple Explanation
When a car stops at a red light, its true speed is zero. 

Without ZUPT, an IMU keeps integrating tiny sensor imperfections, causing the vehicle's dot on the map to creep across the street into a building while sitting at the light. ZUPT tells the filter: "We are at a dead stop. Set velocity uncertainty to near-zero, stop moving on the map, and use this pause to recalibrate the accelerometer!"

### 4.3 Why This Matters for Our Project
In Phase 5, sequence S1 had a 50-second standstill ($t=4618$ to $4668\text{ s}$) where the filter drifted over $500\text{ m}$ while standing still! ZUPT eliminates standstill creep completely.

---

## 5. Zero-Angular-Rate Update (ZARU)

### 5.1 Technical Derivation
When the vehicle is stationary, its true angular velocity is zero (neglecting Earth rotation rate $\omega_e \approx 7.29 \times 10^{-5}\text{ rad/s} \approx 0.004^\circ/\text{s}$, which is two orders of magnitude below smartphone consumer MEMS noise):
$$\boldsymbol{\omega}_{\text{true}}^b \approx \mathbf{0}_{3\times 1}$$

The calibrated gyroscope measurement is:
$$\boldsymbol{\omega}_m = \boldsymbol{\omega}_{\text{true}}^b + \mathbf{b}_g + \mathbf{n}_g \approx \mathbf{b}_g + \mathbf{n}_g$$

We formulate ZARU as a direct observation of gyroscope bias:
$$\mathbf{z}_{ZARU} = \boldsymbol{\omega}_m \in \mathbb{R}^3$$
$$\mathbf{h}_{ZARU}(\mathbf{x}) = \hat{\mathbf{b}}_g \in \mathbb{R}^3$$
$$\boldsymbol{\nu}_{ZARU} = \boldsymbol{\omega}_m - \hat{\mathbf{b}}_g$$

The measurement Jacobian with respect to $\delta\mathbf{x}$ is:
$$\mathbf{H}_{ZARU} = \begin{bmatrix} \mathbf{0}_{3\times 3} & \mathbf{0}_{3\times 3} & \mathbf{0}_{3\times 3} & \mathbf{0}_{3\times 3} & I_3 \end{bmatrix} \in \mathbb{R}^{3\times 15}$$

Covariance:
$$\mathbf{R}_{ZARU} = \sigma_{\text{zaru}}^2 I_3, \quad \sigma_{\text{zaru}} = \sigma_g = 0.005\text{ rad/s}$$

> **CRITICAL RULE**: ZARU does **NOT** artificially force heading ($\psi$) or small-angle attitude ($\delta\boldsymbol{\theta}$) to zero. It acts strictly on the gyroscope bias state $\delta\mathbf{b}_g$. This prevents corrupting the vehicle's true orientation while resting on an inclined street or parked at an arbitrary heading.

### 5.2 Simple Explanation
If the car is stopped at a traffic light and not spinning like a ballerina on ice, any spinning measured by the gyroscope must be an internal flaw (gyro bias).

ZARU catches this flaw red-handed: it measures the leftover spin and updates the gyro bias estimate so that when the car drives off again, the compass heading stays steady instead of drifting wildly.

### 5.3 Why This Matters for Our Project
Gyro bias drift is the #1 killer of dead reckoning over 60–120s blackouts. By recalibrating gyro bias at every red light, ZARU prevents long-term heading divergence.

---

## 6. Multi-Signal Stationary Detection with Hysteresis

To prevent false ZUPT/ZARU triggering during slow stop-and-go crawling, we deploy a multi-signal decision gate fusing Phase 3 IMU statistics with Phase 4 AI motion predictions:

$$\text{Stationary Candidate} = (\text{Var}(\mathbf{f}_b) < \tau_{\text{var}}) \land (\|\boldsymbol{\omega}_b\| < \tau_{\omega}) \land (|\|\mathbf{f}_b\| - g| < \tau_g) \land (\hat{v}_{\text{fwd}} < \tau_v)$$

### Hysteresis State Machine
To eliminate high-frequency chatter:
- **Transition to Stationary**: Must satisfy stationary candidate continuously for $N_{\text{enter}} \ge 5$ samples ($0.5\text{ s}$).
- **Transition to Moving**: If any moving condition triggers for $N_{\text{exit}} \ge 2$ samples ($0.2\text{ s}$), immediately exit stationary mode.

---

## 7. Generalized Joseph-Form Vector Kalman Update

For an arbitrary $m$-dimensional measurement with innovation $\boldsymbol{\nu} \in \mathbb{R}^m$, Jacobian $H \in \mathbb{R}^{m\times 15}$, and noise $R \in \mathbb{R}^{m\times m}$:

1. **Innovation Covariance**:
   $$S = H P^- H^T + R \in \mathbb{R}^{m\times m}$$
2. **Normalized Innovation Squared (NIS)**:
   $$\text{NIS} = \boldsymbol{\nu}^T S^{-1} \boldsymbol{\nu}$$
   *Gating Rule*: Accept if $\text{NIS} \le \chi^2_{\alpha}(m)$ (e.g., $\chi^2_{0.99}(2) = 9.21$ for NHC; $\chi^2_{0.99}(3) = 11.34$ for ZUPT).
3. **Kalman Gain**:
   $$K = P^- H^T S^{-1} \in \mathbb{R}^{15\times m}$$
4. **Numerically Stable Joseph Covariance Update**:
   $$P^+ = (I_{15} - KH) P^- (I_{15} - KH)^T + K R K^T$$
   $$P^+ \leftarrow \frac{1}{2}\left(P^+ + (P^+)^T\right)$$
5. **State Error Injection & Reset**:
   $$\hat{\mathbf{p}}^+ = \hat{\mathbf{p}}^- + \delta\hat{\mathbf{p}}, \quad \hat{\mathbf{v}}^+ = \hat{\mathbf{v}}^- + \delta\hat{\mathbf{v}}$$
   $$\hat{\mathbf{q}}^+ = \hat{\mathbf{q}}^- \otimes \begin{bmatrix} 1 \\ \frac{1}{2}\delta\hat{\boldsymbol{\theta}} \end{bmatrix}, \quad \hat{\mathbf{q}}^+ \leftarrow \frac{\hat{\mathbf{q}}^+}{\|\hat{\mathbf{q}}^+\|}$$
   $$\hat{\mathbf{b}}_a^+ = \hat{\mathbf{b}}_a^- + \delta\hat{\mathbf{b}}_a, \quad \hat{\mathbf{b}}_g^+ = \hat{\mathbf{b}}_g^- + \delta\hat{\mathbf{b}}_g$$
   $$\delta\hat{\mathbf{x}} \leftarrow \mathbf{0}_{15\times 1}$$

---

## 8. Summary of Measurement Specifications

| Measurement | Dimension $m$ | Observation $h(\mathbf{x})$ | Jacobian $H$ Non-Zero Blocks | Nominal Covariance $R$ | NIS Gate $\chi^2_{0.99}(m)$ |
|:------------|:-------------:|:----------------------------|:-----------------------------|:-----------------------|:---------------------------:|
| **AI Speed** | 1 | $v_x^b = \mathbf{e}_1^T R_{nb}^T \mathbf{v}_n$ | $\mathbf{H}_v = \mathbf{e}_1^T R_{nb}^T, \mathbf{H}_\theta = [0, -v_z, v_y]$ | Dynamic Phase 4 $\sigma_v^2$ | $6.63$ (or 9.0) |
| **NHC** | 2 | $[v_y^b, v_z^b]^T = [\mathbf{e}_2, \mathbf{e}_3]^T R_{nb}^T \mathbf{v}_n$ | $\mathbf{H}_v = [\mathbf{e}_2, \mathbf{e}_3]^T R_{nb}^T, \mathbf{H}_\theta = \begin{bmatrix} v_z & 0 & -v_x \\ -v_y & v_x & 0 \end{bmatrix}$ | Adaptive $\text{diag}(\sigma_{\text{lat}}^2, \sigma_{\text{up}}^2)$ | $9.21$ |
| **ZUPT** | 3 | $\mathbf{v}_n$ | $\mathbf{H}_v = I_3$ | $\sigma_{\text{zupt}}^2 I_3$ ($0.05^2$) | $11.34$ |
| **ZARU** | 3 | $\mathbf{b}_g$ | $\mathbf{H}_{bg} = I_3$ | $\sigma_{\text{zaru}}^2 I_3$ ($0.005^2$) | $11.34$ |
