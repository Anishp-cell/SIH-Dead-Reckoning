# Phase 5 Mathematical Foundation: 15-State Error-State Kalman Filter (ESKF)

**Project**: Smart India Hackathon 2026 — Problem Statement SIH26168  
**Sponsoring Agency**: Indian Space Research Organisation (ISRO)  
**System**: AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation  
**Phase**: Phase 5 — 15-State Error-State Kalman Filter Navigation Core  
**Document Code**: `SIH26168-MATH-PHASE5-01`  
**Classification**: RIGOROUS MATHEMATICAL DERIVATION & FOUNDATION  

---

## Table of Contents
1. [State Definition & Error-State Philosophy](#1-state-definition--error-state-philosophy)
2. [IMU Sensor Measurement Model](#2-imu-sensor-measurement-model)
3. [Continuous-Time Nominal Navigation Kinematics](#3-continuous-time-nominal-navigation-kinematics)
4. [Continuous-Time Error-State Dynamics](#4-continuous-time-error-state-dynamics)
5. [Discrete-Time System Propagation (F and G Matrices)](#5-discrete-time-system-propagation-f-and-g-matrices)
6. [Process Noise Formulation (Q Matrix)](#6-process-noise-formulation-q-matrix)
7. [Covariance Propagation & Numerical Stabilization](#7-covariance-propagation--numerical-stabilization)
8. [Phase 4 AI Forward-Speed Measurement Model](#8-phase-4-ai-forward-speed-measurement-model)
9. [Measurement Jacobian Matrix (H Matrix)](#9-measurement-jacobian-matrix-h-matrix)
10. [Observability & Information Analysis](#10-observability--information-analysis)
11. [Kalman Correction & Joseph Form Covariance Update](#11-kalman-correction--joseph-form-covariance-update)
12. [Normalized Innovation Squared (NIS) Gating](#12-normalized-innovation-squared-nis-gating)
13. [Error State Injection & Attitude Reset](#13-error-state-injection--attitude-reset)

---

## 1. State Definition & Error-State Philosophy

### Technical Derivation
In the Error-State Kalman Filter (ESKF, or indirect Kalman filter), the navigation system maintains two distinct state representations:
1. **Nominal State ($\mathbf{x} \in \mathbb{R}^{16}$)**: High-rate, non-linear physical state propagated directly from raw high-frequency IMU measurements.
2. **Error State ($\delta \mathbf{x} \in \mathbb{R}^{15}$)**: Minimal, linearized small-perturbation state representing navigation errors and sensor biases, estimated by the Kalman filter.

The true state $\mathbf{x}_{\text{true}}$ is decomposed into nominal and error components:
$$\mathbf{x}_{\text{true}} = \mathbf{x} \oplus \delta \mathbf{x}$$

#### Nominal State Vector:
$$\mathbf{x} = \begin{bmatrix} \mathbf{p}_n \\ \mathbf{v}_n \\ \mathbf{q} \\ \mathbf{b}_a \\ \mathbf{b}_g \end{bmatrix} \in \mathbb{R}^{16}$$
where:
- $\mathbf{p}_n = [p_x, p_y, p_z]^T \in \mathbb{R}^3$: Position in Navigation (ENU) frame (meters).
- $\mathbf{v}_n = [v_x, v_y, v_z]^T \in \mathbb{R}^3$: Velocity in Navigation (ENU) frame (m/s).
- $\mathbf{q} = [q_w, q_x, q_y, q_z]^T \in \mathbb{H}, \|\mathbf{q}\|=1$: Unit attitude quaternion (Vehicle Body to Navigation frame).
- $\mathbf{b}_a = [b_{ax}, b_{ay}, b_{az}]^T \in \mathbb{R}^3$: Accelerometer bias in Body frame ($\text{m/s}^2$).
- $\mathbf{b}_g = [b_{gx}, b_{gy}, b_{gz}]^T \in \mathbb{R}^3$: Gyroscope bias in Body frame ($\text{rad/s}$).

#### Error State Vector:
$$\delta \mathbf{x} = \begin{bmatrix} \delta \mathbf{p}_n \\ \delta \mathbf{v}_n \\ \delta \boldsymbol{\theta} \\ \delta \mathbf{b}_a \\ \delta \mathbf{b}_g \end{bmatrix} \in \mathbb{R}^{15}$$
where:
- $\delta \mathbf{p}_n \in \mathbb{R}^3$: Position error vector in Navigation frame.
- $\delta \mathbf{v}_n \in \mathbb{R}^3$: Velocity error vector in Navigation frame.
- $\delta \boldsymbol{\theta} \in \mathbb{R}^3$: Small-angle rotation error vector in **Body frame**.
- $\delta \mathbf{b}_a \in \mathbb{R}^3$: Accelerometer bias error in Body frame.
- $\delta \mathbf{b}_g \in \mathbb{R}^3$: Gyroscope bias error in Body frame.

### Simple Explanation
Imagine driving a car in thick fog. You have a speedometer and a steering wheel, but your instruments are slightly out of calibration.
Instead of trying to predict all the crazy curves of the road with a complex, fragile math formula, we split the job into two teammates:
1. **The Fast Navigator (Nominal State)**: Rapidly tracks your position and speed step-by-step using the raw sensor readings.
2. **The Quiet Detective (Error State)**: Sits in the back seat and only calculates: *"How much has the navigator drifted off course, and how biased are the sensors?"*
Because the errors are tiny and smooth, the detective can use simple, rock-solid linear math. Once the detective figures out the error, he tells the navigator: *"Nudge the car 2 meters north and adjust your speedometer by 0.1 m/s!"*

### Why This Matters for Our Project
Standard Kalman filters (EKF) track the full orientation directly, which causes mathematical singularities (gimbal lock with Euler angles) or covariance covariance matrix degradation (normalizing 4-element quaternions inside a 16x16 covariance matrix violates Gaussian assumptions). By using a 15-state minimal error vector with body-frame small angles $\delta \boldsymbol{\theta}$, our filter is mathematically singularity-free, unconditionally stable, and computationally featherweight on mobile hardware.

---

## 2. IMU Sensor Measurement Model

### Technical Derivation
A tri-axial inertial measurement unit rigidly attached to the vehicle body measures specific force $\mathbf{f}_m$ and angular rate $\boldsymbol{\omega}_m$:
$$\mathbf{f}_m = \mathbf{f}_b + \mathbf{b}_a + \mathbf{n}_a$$
$$\boldsymbol{\omega}_m = \boldsymbol{\omega}_{ib}^b + \mathbf{b}_g + \mathbf{n}_g$$
where:
- $\mathbf{f}_b$: True specific force exerted on the vehicle body (including reaction to gravity):
  $$\mathbf{f}_b = \mathbf{R}_{bn} (\mathbf{a}_n - \mathbf{g}_n) = \mathbf{R}_{nb}^T (\mathbf{a}_n - \mathbf{g}_n)$$
- $\boldsymbol{\omega}_{ib}^b$: True angular velocity of body frame with respect to inertial frame.
- $\mathbf{b}_a, \mathbf{b}_g$: Quasi-static sensor biases, modeled as random walks:
  $$\dot{\mathbf{b}}_a = \mathbf{n}_{ba}, \quad \mathbf{n}_{ba} \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}_{ba})$$
  $$\dot{\mathbf{b}}_g = \mathbf{n}_{bg}, \quad \mathbf{n}_{bg} \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}_{bg})$$
- $\mathbf{n}_a, \mathbf{n}_g$: Zero-mean uncorrelated Gaussian white noise:
  $$\mathbf{n}_a \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}_a), \quad \mathbf{n}_g \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}_g)$$

When bias-corrected measurements are formed using nominal estimates $\hat{\mathbf{b}}_a$ and $\hat{\mathbf{b}}_g$:
$$\hat{\mathbf{f}}_b = \mathbf{f}_m - \hat{\mathbf{b}}_a = \mathbf{f}_b + (\mathbf{b}_a - \hat{\mathbf{b}}_a) + \mathbf{n}_a = \mathbf{f}_b + \delta \mathbf{b}_a + \mathbf{n}_a$$
$$\hat{\boldsymbol{\omega}}_b = \boldsymbol{\omega}_m - \hat{\mathbf{b}}_g = \boldsymbol{\omega}_{ib}^b + (\mathbf{b}_g - \hat{\mathbf{b}}_g) + \mathbf{n}_g = \boldsymbol{\omega}_{ib}^b + \delta \mathbf{b}_g + \mathbf{n}_g$$

### Simple Explanation
Inertial sensors are imperfect:
1. They have static "stubbornness" (bias) that slowly drifts with temperature.
2. They have high-frequency "jitter" (white noise) from engine vibrations and electrical noise.
When the car is stopped on a flat road, the accelerometer does not read zero; it reads $+9.81\text{ m/s}^2$ pointing straight up because the ground is pushing up against the tires! Our model subtracts the estimated bias and accounts for gravity so we only integrate real vehicle motion.

### Why This Matters for Our Project
If an accelerometer has an uncompensated bias of just $0.05\text{ m/s}^2$ ($0.5\%$ of gravity), integrating it unconstrained for 60 seconds produces:
$$\Delta p = \frac{1}{2} b_a t^2 = \frac{1}{2} (0.05)(60)^2 = 90\text{ meters of error!}$$
By estimating $\mathbf{b}_a$ and $\mathbf{b}_g$ dynamically in the 15-state filter, Phase 5 eliminates runaway quadratic trajectory explosion.

---

## 3. Continuous-Time Nominal Navigation Kinematics

### Technical Derivation
The nominal state evolves according to the non-linear mechanization equations:
$$\dot{\hat{\mathbf{p}}}_n = \hat{\mathbf{v}}_n$$
$$\dot{\hat{\mathbf{v}}}_n = \mathbf{R}_{nb}(\hat{\mathbf{q}}) \hat{\mathbf{f}}_b + \mathbf{g}_n = \mathbf{R}_{nb}(\hat{\mathbf{q}}) (\mathbf{f}_m - \hat{\mathbf{b}}_a) + \begin{bmatrix} 0 \\ 0 \\ -g_0 \end{bmatrix}$$
$$\dot{\hat{\mathbf{q}}} = \frac{1}{2} \hat{\mathbf{q}} \otimes \begin{bmatrix} 0 \\ \hat{\boldsymbol{\omega}}_b \end{bmatrix} = \frac{1}{2} \hat{\mathbf{q}} \otimes \begin{bmatrix} 0 \\ \boldsymbol{\omega}_m - \hat{\mathbf{b}}_g \end{bmatrix}$$
$$\dot{\hat{\mathbf{b}}}_a = \mathbf{0}_{3 \times 1}$$
$$\dot{\hat{\mathbf{b}}}_g = \mathbf{0}_{3 \times 1}$$

Here, $\mathbf{g}_n = [0, 0, -9.80665]^T\text{ m/s}^2$ is the constant gravity vector in ENU coordinates.

### Simple Explanation
These five equations are the laws of physics written for navigation:
- Speed changes position: $\text{distance} = \text{speed} \times \text{time}$.
- Acceleration changes speed, but only after you rotate the accelerometer reading from the car's frame to the world's frame and cancel gravity.
- Gyroscope rates rotate your compass direction (quaternion).
- Sensor biases stay constant until the Kalman update corrects them.

### Why This Matters for Our Project
This nominal mechanization runs rapidly at $10\text{ Hz}$ on calibrated Phase 3 signals without invoking heavy matrix inverses, guaranteeing high-rate smooth trajectory output with minimal CPU overhead.

---

## 4. Continuous-Time Error-State Dynamics

### Technical Derivation
We now derive the differential equations governing the error state $\delta \mathbf{x}$.

#### 4.1 Attitude Error Kinematics:
The true attitude is related to nominal attitude by:
$$\mathbf{q} = \hat{\mathbf{q}} \otimes \delta \mathbf{q}$$
where $\delta \mathbf{q} \approx [1, \frac{1}{2} \delta \boldsymbol{\theta}]^T$. Taking time derivatives:
$$\dot{\mathbf{q}} = \dot{\hat{\mathbf{q}}} \otimes \delta \mathbf{q} + \hat{\mathbf{q}} \otimes \dot{\delta \mathbf{q}}$$

Substituting the quaternion kinematics $\dot{\mathbf{q}} = \frac{1}{2} \mathbf{q} \otimes \boldsymbol{\omega}_b$ and $\dot{\hat{\mathbf{q}}} = \frac{1}{2} \hat{\mathbf{q}} \otimes \hat{\boldsymbol{\omega}}_b$:
$$\frac{1}{2} \hat{\mathbf{q}} \otimes \delta \mathbf{q} \otimes \boldsymbol{\omega}_b = \frac{1}{2} \hat{\mathbf{q}} \otimes \hat{\boldsymbol{\omega}}_b \otimes \delta \mathbf{q} + \hat{\mathbf{q}} \otimes \dot{\delta \mathbf{q}}$$

Left-multiplying by $\hat{\mathbf{q}}^*$ and retaining first-order terms:
$$\dot{\delta \boldsymbol{\theta}} = - [\hat{\boldsymbol{\omega}}_b]_\times \delta \boldsymbol{\theta} - \delta \mathbf{b}_g - \mathbf{n}_g$$
where $[\mathbf{u}]_\times$ denotes the skew-symmetric cross-product matrix:
$$[\mathbf{u}]_\times = \begin{bmatrix} 0 & -u_z & u_y \\ u_z & 0 & -u_x \\ -u_y & u_x & 0 \end{bmatrix}$$

#### 4.2 Velocity Error Kinematics:
The true velocity derivative is:
$$\dot{\mathbf{v}}_n = \mathbf{R}_{nb}(\mathbf{q}) \mathbf{f}_b + \mathbf{g}_n$$
Using the first-order rotation perturbation $\mathbf{R}_{nb}(\mathbf{q}) \approx \mathbf{R}_{nb}(\hat{\mathbf{q}}) (\mathbf{I}_{3 \times 3} + [\delta \boldsymbol{\theta}]_\times)$:
$$\dot{\hat{\mathbf{v}}}_n + \delta \dot{\mathbf{v}}_n = \mathbf{R}_{nb}(\hat{\mathbf{q}}) (\mathbf{I} + [\delta \boldsymbol{\theta}]_\times) (\hat{\mathbf{f}}_b - \delta \mathbf{b}_a - \mathbf{n}_a) + \mathbf{g}_n$$

Subtracting the nominal equation $\dot{\hat{\mathbf{v}}}_n = \mathbf{R}_{nb}(\hat{\mathbf{q}}) \hat{\mathbf{f}}_b + \mathbf{g}_n$ and neglecting second-order products:
$$\delta \dot{\mathbf{v}}_n = \mathbf{R}_{nb}(\hat{\mathbf{q}}) [\delta \boldsymbol{\theta}]_\times \hat{\mathbf{f}}_b - \mathbf{R}_{nb}(\hat{\mathbf{q}}) \delta \mathbf{b}_a - \mathbf{R}_{nb}(\hat{\mathbf{q}}) \mathbf{n}_a$$

Using the identity $[\mathbf{a}]_\times \mathbf{b} = -[\mathbf{b}]_\times \mathbf{a}$:
$$\delta \dot{\mathbf{v}}_n = - \mathbf{R}_{nb}(\hat{\mathbf{q}}) [\hat{\mathbf{f}}_b]_\times \delta \boldsymbol{\theta} - \mathbf{R}_{nb}(\hat{\mathbf{q}}) \delta \mathbf{b}_a - \mathbf{R}_{nb}(\hat{\mathbf{q}}) \mathbf{n}_a$$

#### 4.3 Position Error Kinematics:
$$\delta \dot{\mathbf{p}}_n = \delta \mathbf{v}_n$$

#### 4.4 Bias Error Kinematics:
$$\delta \dot{\mathbf{b}}_a = \mathbf{n}_{ba}$$
$$\delta \dot{\mathbf{b}}_g = \mathbf{n}_{bg}$$

#### 4.5 Consolidated Continuous System:
$$\delta \dot{\mathbf{x}}(t) = \mathbf{F}_c(t) \delta \mathbf{x}(t) + \mathbf{G}_c(t) \mathbf{w}(t)$$
where the continuous state transition matrix $\mathbf{F}_c \in \mathbb{R}^{15 \times 15}$ is:
$$\mathbf{F}_c = \begin{bmatrix}
\mathbf{0}_{3 \times 3} & \mathbf{I}_{3 \times 3} & \mathbf{0}_{3 \times 3} & \mathbf{0}_{3 \times 3} & \mathbf{0}_{3 \times 3} \\
\mathbf{0}_{3 \times 3} & \mathbf{0}_{3 \times 3} & -\mathbf{R}_{nb}[\hat{\mathbf{f}}_b]_\times & -\mathbf{R}_{nb} & \mathbf{0}_{3 \times 3} \\
\mathbf{0}_{3 \times 3} & \mathbf{0}_{3 \times 3} & -[\hat{\boldsymbol{\omega}}_b]_\times & \mathbf{0}_{3 \times 3} & -\mathbf{I}_{3 \times 3} \\
\mathbf{0}_{3 \times 3} & \mathbf{0}_{3 \times 3} & \mathbf{0}_{3 \times 3} & \mathbf{0}_{3 \times 3} & \mathbf{0}_{3 \times 3} \\
\mathbf{0}_{3 \times 3} & \mathbf{0}_{3 \times 3} & \mathbf{0}_{3 \times 3} & \mathbf{0}_{3 \times 3} & \mathbf{0}_{3 \times 3}
\end{bmatrix}$$

and the noise coupling matrix $\mathbf{G}_c \in \mathbb{R}^{15 \times 12}$ is:
$$\mathbf{G}_c = \begin{bmatrix}
\mathbf{0}_{3 \times 3} & \mathbf{0}_{3 \times 3} & \mathbf{0}_{3 \times 3} & \mathbf{0}_{3 \times 3} \\
-\mathbf{R}_{nb} & \mathbf{0}_{3 \times 3} & \mathbf{0}_{3 \times 3} & \mathbf{0}_{3 \times 3} \\
\mathbf{0}_{3 \times 3} & -\mathbf{I}_{3 \times 3} & \mathbf{0}_{3 \times 3} & \mathbf{0}_{3 \times 3} \\
\mathbf{0}_{3 \times 3} & \mathbf{0}_{3 \times 3} & \mathbf{I}_{3 \times 3} & \mathbf{0}_{3 \times 3} \\
\mathbf{0}_{3 \times 3} & \mathbf{0}_{3 \times 3} & \mathbf{0}_{3 \times 3} & \mathbf{I}_{3 \times 3}
\end{bmatrix}, \quad
\mathbf{w}(t) = \begin{bmatrix} \mathbf{n}_a \\ \mathbf{n}_g \\ \mathbf{n}_{ba} \\ \mathbf{n}_{bg} \end{bmatrix}$$

### Simple Explanation
This shows how errors spread through the system like falling dominos:
- If your gyro has a tiny angle error ($\delta\boldsymbol{\theta}$), your accelerometer points in the wrong direction, leaking Earth's gravity into your forward speed ($\delta\mathbf{v}$).
- If your speed has an error ($\delta\mathbf{v}$), your position drifts off course ($\delta\mathbf{p}$).
- The matrix $\mathbf{F}_c$ is the mathematical blueprint of exactly how each error contaminates every other error.

### Why This Matters for Our Project
Notice the block $-\mathbf{R}_{nb} [\hat{\mathbf{f}}_b]_\times$ in row 2, column 3. This proves mathematically that attitude tilt directly induces acceleration error. Without this cross-coupling block in the Kalman filter, the system could never learn to correct heading drift from velocity measurements.

---

## 5. Discrete-Time System Propagation (F and G Matrices)

### Technical Derivation
Over sample interval $\Delta t = t_{k+1} - t_k$, the continuous system is discretized. Using first-order Taylor series expansion:
$$\mathbf{F}_k = \exp(\mathbf{F}_c \Delta t) \approx \mathbf{I}_{15 \times 15} + \mathbf{F}_c \Delta t + \frac{1}{2} \mathbf{F}_c^2 \Delta t^2$$

For our navigation filter with $\Delta t = 0.100\text{ s}$, second-order position and velocity expansion provides high accuracy:

$$\mathbf{F}_k = \begin{bmatrix}
\mathbf{I}_3 & \mathbf{I}_3 \Delta t & -\frac{1}{2} \mathbf{R}_{nb} [\hat{\mathbf{f}}_b]_\times \Delta t^2 & -\frac{1}{2} \mathbf{R}_{nb} \Delta t^2 & \mathbf{0}_3 \\
\mathbf{0}_3 & \mathbf{I}_3 & -\mathbf{R}_{nb} [\hat{\mathbf{f}}_b]_\times \Delta t & -\mathbf{R}_{nb} \Delta t & \mathbf{0}_3 \\
\mathbf{0}_3 & \mathbf{0}_3 & \mathbf{I}_3 - [\hat{\boldsymbol{\omega}}_b]_\times \Delta t & \mathbf{0}_3 & -\mathbf{I}_3 \Delta t \\
\mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{I}_3 & \mathbf{0}_3 \\
\mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{I}_3
\end{bmatrix}$$

### Nominal Discrete Propagation
For nominal state integration over $\Delta t$:
$$\hat{\mathbf{p}}_{k+1} = \hat{\mathbf{p}}_k + \hat{\mathbf{v}}_k \Delta t + \frac{1}{2} \mathbf{a}_{n, k} \Delta t^2$$
$$\hat{\mathbf{v}}_{k+1} = \hat{\mathbf{v}}_k + \mathbf{a}_{n, k} \Delta t$$
where $\mathbf{a}_{n, k} = \mathbf{R}_{nb}(\hat{\mathbf{q}}_k) (\mathbf{f}_{m, k} - \hat{\mathbf{b}}_{a, k}) + \mathbf{g}_n$.

Attitude quaternion propagation uses the exact matrix exponential for angular rate $\hat{\boldsymbol{\omega}}_k = \boldsymbol{\omega}_{m, k} - \hat{\mathbf{b}}_{g, k}$:
$$\Delta \theta = \|\hat{\boldsymbol{\omega}}_k\| \Delta t$$
$$\Delta \mathbf{q} = \begin{cases}
\begin{bmatrix} \cos(\Delta \theta / 2) \\ \frac{\sin(\Delta \theta / 2)}{\Delta \theta} \hat{\boldsymbol{\omega}}_k \Delta t \end{bmatrix} & \text{if } \Delta \theta > 10^{-8} \\
\begin{bmatrix} 1.0 \\ \frac{1}{2} \hat{\boldsymbol{\omega}}_k \Delta t \end{bmatrix} & \text{if } \Delta \theta \le 10^{-8}
\end{cases}$$
$$\hat{\mathbf{q}}_{k+1} = \hat{\mathbf{q}}_k \otimes \Delta \mathbf{q}, \quad \hat{\mathbf{q}}_{k+1} \leftarrow \frac{\hat{\mathbf{q}}_{k+1}}{\|\hat{\mathbf{q}}_{k+1}\|}$$

---

## 6. Process Noise Formulation (Q Matrix)

### Technical Derivation
Continuous white noise processes have spectral densities:
- Accelerometer noise spectral density: $\mathbf{S}_a = \sigma_a^2 \mathbf{I}_3\text{ (m}^2/\text{s}^3)$
- Gyroscope noise spectral density: $\mathbf{S}_g = \sigma_g^2 \mathbf{I}_3\text{ (rad}^2/\text{s})$
- Accelerometer bias random walk density: $\mathbf{S}_{ba} = \sigma_{ba}^2 \mathbf{I}_3\text{ (m}^2/\text{s}^5)$
- Gyroscope bias random walk density: $\mathbf{S}_{bg} = \sigma_{bg}^2 \mathbf{I}_3\text{ (rad}^2/\text{s}^3)$

The discrete process noise covariance $\mathbf{Q}_k$ is computed via van Loan's method or the standard trapezoidal approximation:
$$\mathbf{Q}_k \approx \mathbf{G}_c \mathbf{Q}_{\text{continuous}} \mathbf{G}_c^T \Delta t$$

The diagonal block elements of $\mathbf{Q}_k \in \mathbb{R}^{15 \times 15}$ are:
$$\mathbf{Q}_{\delta p} = \frac{1}{3} \sigma_a^2 \Delta t^3 \mathbf{I}_3$$
$$\mathbf{Q}_{\delta v} = \sigma_a^2 \Delta t \mathbf{I}_3$$
$$\mathbf{Q}_{\delta \theta} = \sigma_g^2 \Delta t \mathbf{I}_3$$
$$\mathbf{Q}_{\delta ba} = \sigma_{ba}^2 \Delta t \mathbf{I}_3$$
$$\mathbf{Q}_{\delta bg} = \sigma_{bg}^2 \Delta t \mathbf{I}_3$$

Cross-terms between position and velocity noise:
$$\mathbf{Q}_{\delta p, \delta v} = \mathbf{Q}_{\delta v, \delta p}^T = \frac{1}{2} \sigma_a^2 \Delta t^2 \mathbf{I}_3$$

### Realistic Sensor Parameters for IO-VNBD Smartphone IMU
Estimated from stationary sequence Vw1 ($3,524\text{ samples}$):
- $\sigma_a = 0.08\text{ m/s}^2 / \sqrt{\text{Hz}}$
- $\sigma_g = 0.005\text{ rad/s} / \sqrt{\text{Hz}}$
- $\sigma_{ba} = 1.0 \times 10^{-4}\text{ m/s}^3 / \sqrt{\text{Hz}}$
- $\sigma_{bg} = 1.0 \times 10^{-5}\text{ rad/s}^2 / \sqrt{\text{Hz}}$

---

## 7. Covariance Propagation & Numerical Stabilization

### Technical Derivation
At each IMU step, the $15 \times 15$ error covariance matrix $\mathbf{P}_k$ propagates forward:
$$\mathbf{P}_{k+1}^- = \mathbf{F}_k \mathbf{P}_k \mathbf{F}_k^T + \mathbf{Q}_k$$

#### Numerical Stabilization:
Because roundoff errors accumulate over tens of thousands of matrix multiplications, $\mathbf{P}$ can lose symmetry and positive-definiteness. We apply forced symmetrization after every propagation step:
$$\mathbf{P} \leftarrow \frac{1}{2} (\mathbf{P} + \mathbf{P}^T)$$

Furthermore, all diagonal elements are monitored:
$$P_{ii} \leftarrow \max(P_{ii}, \epsilon_{\text{floor}}), \quad \epsilon_{\text{floor}} = 10^{-12}$$
If any diagonal element becomes negative, NaN, or infinite, a numerical fault is immediately logged.

---

## 8. Phase 4 AI Forward-Speed Measurement Model

### Technical Derivation
At time $t_k$, the Phase 4 Motion Intelligence module outputs:
$$z_{v, k} = \hat{v}_{\text{fwd}, k} \in \mathbb{R}_{\ge 0}$$
with associated predicted uncertainty $\sigma_{v, k} \in \mathbb{R}_{> 0}$.

The measurement function $h(\mathbf{x}_k)$ relates the nominal state to forward speed. In the vehicle body frame, forward speed is the $x$-component of body velocity:
$$\mathbf{v}_b = \mathbf{R}_{bn} \mathbf{v}_n = \mathbf{R}_{nb}^T \mathbf{v}_n$$
$$h(\mathbf{x}_k) = \mathbf{e}_1^T \mathbf{v}_b = \begin{bmatrix} 1 & 0 & 0 \end{bmatrix} \mathbf{R}_{nb}^T(\mathbf{q}_k) \mathbf{v}_{n, k}$$
where $\mathbf{e}_1 = [1, 0, 0]^T$ is the longitudinal forward selector vector.

The scalar measurement equation is:
$$z_{v, k} = h(\mathbf{x}_k) + r_k, \quad r_k \sim \mathcal{N}(0, R_k)$$
where:
$$R_k = \sigma_{v, k}^2$$

---

## 9. Measurement Jacobian Matrix (H Matrix)

### Technical Derivation
The measurement Jacobian $\mathbf{H}_k \in \mathbb{R}^{1 \times 15}$ is defined as the partial derivative of $h(\mathbf{x} \oplus \delta \mathbf{x})$ with respect to the error state $\delta \mathbf{x}$:
$$\mathbf{H}_k = \left. \frac{\partial h}{\partial \delta \mathbf{x}} \right|_{\mathbf{x} = \hat{\mathbf{x}}}$$

Let us perturb each component of the state:
$$h(\hat{\mathbf{x}} \oplus \delta \mathbf{x}) = \mathbf{e}_1^T \mathbf{R}_{nb}^T(\hat{\mathbf{q}} \otimes \delta \mathbf{q}) (\hat{\mathbf{v}}_n + \delta \mathbf{v}_n)$$

Using $\mathbf{R}_{nb}(\hat{\mathbf{q}} \otimes \delta \mathbf{q}) \approx \mathbf{R}_{nb}(\hat{\mathbf{q}}) (\mathbf{I} + [\delta \boldsymbol{\theta}]_\times)$:
$$\mathbf{R}_{nb}^T(\hat{\mathbf{q}} \otimes \delta \mathbf{q}) = (\mathbf{I} - [\delta \boldsymbol{\theta}]_\times) \mathbf{R}_{nb}^T(\hat{\mathbf{q}})$$

Substituting into the measurement function:
$$h(\hat{\mathbf{x}} \oplus \delta \mathbf{x}) \approx \mathbf{e}_1^T (\mathbf{I} - [\delta \boldsymbol{\theta}]_\times) \mathbf{R}_{nb}^T (\hat{\mathbf{v}}_n + \delta \mathbf{v}_n)$$
$$h \approx \mathbf{e}_1^T \mathbf{R}_{nb}^T \hat{\mathbf{v}}_n + \mathbf{e}_1^T \mathbf{R}_{nb}^T \delta \mathbf{v}_n - \mathbf{e}_1^T [\delta \boldsymbol{\theta}]_\times \mathbf{R}_{nb}^T \hat{\mathbf{v}}_n$$

Notice that $\mathbf{R}_{nb}^T \hat{\mathbf{v}}_n = \hat{\mathbf{v}}_b = [\hat{v}_{\text{fwd}}, \hat{v}_{\text{lat}}, \hat{v}_{\text{up}}]^T$ is the nominal body-frame velocity.
Using the vector identity $[\mathbf{a}]_\times \mathbf{b} = - [\mathbf{b}]_\times \mathbf{a}$:
$$- \mathbf{e}_1^T [\delta \boldsymbol{\theta}]_\times \hat{\mathbf{v}}_b = \mathbf{e}_1^T [\hat{\mathbf{v}}_b]_\times \delta \boldsymbol{\theta}$$

Therefore:
$$h(\hat{\mathbf{x}} \oplus \delta \mathbf{x}) = h(\hat{\mathbf{x}}) + \mathbf{e}_1^T \mathbf{R}_{nb}^T \delta \mathbf{v}_n + \mathbf{e}_1^T [\hat{\mathbf{v}}_b]_\times \delta \boldsymbol{\theta}$$

Evaluating the partial derivatives across all 5 error blocks:
1. **Position Error**:
   $$\frac{\partial h}{\partial \delta \mathbf{p}_n} = \mathbf{0}_{1 \times 3}$$
2. **Velocity Error**:
   $$\frac{\partial h}{\partial \delta \mathbf{v}_n} = \mathbf{e}_1^T \mathbf{R}_{nb}^T = \mathbf{R}_{nb}[:, 0]^T \in \mathbb{R}^{1 \times 3}$$
3. **Attitude Error**:
   $$\frac{\partial h}{\partial \delta \boldsymbol{\theta}} = \mathbf{e}_1^T [\hat{\mathbf{v}}_b]_\times = \begin{bmatrix} 1 & 0 & 0 \end{bmatrix} \begin{bmatrix} 0 & -\hat{v}_{\text{up}} & \hat{v}_{\text{lat}} \\ \hat{v}_{\text{up}} & 0 & -\hat{v}_{\text{fwd}} \\ -\hat{v}_{\text{lat}} & \hat{v}_{\text{fwd}} & 0 \end{bmatrix} = \begin{bmatrix} 0 & -\hat{v}_{\text{up}} & \hat{v}_{\text{lat}} \end{bmatrix}$$
4. **Accelerometer Bias Error**:
   $$\frac{\partial h}{\partial \delta \mathbf{b}_a} = \mathbf{0}_{1 \times 3}$$
5. **Gyroscope Bias Error**:
   $$\frac{\partial h}{\partial \delta \mathbf{b}_g} = \mathbf{0}_{1 \times 3}$$

Consolidating into the $1 \times 15$ measurement Jacobian:
$$\boxed{\mathbf{H}_k = \begin{bmatrix} \mathbf{0}_{1 \times 3} & \mathbf{e}_1^T \mathbf{R}_{nb}^T & \begin{bmatrix} 0 & -\hat{v}_{\text{up}} & \hat{v}_{\text{lat}} \end{bmatrix} & \mathbf{0}_{1 \times 3} & \mathbf{0}_{1 \times 3} \end{bmatrix}}$$

### Simple Explanation
When the AI tells the filter: *"The car is moving forward at 15 m/s"*, the filter asks:
- Does this tell me my $(x, y)$ position? **No** (0 in position column).
- Does this tell me my velocity? **Yes!** If my speed is wrong, my velocity vector is wrong.
- Does this tell me my tilt/heading? **Yes!** If the car is turning, lateral velocity appears, coupling speed with pitch and yaw errors.

---

## 10. Observability & Information Analysis

### Technical Derivation
The observability of the error state $(\mathbf{F}_k, \mathbf{H}_k)$ reveals which physical states are observable from forward speed alone:
1. **Directly Observable**:
   - Velocity along the forward trajectory axis: $\mathbf{H}$ projects directly onto $\delta \mathbf{v}_n$.
2. **Coupled Observable (during accelerations and turns)**:
   - Pitch angle $\theta$: Gravity component along longitudinal axis $a_{\text{fwd}} = \dot{v} - g \sin\theta$ couples pitch with acceleration.
   - Longitudinal accelerometer bias $b_{ax}$: Indistinguishable from constant acceleration during steady cruising, but separated during dynamic speed transients ($\dot{v} \ne 0$).
3. **Unobservable Subspace**:
   - **Absolute Position $\mathbf{p}_n$**: Speed provides only relative displacement rate. The absolute translation origin is unobservable without GNSS.
   - **Absolute Yaw Angle $\psi$ (during straight cruising)**: Rotating the entire ENU coordinate system around vertical $z_n$ preserves forward speed magnitude. Yaw drift is partially constrained during turns via centripetal acceleration $a_{\text{lat}} = v \cdot \omega_{\text{yaw}}$, but cannot be fully stabilized without magnetometer, NHC, or GNSS.

---

## 11. Kalman Correction & Joseph Form Covariance Update

### Technical Derivation
When an AI speed measurement $z_k$ arrives:

#### 1. Innovation (Residual):
$$\nu_k = z_k - h(\hat{\mathbf{x}}_k^-) = \hat{v}_{\text{fwd}, k} - \mathbf{e}_1^T \mathbf{R}_{nb}^T \hat{\mathbf{v}}_n^-$$

#### 2. Innovation Covariance (Scalar):
$$S_k = \mathbf{H}_k \mathbf{P}_k^- \mathbf{H}_k^T + R_k$$

#### 3. Kalman Gain (15x1 Column Vector):
$$\mathbf{K}_k = \mathbf{P}_k^- \mathbf{H}_k^T S_k^{-1} = \frac{\mathbf{P}_k^- \mathbf{H}_k^T}{S_k}$$

#### 4. Error State Estimate:
$$\delta \hat{\mathbf{x}}_k = \mathbf{K}_k \nu_k \in \mathbb{R}^{15}$$

#### 5. Joseph Form Covariance Update:
Standard textbook covariance update $\mathbf{P}^+ = (\mathbf{I} - \mathbf{K}\mathbf{H}) \mathbf{P}^-$ is notorious for numerical instability because subtraction of positive semi-definite matrices can produce negative eigenvalues.
We implement the **Joseph Stabilized Covariance Update**:
$$\boxed{\mathbf{P}_k^+ = (\mathbf{I}_{15} - \mathbf{K}_k \mathbf{H}_k) \mathbf{P}_k^- (\mathbf{I}_{15} - \mathbf{K}_k \mathbf{H}_k)^T + \mathbf{K}_k R_k \mathbf{K}_k^T}$$

Because both terms are quadratic forms ($\mathbf{A} \mathbf{P} \mathbf{A}^T$ and $\mathbf{B} R \mathbf{B}^T$), this update is **strictly guaranteed** to remain symmetric and positive semi-definite for any Kalman gain $\mathbf{K}$.

---

## 12. Normalized Innovation Squared (NIS) Gating

### Technical Derivation
To prevent corrupted or outlier speed estimates (such as when the phone shifts in its mount or undergoes violent shaking) from contaminating the filter, we compute the Normalized Innovation Squared (NIS):
$$\text{NIS}_k = \nu_k^T S_k^{-1} \nu_k = \frac{\nu_k^2}{S_k}$$

Under nominal Gaussian conditions, $\text{NIS}_k$ follows a Chi-Square distribution with $m=1$ degree of freedom:
$$\text{NIS}_k \sim \chi^2(1)$$

For a $99.7\%$ two-sided confidence bound ($\alpha = 0.003$):
$$\gamma_{\text{threshold}} = \chi_{0.997}^2(1) \approx 9.0 \quad (3\sigma \text{ bound})$$

#### Decision Rule:
$$\text{Action} = \begin{cases}
\text{ACCEPT}: \text{Execute Kalman update with } \delta \hat{\mathbf{x}} = \mathbf{K} \nu & \text{if } \text{NIS}_k \le 9.0 \\
\text{REJECT}: \text{Skip update, maintain nominal propagation } \mathbf{P}^+ = \mathbf{P}^- & \text{if } \text{NIS}_k > 9.0
\end{cases}$$

---

## 13. Error State Injection & Attitude Reset

### Technical Derivation
After the error state $\delta \hat{\mathbf{x}}_k$ is computed, it must be injected into the nominal state:

1. **Position Injection**:
   $$\hat{\mathbf{p}}_n^+ = \hat{\mathbf{p}}_n^- + \delta \hat{\mathbf{p}}_n$$
2. **Velocity Injection**:
   $$\hat{\mathbf{v}}_n^+ = \hat{\mathbf{v}}_n^- + \delta \hat{\mathbf{v}}_n$$
3. **Attitude Quaternion Injection**:
   $$\hat{\mathbf{q}}^+ = \hat{\mathbf{q}}^- \otimes \begin{bmatrix} 1 \\ \frac{1}{2} \delta \hat{\boldsymbol{\theta}} \end{bmatrix}$$
   $$\hat{\mathbf{q}}^+ \leftarrow \frac{\hat{\mathbf{q}}^+}{\|\hat{\mathbf{q}}^+\|}$$
4. **Accelerometer Bias Injection**:
   $$\hat{\mathbf{b}}_a^+ = \hat{\mathbf{b}}_a^- + \delta \hat{\mathbf{b}}_a$$
5. **Gyroscope Bias Injection**:
   $$\hat{\mathbf{b}}_g^+ = \hat{\mathbf{b}}_g^- + \delta \hat{\mathbf{b}}_g$$

#### Error State Reset:
Because the error has been fully transferred to the nominal state, the error state expectation resets to zero:
$$\delta \hat{\mathbf{x}} \leftarrow \mathbf{0}_{15 \times 1}$$

The covariance $\mathbf{P}^+$ correctly reflects the residual uncertainty after error injection.
