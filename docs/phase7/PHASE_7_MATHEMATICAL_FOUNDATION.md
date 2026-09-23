# Phase 7 Mathematical Foundations: Offline OpenStreetMap Map Matching & Road-Constrained Navigation

**Project**: Smart India Hackathon 2026 — Problem Statement SIH26168  
**Sponsoring Agency**: Indian Space Research Organisation (ISRO)  
**System**: AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation  
**Document**: `docs/phase7/PHASE_7_MATHEMATICAL_FOUNDATION.md`  

---

## 1. Introduction & Theoretical Motivation

In Phase 5 and Phase 6, we developed an inertial dead-reckoning engine combining a 15-state Error-State Kalman Filter (ESKF) with AI forward velocity from smartphone IMU signals, Non-Holonomic Constraints (NHC), and Zero-Velocity/Zero-Angular-Rate Updates (ZUPT/ZARU). 

While Phase 6 successfully halted lateral and vertical velocity divergence and clamped stationary creep, dead reckoning over extended GNSS outages ($>30\text{ s}$) still suffers from fundamental physical limits:
1. **Unobservable Yaw Drift During Straight Motion**: Under constant velocity straight-line driving, gyroscope bias about the vertical axis is only weakly observable from vehicle physics constraints alone. A heading error $\delta\psi$ causes the forward velocity $v_{\text{fwd}}$ to project into an unobserved lateral drift:
   $$\dot{p}_{\text{cross}} \approx v_{\text{fwd}} \sin(\delta\psi) \approx v_{\text{fwd}} \, \delta\psi$$
2. **Longitudinal Dead-Reckoning Integration**: Integrating wheel or AI forward speed without absolute position anchoring leads to unbounded along-track position variance $\sigma_p^2(t) \propto t$.

Phase 7 leverages the geometry and topology of the physical road network stored offline in OpenStreetMap (OSM) to provide external pseudo-measurements. Because vehicles travel along designated road corridors, the road centerline provides:
- **Lateral Position Anchoring**: Directly measuring and bounding cross-track position error.
- **Direct Yaw Observability**: Observing the vehicle's heading relative to the road segment tangent vector $\mathbf{t}_{\text{road}}$.

This document provides the complete mathematical derivation of the Phase 7 map-matching engine, structured into:
- **Layer A**: Rigorous mathematical derivations, Jacobians, Bayesian filters, and covariance formulations.
- **Layer B**: Plain English intuitive explanations and physical rationale.

---

## 2. Road Geometry & Orthogonal Projection

### 2.1 Technical Derivation (Layer A)
Let a directed road segment $s$ in the 2D local ENU plane be defined by start vertex $\mathbf{p}_1 = [e_1, n_1]^T$ and end vertex $\mathbf{p}_2 = [e_2, n_2]^T$.

The segment displacement vector and length are:
$$\Delta\mathbf{p} = \mathbf{p}_2 - \mathbf{p}_1, \quad L = \|\Delta\mathbf{p}\|_2$$

The unit tangent vector $\mathbf{t}$ and left-handed unit normal vector $\mathbf{n}$ are:
$$\mathbf{t} = \frac{\Delta\mathbf{p}}{L} = \begin{bmatrix} t_e \\ t_n \end{bmatrix} = \begin{bmatrix} \cos\psi_{\text{road}} \\ \sin\psi_{\text{road}} \end{bmatrix}$$
$$\mathbf{n} = \begin{bmatrix} -t_n \\ t_e \end{bmatrix} = \begin{bmatrix} -\sin\psi_{\text{road}} \\ \cos\psi_{\text{road}} \end{bmatrix}$$
where $\psi_{\text{road}} = \text{atan2}(t_n, t_e) \in [-\pi, \pi]$ is the segment azimuth.

Let the current filter position estimate in the ENU horizontal plane be $\mathbf{p} = [e, n]^T$. The vector from the segment start to the estimated position is:
$$\mathbf{d} = \mathbf{p} - \mathbf{p}_1$$

#### Projection Parameter
The normalized along-track projection parameter $t_{\text{proj}}$ is computed via the dot product with the segment vector:
$$t_{\text{proj}} = \frac{\mathbf{d}^T \Delta\mathbf{p}}{\|\Delta\mathbf{p}\|_2^2} = \frac{\mathbf{d}^T \mathbf{t}}{L}$$

To constrain the projection to the finite physical road segment, $t_{\text{proj}}$ is clamped:
$$t_{\text{clamped}} = \max\left(0, \min(1, t_{\text{proj}})\right)$$

The projected point on the segment centerline $\mathbf{p}_{\text{proj}}$ is:
$$\mathbf{p}_{\text{proj}} = \mathbf{p}_1 + t_{\text{clamped}} \, \Delta\mathbf{p} = \mathbf{p}_1 + t_{\text{clamped}} \, L \, \mathbf{t}$$

#### Cross-Track & Along-Track Metrics
1. **Along-Track Distance**:
   $$s_{\parallel} = t_{\text{clamped}} \, L \in [0, L]$$
2. **Signed Cross-Track Distance**:
   $$d_{\perp} = \mathbf{n}^T (\mathbf{p} - \mathbf{p}_1) = -t_n (e - e_1) + t_e (n - n_1)$$
   The sign indicates which side of the road centerline the vehicle occupies: $d_{\perp} > 0$ denotes left of the centerline, and $d_{\perp} < 0$ denotes right of the centerline.
3. **Orthogonal Euclidean Distance**:
   $$d = \|\mathbf{p} - \mathbf{p}_{\text{proj}}\|_2$$
   When $t_{\text{proj}} \in (0, 1)$, $d = |d_{\perp}|$. At segment endpoints, $d$ accounts for end-point proximity.

### 2.2 Plain English Intuition (Layer B)
Imagine drawing a straight chalk line on the ground between two road markers. If our estimated car position is hovering somewhere nearby, we drop a perpendicular plumb line from the car onto that chalk line.

- The spot where the plumb line hits the chalk is our "projected point" ($\mathbf{p}_{\text{proj}}$).
- The distance along the chalk line is our "along-track position".
- The length of the plumb line is our "cross-track distance" ($d_{\perp}$). If the car is to the left of the line, $d_{\perp}$ is positive; if to the right, it is negative.
- If the car has driven past the end marker, we clamp our plumb line to the tip of the segment so the math never projects into empty space.

### 2.3 Significance for SIH26168
Without orthogonal projection, we cannot determine how far off the road corridor the vehicle has drifted. By projecting onto the exact centerline, we create an error signal that can be directly consumed by the Kalman filter.

---

## 3. Probabilistic Candidate Generation & Scoring

When navigating through complex road networks, multiple roads may lie near the estimated position (e.g., parallel service roads, overpasses, or multi-branch intersections). Hard-snapping to the closest road can be catastrophic if the vehicle is actually on an adjacent street. A probabilistic multi-hypothesis scoring engine is therefore required.

### 3.1 Dynamic Candidate Search Radius (Layer A)
Let the filter horizontal position covariance be:
$$P_{pos} = \begin{bmatrix} P_{ee} & P_{en} \\ P_{ne} & P_{nn} \end{bmatrix}$$
The horizontal position standard deviation is $\sigma_p = \sqrt{\lambda_{\max}(P_{pos})}$, where $\lambda_{\max}$ is the maximum eigenvalue.

The search radius $d_{\text{max}}$ is dynamically scaled by filter uncertainty:
$$d_{\text{max}} = \max\left(d_{\min}, \, \min\left(d_{\text{upper}}, \, k_{\sigma} \sigma_p + d_{\min}\right)\right)$$
where $d_{\min} = 15.0\text{ m}$, $d_{\text{upper}} = 60.0\text{ m}$, and $k_{\sigma} = 2.5$. This guarantees that candidate roads are searched within the $99\%$ confidence ellipse of the ESKF.

### 3.2 Candidate Emission Likelihood $\Lambda(\mathbf{z}_t \mid c_t)$
For each candidate segment $c$, three independent physical likelihoods are evaluated:

#### 1. Distance Likelihood $P_{\text{dist}}(c)$
The cross-track distance $d$ is modeled as a zero-mean Gaussian distribution with standard deviation $\sigma_d = 4.0\text{ m}$ (representing lane width and GPS mapping accuracy):
$$P_{\text{dist}}(c) = \exp\left(-\frac{d^2}{2\sigma_d^2}\right)$$

#### 2. Heading Likelihood $P_{\text{head}}(c)$
Let $\psi_v$ be the vehicle's estimated yaw angle in the ENU plane, and $\psi_{\text{road}}$ be the road segment tangent heading. The angular discrepancy is:
$$\Delta\psi = \text{wrap}_{\pi}(\psi_v - \psi_{\text{road}}) \in [-\pi, \pi]$$
Modeled as a circular normal (von Mises / Gaussian) likelihood with dispersion $\sigma_\psi = 25^\circ \approx 0.436\text{ rad}$:
$$P_{\text{head}}(c) = \exp\left(-\frac{\Delta\psi^2}{2\sigma_\psi^2}\right)$$

#### 3. Dynamic Velocity Likelihood $P_{\text{vel}}(c)$
If the vehicle's forward speed $v_{\text{fwd}}$ exceeds the segment legal speed limit $v_{\text{max}}$ by a tolerance margin $\Delta v_{\text{tol}} = 5.0\text{ m/s}$:
$$P_{\text{vel}}(c) = \begin{cases} 1.0 & v_{\text{fwd}} \le v_{\text{max}} + \Delta v_{\text{tol}} \\ \exp\left(-\frac{(v_{\text{fwd}} - (v_{\text{max}} + \Delta v_{\text{tol}}))^2}{2\sigma_v^2}\right) & v_{\text{fwd}} > v_{\text{max}} + \Delta v_{\text{tol}} \end{cases}$$

#### Total Composite Emission Likelihood
$$\Lambda(\mathbf{z}_t \mid c) = P_{\text{dist}}(c) \cdot P_{\text{head}}(c) \cdot P_{\text{vel}}(c)$$

### 3.3 Plain English Intuition (Layer B)
When the car enters an area with several possible roads, we do not simply pick whichever road is $1\text{ foot}$ closer. We look at three clues:
1. **Distance**: Is the car physically close to this road?
2. **Heading**: Is the car driving in the same direction this road runs? (A highway going north cannot match a car driving east).
3. **Speed**: Is the car driving at a speed that makes sense for this road? (Driving $100\text{ km/h}$ on a tiny residential cul-de-sac is penalized).

Multiplying these probabilities gives an overall confidence score for each candidate road.

---

## 4. Directed Road Topology & Causal Bayesian Tracking

### 4.1 Topology Markov Model (Layer A)
Roads are physically connected into a directed graph $G = (V, E)$. A vehicle cannot teleport instantaneously between disconnected streets. The transition probability between candidate segment $c_{t-1}$ at time $t-1$ and candidate segment $c_t$ at time $t$ is formulated as a first-order Markov process:

$$P(c_t \mid c_{t-1}) = \begin{cases} 
p_{\text{stay}} = 0.85 & \text{if } c_t = c_{t-1} \text{ (same segment)} \\
p_{\text{conn}} = 0.14 & \text{if } c_t \in \text{Successors}(c_{t-1}) \text{ (valid topological connection)} \\
p_{\text{jump}} = 0.0001 & \text{if } c_t \text{ is topologically disconnected}
\end{cases}$$

If candidate $c_t$ is a successor of $c_{t-1}$, the transition is further weighted by the difference between the vehicle's integrated travel distance $\Delta s_{\text{veh}} = \int v_{\text{fwd}} dt$ and the topological graph distance $\Delta s_{\text{topo}}$:
$$P(c_t \mid c_{t-1}) = p_{\text{conn}} \cdot \exp\left(-\frac{|\Delta s_{\text{veh}} - \Delta s_{\text{topo}}|}{\sigma_{\text{topo}}}\right)$$

### 4.2 Causal Forward Recursive Bayesian Filtering
Unlike offline Viterbi smoothing (which uses future frames to retroactively pick the best path), our system must navigate **strictly causally in real time**. 

At each time step $t$, the belief distribution over candidate segments $B_t(c_t)$ is updated recursively:
$$B_t^-(c_t) = \sum_{c_{t-1}} P(c_t \mid c_{t-1}) B_{t-1}(c_{t-1})$$
$$B_t(c_t) = \frac{\Lambda(\mathbf{z}_t \mid c_t) B_t^-(c_t)}{\sum_{c'} \Lambda(\mathbf{z}_t \mid c') B_t^-(c')}$$

The primary candidate is the maximum a posteriori (MAP) segment:
$$c^*_t = \arg\max_{c_t} B_t(c_t)$$

### 4.3 Continuous Map Confidence Metric $c_{\text{map}}$
To protect the Kalman filter from corrupted updates when driving off-road or encountering ambiguous intersections, we define a continuous confidence metric $c_{\text{map}} \in [0, 1]$:
$$c_{\text{map}} = \left(B_t(c^*_t)\right) \cdot \exp\left(-\frac{d(c^*_t)^2}{2\sigma_d^2}\right) \cdot \cos^2(\Delta\psi(c^*_t))$$

- When the vehicle is on a single, clear road with matching heading, $c_{\text{map}} \approx 0.95 - 1.0$.
- In ambiguous multi-road junctions, belief entropy spreads across candidates, reducing $c_{\text{map}} \to 0.3 - 0.5$.
- When leaving the road network entirely (e.g., parking lot or off-road), $c_{\text{map}} \to 0$.

### 4.4 Plain English Intuition (Layer B)
Vehicles cannot jump across buildings or flip instantaneously from a highway to an underpass. If the car was on Highway A a tenth of a second ago, it is almost certainly still on Highway A (85% probability) or smoothly transitioning onto an off-ramp connected to Highway A (14% probability).

Even if sensor noise momentarily makes an adjacent side street look slightly closer, the topological Markov model says: *"You cannot have reached that side street yet without flying over a concrete barrier!"* This prevents false-switching on parallel roads.

---

## 5. Indirect Error-State Kalman Filter Updates

Rather than "hard snapping" the filter state (which destroys the covariance matrix and introduces artificial discontinuous velocity spikes), Phase 7 uses **soft Kalman measurement updates**.

The 15-state indirect error state is:
$$\delta\mathbf{x} = \begin{bmatrix} \delta\mathbf{p}_n^T & \delta\mathbf{v}_n^T & \delta\boldsymbol{\theta}^T & \delta\mathbf{b}_a^T & \delta\mathbf{b}_g^T \end{bmatrix}^T \in \mathbb{R}^{15}$$

### 5.1 Cross-Track Position Measurement Update (Layer A)

#### Observation Model
The physical constraint is that the vehicle lies on the road centerline. The scalar virtual measurement is $z_p = 0$.

The non-linear observation model $h_p(\mathbf{x})$ computes the signed cross-track distance to the matched segment:
$$h_p(\mathbf{x}) = \mathbf{n}^T (\mathbf{p}_{2D} - \mathbf{p}_1) = -t_n (e - e_1) + t_e (n - n_1)$$
where $\mathbf{n} = [-t_n, t_e]^T$ is the unit normal vector of the road segment in the ENU plane.

The innovation residual $\nu_p$ is:
$$\nu_p = z_p - h_p(\hat{\mathbf{x}}) = 0 - d_{\perp}(\hat{\mathbf{p}}) = -d_{\perp}$$

#### Analytical Measurement Jacobian $\mathbf{H}_p \in \mathbb{R}^{1 \times 15}$
Linearizing with respect to the error state $\delta\mathbf{x}$:
$$\frac{\partial h_p}{\partial \delta\mathbf{p}_n} = \begin{bmatrix} \frac{\partial h_p}{\partial \delta e} & \frac{\partial h_p}{\partial \delta n} & \frac{\partial h_p}{\partial \delta u} \end{bmatrix} = \begin{bmatrix} -t_n & t_e & 0 \end{bmatrix} = \begin{bmatrix} \mathbf{n}^T & 0 \end{bmatrix}$$

Since cross-track position does not depend directly on velocity, attitude error, or sensor biases:
$$\mathbf{H}_p = \begin{bmatrix} -t_n & t_e & 0 & \mathbf{0}_{1 \times 3} & \mathbf{0}_{1 \times 3} & \mathbf{0}_{1 \times 3} & \mathbf{0}_{1 \times 3} \end{bmatrix} \in \mathbb{R}^{1 \times 15}$$

#### Adaptive Measurement Covariance
The nominal cross-track observation variance is $\sigma_{\text{cross}, 0}^2 = (1.5\text{ m})^2$ (accounting for road half-width). It is dynamically inflated by map confidence $c_{\text{map}}$:
$$R_p = \frac{\sigma_{\text{cross}, 0}^2}{c_{\text{map}}^2 + \epsilon_{\text{map}}}$$
When confidence is low ($c_{\text{map}} \to 0$), $R_p \to \infty$, gracefully disengaging the measurement update without filter destabilization.

---

### 5.2 Road Heading Measurement Update (Layer A)

#### Observation Model
The road segment defines an absolute heading reference $\psi_{\text{road}} = \text{atan2}(t_n, t_e)$.

The measurement is $z_\psi = \psi_{\text{road}}$.

The observation function $h_\psi(\mathbf{x})$ extracts the yaw angle $\psi$ from the vehicle attitude quaternion $\mathbf{q} = [q_w, q_x, q_y, q_z]^T$:
$$\psi = \text{atan2}\left(2(q_w q_z + q_x q_y), \, 1 - 2(q_y^2 + q_z^2)\right)$$

The innovation residual $\nu_\psi$ is wrapped to $[-\pi, \pi]$:
$$\nu_\psi = \text{wrap}_{\pi}(z_\psi - \hat{\psi})$$

#### Analytical Measurement Jacobian $\mathbf{H}_\psi \in \mathbb{R}^{1 \times 15}$
Recalling the ESKF attitude error definition $\mathbf{q} \approx \hat{\mathbf{q}} \otimes [1, \, \frac{1}{2}\delta\boldsymbol{\theta}^T]^T$:
A small rotation perturbation $\delta\boldsymbol{\theta} = [\delta\theta_x, \delta\theta_y, \delta\theta_z]^T$ in the body frame produces a change in yaw:
$$\delta\psi \approx \frac{\partial \psi}{\partial \delta\boldsymbol{\theta}} \delta\boldsymbol{\theta}$$

For near-level driving (roll $\phi \approx 0$, pitch $\theta \approx 0$), yaw error relates directly to the vertical body axis rotation:
$$\frac{\partial \psi}{\partial \delta\boldsymbol{\theta}} = \begin{bmatrix} 0 & 0 & 1 \end{bmatrix}$$
More generally, using the rotation matrix from body to navigation frame $R_{nb}$:
$$\frac{\partial \psi}{\partial \delta\boldsymbol{\theta}} = \mathbf{e}_3^T R_{nb} \approx \begin{bmatrix} 0 & 0 & 1 \end{bmatrix}$$

Therefore, the full heading Jacobian is:
$$\mathbf{H}_\psi = \begin{bmatrix} \mathbf{0}_{1 \times 3} & \mathbf{0}_{1 \times 3} & 0 & 0 & 1 & \mathbf{0}_{1 \times 3} & \mathbf{0}_{1 \times 3} \end{bmatrix} \in \mathbb{R}^{1 \times 15}$$

#### Adaptive Measurement Covariance
The nominal heading variance is $\sigma_{\text{head}, 0}^2 = (5.0^\circ \times \frac{\pi}{180})^2 \approx (0.0873\text{ rad})^2$. It is dynamically modulated:
$$R_\psi = \frac{\sigma_{\text{head}, 0}^2}{c_{\text{map}}^2 + \epsilon_{\text{map}}}$$

---

### 5.3 Chi-Square Innovation Gating (Layer A)
Before applying any measurement update, the Normalized Innovation Squared (NIS) is checked against a $\chi^2$ threshold to reject spurious outliers:

$$d_M^2 = \frac{\nu^2}{S} = \frac{\nu^2}{\mathbf{H} P^- \mathbf{H}^T + R}$$

For a 1-DOF scalar measurement ($k=1$) at significance level $\alpha = 0.01$ ($99\%$ confidence):
$$\gamma_{\text{gate}} = \chi_1^2(0.99) = 6.635$$

- If $d_M^2 \le 6.635$: Innovation accepted; state and covariance updated.
- If $d_M^2 > 6.635$: Innovation rejected as an outlier (e.g., during off-road detour or sharp lane change).

### 5.4 Joseph-Form Stabilized Covariance Update (Layer A)
To guarantee strict positive semi-definiteness and prevent numerical degradation during long outage runs:

$$\mathbf{K} = P^- \mathbf{H}^T S^{-1}$$
$$\delta\hat{\mathbf{x}} = \mathbf{K} \nu$$
$$P^+ = (I_{15} - \mathbf{K}\mathbf{H}) P^- (I_{15} - \mathbf{K}\mathbf{H})^T + \mathbf{K} R \mathbf{K}^T$$

The indirect error state is then injected into the nominal state:
$$\hat{\mathbf{p}}^+ = \hat{\mathbf{p}}^- + \delta\hat{\mathbf{p}}$$
$$\hat{\mathbf{v}}^+ = \hat{\mathbf{v}}^- + \delta\hat{\mathbf{v}}$$
$$\hat{\mathbf{q}}^+ = \hat{\mathbf{q}}^- \otimes \begin{bmatrix} 1 \\ \frac{1}{2}\delta\hat{\boldsymbol{\theta}} \end{bmatrix}$$
$$\hat{\mathbf{b}}_a^+ = \hat{\mathbf{b}}_a^- + \delta\hat{\mathbf{b}}_a, \quad \hat{\mathbf{b}}_g^+ = \hat{\mathbf{b}}_g^- + \delta\hat{\mathbf{b}}_g$$

---

## 6. Plain English Summary (Layer B)

| Concept | What It Does | Why It Prevents Disaster |
| :--- | :--- | :--- |
| **Cross-Track Update** | Pulls the car gently back toward the road centerline. | Stops position from wandering off into fields, rivers, or buildings. |
| **Road Heading Update** | Aligns the car's estimated compass heading with the direction the road is running. | Completely cures the unobserved gyroscope drift that ruined Phase 5 dead reckoning. |
| **Adaptive Noise $R / c^2$** | If confidence drops (e.g., at a tricky 5-way junction), the filter loosens its grip. | Prevents the system from violently snapping onto the wrong street if it isn't sure. |
| **Chi-Square Gating** | Rejects updates if the road is suddenly miles away from where physics says we are. | Prevents corrupted map data or map bugs from ripping the navigation solution apart. |
| **Joseph Stabilized Covariance** | Mathematically guarantees that uncertainty matrices never become negative or numerically ill-conditioned. | Ensures the filter runs continuously for hours without crashing or producing imaginary numbers. |
