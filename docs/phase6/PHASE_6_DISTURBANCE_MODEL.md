# Phase 6 Disturbance-Aware Adaptive Weighting Specification

**Project**: Smart India Hackathon 2026 — Problem Statement SIH26168  
**Sponsoring Agency**: Indian Space Research Organisation (ISRO)  
**System**: AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation  
**Document**: `docs/phase6/PHASE_6_DISTURBANCE_MODEL.md`  

---

## 1. Technical Derivation

### 1.1 Dynamics vs Virtual Constraint Conflict
Non-Holonomic Constraints assume zero lateral velocity. However, in physical vehicle dynamics:
1. **Cornering Maneuvers**: Centripetal acceleration $a_y = v \dot{\psi}$ produces lateral tire deformation and slip angles ($\alpha \ne 0$). Enforcing an overly rigid constraint causes the filter to misinterpret cornering forces as orientation errors.
2. **Road Irregularities & Potholes**: Vertical accelerations from road bumps produce high-frequency specific force spikes, causing non-zero vertical velocity transients.
3. **Sensor Noise & Vibration**: Engine idling or high-speed chassis vibration elevates high-frequency noise.

### 1.2 Causal Disturbance Indicators
We define two causal disturbance scores normalized to $[0, \infty)$:

1. **Lateral Dynamics Disturbance ($D_{\text{lat}}$)**:
   $$D_{\text{lat}}(t) = \max\left(0, \frac{|a_{\text{lat}}(t)|}{1.5} + \frac{|\omega_{\text{yaw}}(t)|}{0.15} - 1.0\right)$$
   where $1.5\text{ m/s}^2$ represents the onset of active cornering and $0.15\text{ rad/s} \approx 8.6^\circ/\text{s}$ represents active turning rate.

2. **Vertical Shock / Pothole Disturbance ($D_{\text{up}}$)**:
   $$D_{\text{up}}(t) = \max\left(0, \frac{|a_{\text{up}}(t) - g|}{2.0} + \frac{\text{VibrationEnergy}(t)}{1.0} - 1.0\right)$$
   where $2.0\text{ m/s}^2$ represents significant vertical road displacement.

### 1.3 Adaptive Covariance Inflation
Nominal NHC measurement standard deviations are set to baseline values:
$$\sigma_{\text{lat}, 0} = 0.20\text{ m/s}$$
$$\sigma_{\text{up}, 0} = 0.20\text{ m/s}$$

During disturbances, the effective standard deviations scale dynamically:
$$\sigma_{\text{lat}}(t) = \sigma_{\text{lat}, 0} \cdot \left(1.0 + 3.0 \cdot D_{\text{lat}}(t)\right)$$
$$\sigma_{\text{up}}(t) = \sigma_{\text{up}, 0} \cdot \left(1.0 + 3.0 \cdot D_{\text{up}}(t)\right)$$

The resulting measurement covariance is:
$$\mathbf{R}_{NHC}(t) = \begin{bmatrix} \sigma_{\text{lat}}^2(t) & 0 \\ 0 & \sigma_{\text{up}}^2(t) \end{bmatrix}$$

---

## 2. Simple Explanation
Imagine walking with a friend who gently holds your arm to keep you walking in a straight line. If you are walking down a quiet hallway, the gentle guidance helps you walk straight.

However, if you suddenly need to step aside to dodge an obstacle or jump over a puddle, your friend shouldn't yank your arm hard to force you back into the puddle! They should loosen their grip, let you step around the puddle, and then gently guide you again once you are past it.

Our disturbance detector loosens the filter's grip whenever the car is cornering or hitting a bump, keeping the vehicle stable and natural.

---

## 3. Why This Matters for SIH26168
Without adaptive disturbance weighting, rigid NHC fails during sharp turns, triggering high innovation residuals that corrupt the attitude quaternion. Disturbance-aware weighting ensures that the constraints remain beneficial across both smooth cruising and aggressive turning maneuvers.
