# Phase 6 Robust Stationary Detection & Hysteresis

**Project**: Smart India Hackathon 2026 — Problem Statement SIH26168  
**Sponsoring Agency**: Indian Space Research Organisation (ISRO)  
**System**: AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation  
**Document**: `docs/phase6/PHASE_6_STATIONARY_DETECTION.md`  

---

## 1. Technical Derivation

### 1.1 Multi-Signal Stationary Fusion
Relying on a single sensor threshold (e.g., speed alone or acceleration variance alone) leads to false stationary triggers when coasting smoothly or false moving triggers when the engine is idling with vibration. We deploy a multi-criteria decision function fusing 4 independent causal channels:

1. **Specific Force Norm Consistency**:
   $$C_g = \left| \|\mathbf{f}_b(t)\| - g \right| < \tau_g \quad (\tau_g = 0.40\text{ m/s}^2)$$
2. **Accelerometer Variance over Rolling Window ($W=10$ samples)**:
   $$C_a = \text{Var}_{10}(\|\mathbf{f}_b\|) < \tau_a \quad (\tau_a = 0.04\text{ (m/s}^2)^2)$$
3. **Gyroscope Angular Rate Norm**:
   $$C_\omega = \|\boldsymbol{\omega}_b(t)\| < \tau_\omega \quad (\tau_\omega = 0.045\text{ rad/s} \approx 2.58^\circ/\text{s})$$
4. **Phase 4 AI Predicted Speed**:
   $$C_v = \hat{v}_{\text{fwd}}(t) < \tau_v \quad (\tau_v = 0.50\text{ m/s})$$

Instantaneous Candidate Flag:
$$S_{\text{cand}}(t) = C_g \land C_a \land C_\omega \land C_v$$

### 1.2 Dual-Threshold Hysteresis State Machine
To avoid rapid bouncing between stationary and moving states at traffic lights:
- **Enter Standstill ($M \to S$)**: Must satisfy $S_{\text{cand}}(t) == \text{True}$ continuously for $N_{\text{enter}} \ge 5$ samples ($0.50\text{ s}$).
- **Exit Standstill ($S \to M$)**: If $\neg S_{\text{cand}}(t)$ holds for $N_{\text{exit}} \ge 2$ consecutive samples ($0.20\text{ s}$), immediately exit to moving state.

---

## 2. Simple Explanation
Think of how your smartphone screen decides whether to rotate or turn off. If it reacted to every tiny twitch, the screen would flicker constantly.

Our stationary detector uses a "calmness test." It asks:
1. Is the phone feeling only gravity and no acceleration?
2. Has the shaking stopped?
3. Has the spinning stopped?
4. Does the AI speed model agree that the car isn't rolling?

If all four agree for at least half a second, the filter declares a confirmed stop. If the car starts accelerating forward for even a fifth of a second, the filter instantly switches back to driving mode!

---

## 3. Why This Matters for SIH26168
False stationary detection during slow crawling traffic would lock velocity to zero while the car is actually moving, causing severe trajectory errors. Hysteresis ensures 100% reliable detection during genuine stops while preserving smooth motion tracking during stop-and-go driving.
