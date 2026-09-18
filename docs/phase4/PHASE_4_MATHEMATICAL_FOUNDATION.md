# Phase 4 Mathematical Foundations: AI Motion Intelligence for Dead Reckoning

**Project**: Smart India Hackathon 2026 — Problem Statement SIH26168  
**Sponsoring Agency**: Indian Space Research Organisation (ISRO)  
**System**: AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation  
**Phase**: Phase 4 — AI Motion Intelligence  
**Document Code**: `SIH26168-MATH-PHASE4-01`  
**Classification**: RESEARCH DERIVATIONS & OPERATIONAL SPECIFICATION  

---

## Table of Contents
1. [Physical Motivation: Double Integration Failure vs Single Integration](#1-physical-motivation-double-integration-failure-vs-single-integration)
2. [Supervised Learning Formulation for Inertial Motion Inference](#2-supervised-learning-formulation-for-inertial-motion-inference)
3. [Regression Loss Functions & Robust Outlier Mitigation](#3-regression-loss-functions--robust-outlier-mitigation)
4. [Train-Only Z-Score Normalization & Leakage Elimination](#4-train-only-z-score-normalization--leakage-elimination)
5. [Causal Convolution & Receptive Field Mathematics](#5-causal-convolution--receptive-field-mathematics)
6. [Recurrent Neural Kinematics: Causal GRU and LSTM](#6-recurrent-neural-kinematics-causal-gru-and-lstm)
7. [Temporal Convolutional Networks (TCN) with Dilated Causal Filters](#7-temporal-convolutional-networks-tcn-with-dilated-causal-filters)
8. [Heteroscedastic Uncertainty & Gaussian Negative Log-Likelihood](#8-heteroscedastic-uncertainty--gaussian-negative-log-likelihood)
9. [Multitask Learning: Joint Speed Regression & Motion State Classification](#9-multitask-learning-joint-speed-regression--motion-state-classification)
10. [Downstream Dead-Reckoning Mechanization & Error Propagation](#10-downstream-dead-reckoning-mechanization--error-propagation)

---

## 1. Physical Motivation: Double Integration Failure vs Single Integration

### Technical Derivation
In classical inertial navigation (INS), position $\mathbf{p}(t) \in \mathbb{R}^3$ is recovered from specific force $\mathbf{f}^b(t)$ measured by an accelerometer by integrating Newton's second law:

$$\mathbf{p}(t) = \mathbf{p}(0) + \mathbf{v}(0)t + \iint_0^t \left( \mathbf{R}_b^n(\tau) \mathbf{f}^b(\tau) + \mathbf{g}^n \right) d\tau^2$$

In consumer MEMS sensors (such as smartphones), the accelerometer measurement contains a residual bias $\mathbf{b}_a(t)$, an orientation error in the estimated body-to-navigation rotation $\delta\boldsymbol{\theta}(t)$, and high-frequency noise $\mathbf{n}_a(t)$:

$$\tilde{\mathbf{f}}^b(t) = \mathbf{f}^b(t) + \mathbf{b}_a(t) + \mathbf{n}_a(t)$$

Under attitude tilt error $\delta\boldsymbol{\theta}(t)$, the estimated rotation matrix relates to the true matrix via $\tilde{\mathbf{R}}_b^n \approx (\mathbf{I} - [\delta\boldsymbol{\theta}\times]) \mathbf{R}_b^n$. Consequently, the apparent acceleration in the navigation frame is:

$$\tilde{\mathbf{a}}^n(t) = \tilde{\mathbf{R}}_b^n \tilde{\mathbf{f}}^b(t) + \mathbf{g}^n \approx \mathbf{a}^n(t) + \mathbf{R}_b^n \mathbf{b}_a(t) - [\delta\boldsymbol{\theta}(t)\times] \mathbf{g}^n + \mathbf{R}_b^n \mathbf{n}_a(t)$$

The acceleration error $\mathbf{a}_{\text{err}}(t) = \mathbf{R}_b^n \mathbf{b}_a(t) - [\delta\boldsymbol{\theta}(t)\times] \mathbf{g}^n$ produces position error growth:

$$\mathbf{e}_p(t) = \iint_0^t \mathbf{a}_{\text{err}}(\tau) \, d\tau^2 = \frac{1}{2} \mathbf{a}_{\text{err}} t^2 \quad \implies \quad \mathcal{O}(t^2)$$

Even a microscopic accelerometer bias of $0.05\text{ m/s}^2$ (or a $0.3^\circ$ tilt error causing gravity leakage of $g \sin(0.3^\circ) \approx 0.051\text{ m/s}^2$) results in:

$$\|\mathbf{e}_p(60\text{ s})\| = \frac{1}{2} (0.05)(60)^2 = 90.0\text{ meters}$$
$$\|\mathbf{e}_p(120\text{ s})\| = \frac{1}{2} (0.05)(120)^2 = 360.0\text{ meters}$$

In contrast, if an AI model directly predicts the forward speed $\hat{v}_{\text{fwd}}(t)$, position dead-reckoning reduces to a **single integration**:

$$\mathbf{p}(t) = \mathbf{p}(0) + \int_0^t \begin{bmatrix} \sin\psi(\tau) \\ \cos\psi(\tau) \end{bmatrix} \hat{v}_{\text{fwd}}(\tau) \, d\tau$$

For a mean speed estimation error $\delta v$:

$$\|\mathbf{e}_p(t)\| \le \int_0^t |\delta v| \, d\tau = |\delta v| \cdot t \quad \implies \quad \mathcal{O}(t)$$

### Simple Explanation
Imagine walking blindfolded. If someone tells you your *acceleration* (how quickly you are speeding up or slowing down), and you try to guess where you are, any tiny mistake in your guess gets multiplied by time squared! A tiny mistake of one centimeter per second squared turns into being hundreds of meters off after just one minute. That is double integration: error explodes like an accelerating rocket.

Instead, if a smart friend directly tells you your exact *speed* (e.g., "you are walking at 1 meter per second"), you only add up speed over time. If your friend is off by a tiny bit, your position mistake only grows steadily in a straight line, not an exploding curve.

### Why This Matters for Our Project
The ISRO problem statement sets a strict target of $< 10\%$ positional drift during GNSS blackouts. Phase 1 showed that raw double integration produces $33\%$ to $60\%$ drift, completely blowing past the limit. By switching to AI-inferred forward speed, we drop error growth from quadratic $\mathcal{O}(t^2)$ to linear $\mathcal{O}(t)$, making the $< 10\%$ target mathematically possible.

---

## 2. Supervised Learning Formulation for Inertial Motion Inference

### Technical Derivation
Let the input space be the set of fixed-length temporal sliding windows of preconditioned inertial measurements:

$$\mathcal{X} = \mathbb{R}^{T \times C}, \quad T = 30\text{ timesteps}, \quad C = 12\text{ feature channels}$$

At $10\text{ Hz}$ sampling frequency, $T = 30$ corresponds to a temporal receptive horizon of $3.0\text{ seconds}$. Each window $\mathbf{X}_k \in \mathcal{X}$ contains the history ending strictly at time $t_k$:

$$\mathbf{X}_k = \begin{bmatrix} \mathbf{x}_{k-T+1}^T \\ \mathbf{x}_{k-T+2}^T \\ \vdots \\ \mathbf{x}_k^T \end{bmatrix} \in \mathbb{R}^{30 \times 12}$$

where $\mathbf{x}_\tau \in \mathbb{R}^{12}$ is the feature vector at instant $\tau$.

The supervised target is the true vehicle longitudinal forward speed $y_k = v_{\text{fwd}}(t_k) \in \mathbb{R}_{\ge 0}$ measured from the reference vehicle instrumentation at the contemporaneous final timestep $t_k$.

The objective is to learn a parameter vector $\boldsymbol{\theta}^* \in \Theta$ of a causal neural hypothesis $f_{\boldsymbol{\theta}}: \mathbb{R}^{T \times C} \to \mathbb{R}$ minimizing empirical risk over a dataset $\mathcal{D}_{\text{train}} = \{(\mathbf{X}_i, y_i)\}_{i=1}^N$:

$$\boldsymbol{\theta}^* = \arg\min_{\boldsymbol{\theta}} \mathcal{R}_{\text{emp}}(\boldsymbol{\theta}) = \arg\min_{\boldsymbol{\theta}} \frac{1}{N} \sum_{i=1}^N \mathcal{L}(f_{\boldsymbol{\theta}}(\mathbf{X}_i), y_i) + \lambda \Omega(\boldsymbol{\theta})$$

where $\mathcal{L}(\cdot, \cdot)$ is the regression loss and $\Omega(\boldsymbol{\theta}) = \|\boldsymbol{\theta}\|_2^2$ is weight decay regularization.

### Simple Explanation
We give our neural network a small movie strip of the last 3 seconds of sensor readings from the phone (30 frames, each with 12 numbers like forward acceleration, roll, pitch, vibration). The network's job is to look only at those past 3 seconds and answer one question: "How fast is the vehicle moving forward right now?" We train it by comparing its guess to the speedometer of the car and tuning its internal settings until its mistakes become as small as possible.

### Why This Matters for Our Project
The neural network learns physical relationships that cannot be captured by simple formulas—such as how engine vibrations, road chatter, and chassis pitch tilt naturally correlate with vehicle speed. Because the window only uses past samples, the model works continuously in real time.

---

## 3. Regression Loss Functions & Robust Outlier Mitigation

### Technical Derivation

#### Mean Squared Error (MSE / $L_2$)
$$\mathcal{L}_{\text{MSE}}(\hat{v}, v) = (\hat{v} - v)^2$$
The gradient with respect to the prediction is:
$$\frac{\partial \mathcal{L}_{\text{MSE}}}{\partial \hat{v}} = 2(\hat{v} - v)$$
The gradient is proportional to the residual error $e = \hat{v} - v$. If a pothole or severe curb strike produces an abnormal acceleration spike, $e$ can become very large, causing huge gradients ($\propto e$) that destabilize the weights.

#### Mean Absolute Error (MAE / $L_1$)
$$\mathcal{L}_{\text{MAE}}(\hat{v}, v) = |\hat{v} - v|$$
The subgradient is:
$$\frac{\partial \mathcal{L}_{\text{MAE}}}{\partial \hat{v}} = \text{sign}(\hat{v} - v) \quad (e \ne 0)$$
The gradient has a constant magnitude $\pm 1$, providing robust resistance to extreme outliers, but suffers from non-smoothness at $e = 0$ and lacks curvature to guide fine convergence when error is small.

#### Huber Loss (Smooth $L_1$)
To balance fast convergence for small errors with robustness against pothole spikes, we employ Huber loss with transition threshold $\delta > 0$:

$$\mathcal{L}_{\text{Huber}}(e; \delta) = \begin{cases} \frac{1}{2} e^2 & \text{if } |e| \le \delta \\ \delta \left(|e| - \frac{1}{2} \delta\right) & \text{if } |e| > \delta \end{cases}$$

The derivative is piecewise continuous:

$$\frac{\partial \mathcal{L}_{\text{Huber}}}{\partial e} = \begin{cases} e & \text{if } |e| \le \delta \\ \delta \cdot \text{sign}(e) & \text{if } |e| > \delta \end{cases}$$

For errors within $\pm\delta$, it behaves like smooth MSE ($L_2$). For errors exceeding $\delta$ (such as during dynamic bumps and road shock), it transitions to linear penalty ($L_1$), capping the maximum gradient at $\delta$.

```
Loss
 ^
 |             / MSE (quadratic growth)
 |            /
 |           /-- Huber (linear growth for |e| > δ)
 |          /
 |   \     /
 |    \---/  Huber (|e| <= δ is parabolic)
 +-------------------------> Residual Error e
```

### Simple Explanation
If the AI makes a tiny mistake (like being off by 0.1 km/h), we want it to gently nudge its settings (quadratic loss). But if the car hits a huge pothole and the phone shakes violently, the error might suddenly look huge for a fraction of a second. If we used standard squared error, the AI would panic and drastically change its settings, ruining what it learned. Huber loss acts like a shock absorber: if a mistake is huge, it caps the punishment so one pothole doesn't ruin the model.

### Why This Matters for Our Project
Real driving datasets contain sudden road bumps, speed breakers, and expansion joints. Using Huber loss protects model training from getting corrupted by transient shock spikes while preserving sharp precision during steady highway cruising and gentle urban turns.

---

## 4. Train-Only Z-Score Normalization & Leakage Elimination

### Technical Derivation
Let the training partition feature matrix be $\mathbf{X}_{\text{train}} \in \mathbb{R}^{N_{\text{train}} \times T \times C}$. For each feature channel $c \in \{1, \dots, C\}$, empirical mean $\mu_c$ and standard deviation $\sigma_c$ are computed **strictly over training windows**:

$$\mu_c = \frac{1}{N_{\text{train}} \cdot T} \sum_{i=1}^{N_{\text{train}}} \sum_{t=1}^T X_{i, t, c}$$

$$\sigma_c = \sqrt{\frac{1}{N_{\text{train}} \cdot T} \sum_{i=1}^{N_{\text{train}}} \sum_{t=1}^T (X_{i, t, c} - \mu_c)^2 + \epsilon}$$

where $\epsilon = 10^{-8}$ prevents division by zero for invariant channels (e.g. standstill indicators).

The normalized feature is:

$$\tilde{X}_{i, t, c} = \frac{X_{i, t, c} - \mu_c}{\sigma_c}$$

### Temporal Purge Gap Formulation
Let window $i$ start at sample $s_i$ and end at $e_i = s_i + T - 1$. If training data ends at sample index $K_{\text{train}}$, any validation window starting at $s_{\text{val}} \le K_{\text{train}} + T - 1$ contains physical raw samples that were seen during training!

To guarantee mathematical independence between training, validation, and testing, we define a **Purge Gap** of $G$ timesteps:

$$G \ge T + S_{\text{buffer}} = 30 + 20 = 50\text{ samples} \quad (5.0\text{ seconds})$$

$$\text{Train samples}: [0, \, K_1]$$
$$\text{Purge Buffer 1}: (K_1, \, K_1 + G) \quad \text{[DISCARDED FROM ALL SETS]}$$
$$\text{Val samples}: [K_1 + G, \, K_2]$$
$$\text{Purge Buffer 2}: (K_2, \, K_2 + G) \quad \text{[DISCARDED FROM ALL SETS]}$$
$$\text{Test samples}: [K_2 + G, \, N_{\text{total}} - 1]$$

### Simple Explanation
If you are studying for an exam, you cannot look at the answer sheet of the test beforehand—that is cheating (data leakage). Because our 3-second windows slide forward every half second, consecutive windows overlap and share the same raw sensor data. If we just randomly shuffled windows, almost identical copies of training data would end up on the test. 

To prevent this, we divide the drive by time: the first part is for learning, the second for tuning, and the last part for testing. We also throw away a 5-second dead-zone buffer between each part so no window can accidentally peek across the border.

### Why This Matters for Our Project
Previous un-audited models suffered from subtle temporal leakage, making accuracy appear artificially high during training but failing in real-world deployment. The temporal block split with a 50-sample purge gap guarantees that test evaluations represent genuine predictive capability on unseen driving segments.

---

## 5. Causal Convolution & Receptive Field Mathematics

### Technical Derivation
A 1D convolutional layer with kernel size $K$, weights $\mathbf{W} \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times K}$, and bias $\mathbf{b} \in \mathbb{R}^{C_{\text{out}}}$ operating on sequence $\mathbf{x} \in \mathbb{R}^{C_{\text{in}} \times T}$ is defined as:

$$\mathbf{y}_t = \mathbf{b} + \sum_{k=0}^{K-1} \mathbf{W}_k \mathbf{x}_{t - k}$$

In standard non-causal convolutions, the filter centers at $t$, accessing inputs from $t - \lfloor K/2 \rfloor$ to $t + \lfloor K/2 \rfloor$, which requires knowledge of the **future** ($t > 0$).

In a **causal convolution**, the output at timestep $t$ depends strictly on inputs at or before $t$:

$$\mathbf{y}_t = \mathbf{b} + \sum_{k=0}^{K-1} \mathbf{W}_k \mathbf{x}_{t - k}$$

This is enforced by applying left padding of size $P_{\text{causal}} = K - 1$ (with zeros) and zero right padding:

$$\text{Input: } [\underbrace{0, \dots, 0}_{K-1 \text{ zeros}}, \, x_1, \, x_2, \, \dots, \, x_T]$$

```
Causal Conv (Kernel=3, Left Pad=2):
Pad  Pad   x[0]   x[1]   x[2]   x[3]   ...   x[t]
 \    |    /      |      |
  \   |   /       |      |
   v  v  v        |      |
    y[0]         y[1]   y[2]                 y[t] (depends only on <= t)
```

The receptive field $R_l$ of layer $l$ with kernel size $K_l$ and dilation $d_l$ expands according to:

$$R_l = R_{l-1} + (K_l - 1) \cdot d_l, \quad R_0 = 1$$

### Simple Explanation
Standard image filters look to the left and to the right of a pixel. But in real life, time only moves forward! You cannot look at the sensor data of tomorrow to guess your speed today. A causal filter puts on blinders so it can only look backwards into the past.

### Why This Matters for Our Project
Any model that accidentally uses even a single future sample cannot be deployed on a smartphone in real time because the future hasn't happened yet. Strict causality ensures the model runs sample-by-sample as data arrives from the sensors.

---

## 6. Recurrent Neural Kinematics: Causal GRU and LSTM

### Technical Derivation

#### Gated Recurrent Unit (GRU)
For input $\mathbf{x}_t \in \mathbb{R}^{12}$ and previous hidden state $\mathbf{h}_{t-1} \in \mathbb{R}^{H}$:

$$\mathbf{z}_t = \sigma(\mathbf{W}_z \mathbf{x}_t + \mathbf{U}_z \mathbf{h}_{t-1} + \mathbf{b}_z) \quad \text{(Update gate)}$$
$$\mathbf{r}_t = \sigma(\mathbf{W}_r \mathbf{x}_t + \mathbf{U}_r \mathbf{h}_{t-1} + \mathbf{b}_r) \quad \text{(Reset gate)}$$
$$\tilde{\mathbf{h}}_t = \tanh(\mathbf{W}_h \mathbf{x}_t + \mathbf{U}_h (\mathbf{r}_t \odot \mathbf{h}_{t-1}) + \mathbf{b}_h) \quad \text{(Candidate state)}$$
$$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t \quad \text{(Updated hidden state)}$$

The GRU preserves vehicle dynamic memory across time steps without vanishing gradients. Since the recurrence operates forward in time ($t = 1 \to 30$), the final state $\mathbf{h}_{30}$ synthesizes the entire 3-second causal history into an $H$-dimensional summary vector:

$$\hat{v}_{\text{fwd}} = \text{Linear}(\mathbf{h}_{30}) = \mathbf{W}_o \mathbf{h}_{30} + b_o$$

```
x_1          x_2                 x_30
 │            │                   │
 ▼            ▼                   ▼
GRU_1  ──►  GRU_2  ──► ... ──►  GRU_30
                                  │
                                  ▼
                             h_30 (Hidden State)
                                  │
                                  ▼
                             Linear Head ──► Speed v_fwd
```

### Simple Explanation
A GRU is like an intelligent memory bank. As each new sensor reading arrives every 0.1 seconds, the memory bank asks two questions: "Should I forget old vibrations that don't matter anymore?" and "Should I update my memory with this new acceleration?" After 30 steps, the memory bank gives its final summary to a small calculator that estimates speed.

### Why This Matters for Our Project
Vehicles have physical inertia—they do not jump from 0 to 60 km/h instantly. Recurrent architectures naturally remember whether the vehicle was accelerating or coasting over the past couple of seconds, smoothing out jittery noise.

---

## 7. Temporal Convolutional Networks (TCN) with Dilated Causal Filters

### Technical Derivation
A TCN uses dilated causal convolutions where the filter taps skip $d - 1$ values:

$$\mathbf{y}_t = \sum_{k=0}^{K-1} \mathbf{W}_k \mathbf{x}_{t - d \cdot k}$$

By exponentially increasing dilation across layers ($d = 1, 2, 4, 8$), the receptive field grows exponentially while the parameter count grows only linearly:

$$R = 1 + \sum_{l=0}^{L-1} (K - 1) \cdot 2^l = 1 + (K - 1)(2^L - 1)$$

For kernel size $K = 3$ and $L = 4$ layers:
$$R = 1 + (3 - 1)(2^4 - 1) = 1 + 2(15) = 31\text{ timesteps}$$

Since $R = 31 \ge 30$, every single one of the 30 input timesteps contributes to the final prediction!

```
Layer 3 (d=4):  o-------o-------o-------o  (Receptive field covers 31 steps)
                |       |       |       |
Layer 2 (d=2):  o---o---o---o---o---o---o
                |   |   |   |   |   |   |
Layer 1 (d=1):  o-o-o-o-o-o-o-o-o-o-o-o-o
                | | | | | | | | | | | | |
Input (t=1..30):x x x x x x x x x x x x x
```

### Residual Connection with $1 \times 1$ Conv
To stabilize deep gradient propagation:

$$\mathbf{z}^{[l]} = \text{Activation}(\mathcal{F}(\mathbf{z}^{[l-1]})) + \text{Conv}_{1\times1}(\mathbf{z}^{[l-1]})$$

### Simple Explanation
Imagine trying to look at a 3-second timeline. A normal filter can only see a tiny slice at a time. A dilated network acts like a telescope: the bottom layer looks closely at every single tick, while higher layers step back and look at wider patterns (skipping steps like $1, 2, 4, 8$) without needing extra math. This lets the network understand the big picture of the vehicle's motion in very few calculations.

### Why This Matters for Our Project
TCNs compute all time steps in parallel during training (unlike RNNs which must step sequentially), and during inference they run with fixed, ultra-low latency. They offer the memory of an LSTM with the speed and simplicity of a CNN.

---

## 8. Heteroscedastic Uncertainty & Gaussian Negative Log-Likelihood

### Technical Derivation
Standard neural networks output only a single point estimate $\hat{v}$, with no indication of whether the prediction is reliable. Under varying road conditions (smooth highway vs rough cobblestone), sensor noise variance is **heteroscedastic** (input-dependent).

We design the neural network to output two values:
1. Predictive mean $\mu(\mathbf{X}) \in \mathbb{R}$
2. Predictive log-variance $s(\mathbf{X}) = \log \sigma^2(\mathbf{X}) \in \mathbb{R}$

Predicting log-variance ensures that $\sigma^2 = \exp(s) > 0$ is strictly positive for all inputs.

Assuming a Gaussian conditional distribution $p(y \mid \mathbf{X}) = \mathcal{N}(y; \mu(\mathbf{X}), \sigma^2(\mathbf{X}))$, the negative log-likelihood (NLL) loss is:

$$\mathcal{L}_{\text{NLL}}(y, \mu, s) = -\log p(y \mid \mathbf{X}) = \frac{1}{2} \log(2\pi) + \frac{1}{2} s + \frac{(y - \mu)^2}{2 \exp(s)}$$

Omitting constant $\frac{1}{2}\log(2\pi)$:

$$\mathcal{L}_{\text{NLL}} = \frac{1}{2} \left[ s + \exp(-s)(y - \mu)^2 \right]$$

The gradients reveal an intuitive self-balancing mechanism:

$$\frac{\partial \mathcal{L}_{\text{NLL}}}{\partial \mu} = -\exp(-s)(y - \mu) = -\frac{y - \mu}{\sigma^2}$$

$$\frac{\partial \mathcal{L}_{\text{NLL}}}{\partial s} = \frac{1}{2} \left[ 1 - \frac{(y - \mu)^2}{\sigma^2} \right]$$

- If the model is confident ($\sigma^2$ is small), any residual error $(y - \mu)^2$ receives a large penalty.
- If the model encounters an irregular, disturbed road section with large residual errors, it increases $\sigma^2$ to attenuate the penalty, paying a controlled cost of $\frac{1}{2}s = \log \sigma$.

### Simple Explanation
A good navigator not only gives an answer but also says how sure they are. When the car is cruising smoothly on a straight road, the AI says: "We are going 50 km/h, and I am 99% sure (low uncertainty)." But when the car hits bumpy railroad tracks and shakes violently, the AI says: "I think we are going 45 km/h, but my uncertainty is high!" The loss function teaches the AI to honestly admit when road vibration makes guessing difficult.

### Why This Matters for Our Project
In Phase 5, the Error-State Kalman Filter (ESKF) will fuse this speed estimate with inertial propagation. The filter's Kalman gain depends directly on measurement variance: $K = P H^T (H P H^T + R)^{-1}$, where $R = \sigma^2$. When the AI reports high uncertainty, the ESKF automatically downweights the measurement, preventing noisy road bumps from corrupting the navigation state.

---

## 9. Multitask Learning: Joint Speed Regression & Motion State Classification

### Technical Derivation
Vehicle motion exhibits distinct discrete kinematic regimes:
1. **Standstill (ZUPT)**: $v = 0$
2. **Cruising**: $a_{\text{fwd}} \approx 0, v > 0$
3. **Accelerating**: $a_{\text{fwd}} > 0.4\text{ m/s}^2$
4. **Braking**: $a_{\text{fwd}} < -0.4\text{ m/s}^2$
5. **Turning**: $|\omega_{\text{yaw}}| > 0.1\text{ rad/s}$

We formulate a shared feature representation $\mathbf{z} = g_{\boldsymbol{\theta}_{\text{shared}}}(\mathbf{X})$ with dual heads:
- **Regression Head**: $\hat{v} = h_{\text{reg}}(\mathbf{z}) \in \mathbb{R}$
- **Classification Head**: $\hat{\mathbf{p}}_{\text{state}} = \text{Softmax}(h_{\text{cls}}(\mathbf{z})) \in \Delta^4$

The joint multi-task loss is:

$$\mathcal{L}_{\text{multi}} = \mathcal{L}_{\text{Huber}}(\hat{v}, v) + \lambda_{\text{cls}} \mathcal{L}_{\text{CE}}(\hat{\mathbf{p}}_{\text{state}}, y_{\text{state}})$$

where $\mathcal{L}_{\text{CE}}$ is Cross-Entropy loss and $\lambda_{\text{cls}} = 0.2$ balances gradient magnitudes.

```
                  Input Window (30 x 12)
                            │
                            ▼
              Shared Causal Feature Backbone
               (Conv1D / GRU / TCN Layers)
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
       Regression Head            Classification Head
              │                           │
              ▼                           ▼
    Forward Speed v_fwd          Motion State Probabilities
      (Scalar m/s)              (Standstill, Cruising, etc.)
```

### Simple Explanation
When a human learns to drive, they don't just memorize numbers on a speedometer—they also understand what the car is doing (stopping at a red light, speeding up, or taking a sharp turn). By training the AI to recognize the *action* at the same time as the *speed*, both tasks help each other. When the network knows the car is stopped, it easily predicts zero speed.

### Why This Matters for Our Project
This provides Phase 5 and Phase 6 with an immediate classification of vehicle state, enabling instant Zero-Velocity Updates (ZUPT) when stopped and Non-Holonomic Constraint (NHC) adjustments during sharp turns.

---

## 10. Downstream Dead-Reckoning Mechanization & Error Propagation

### Technical Derivation
To test whether AI speed actually improves dead-reckoning without building the full Phase 5 ESKF, we implement forward 2D trajectory mechanization using the AI speed $\hat{v}_k$ combined with the smartphone's existing calibrated heading $\psi_k$:

$$\Delta p_{e, k} = \hat{v}_k \sin(\psi_k) \Delta t_k$$
$$\Delta p_{n, k} = \hat{v}_k \cos(\psi_k) \Delta t_k$$

$$p_{e, k} = p_{e, k-1} + \Delta p_{e, k}, \quad p_{n, k} = p_{n, k-1} + \Delta p_{n, k}$$

Over an outage window of duration $T_{\text{outage}} = t_M - t_0$, the endpoint error $E_p$ and percentage drift $D$ are:

$$E_p = \sqrt{(p_{e, M} - p_{e, M}^{\text{ref}})^2 + (p_{n, M} - p_{n, M}^{\text{ref}})^2}$$

$$D = \frac{E_p}{d_{\text{traveled}}} \times 100\%, \quad d_{\text{traveled}} = \sum_{k=1}^M \sqrt{(\Delta p_{e, k}^{\text{ref}})^2 + (\Delta p_{n, k}^{\text{ref}})^2}$$

### Error Sensitivity to Heading Error
If the heading has an error $\delta\psi$, the projected velocity error is:

$$\mathbf{v}_{\text{err}} = \hat{v} \begin{bmatrix} \sin(\psi + \delta\psi) - \sin\psi \\ \cos(\psi + \delta\psi) - \cos\psi \end{bmatrix} \approx \hat{v} \delta\psi \begin{bmatrix} \cos\psi \\ -\sin\psi \end{bmatrix}$$

$$\|\mathbf{v}_{\text{err}}\| \approx \hat{v} |\delta\psi|$$

Even with a perfect speed estimate ($\hat{v} = v$), a heading error of $\delta\psi = 5^\circ \approx 0.087\text{ rad}$ at $v = 15\text{ m/s}$ ($54\text{ km/h}$) generates an instantaneous lateral drift rate of:

$$\|\mathbf{v}_{\text{err}}\| = (15)(0.087) = 1.31\text{ m/s}$$
$$\|\mathbf{e}_p(60\text{ s})\| \approx 1.31 \times 60 = 78.6\text{ meters}$$

### Simple Explanation
Even if you know your walking speed down to the millimeter, if your compass points slightly in the wrong direction, you will gradually veer off course. Dead reckoning needs two things: how fast you are moving (speed) and where you are pointing (heading). In Phase 4, we evaluate speed models by checking if replacing noisy double-integrated acceleration with AI speed reduces positional drift.

### Why This Matters for Our Project
This highlights the vital boundary: Phase 4 provides **motion intelligence (speed)**, while Phase 5 will provide **optimal state fusion (attitude + position)**. We never confuse speed prediction with full trajectory dead reckoning.
