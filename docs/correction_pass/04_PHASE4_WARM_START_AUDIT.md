# Pre-Phase-8 Audit: Phase 4 Warm-Start vs. Cold-Start Temporal Dynamics

**Project**: Smart India Hackathon 2026 — Problem Statement SIH26168  
**Sponsoring Agency**: Indian Space Research Organisation (ISRO)  
**System**: AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation  
**Document**: `docs/correction_pass/04_PHASE4_WARM_START_AUDIT.md`  

---

## 1. Executive Summary & Problem Discovery

The Phase 4 AI Motion Intelligence engine (`CausalStreamingInferenceEngine`) operates over a temporal sliding window of length $W = 30$ samples ($3.0\text{ seconds}$ of context at $10\text{ Hz}$).

In all prior benchmark evaluation scripts across Phases 5, 6, and 7, the benchmarking loop invoked:
```python
ai_engine.reset()
```
immediately before simulating each blackout slice.

### The Mechanism of Failure:
- Calling `reset()` fills the $30 \times 12$ FIFO buffer with zeros.
- When the blackout begins at $t_0 = 4,600.0\text{ s}$, the vehicle is in active cruising motion at $11.01\text{ m/s}$ ($39.6\text{ km/h}$).
- At step 1 ($t = 4,600.1\text{ s}$), the buffer contains **29 rows of zeros** and only **1 row of actual motion data**.
- Consequently, Model F predicts a forward speed of **$2.38\text{ m/s}$** instead of $11.01\text{ m/s}$—an instantaneous speed deficit of **$8.63\text{ m/s}$ ($78\%$ error)**!
- It requires 30 time steps ($3.0\text{ seconds}$) for the zero padding to wash out of the rolling buffer. During these initial 3 seconds, the model integrates a severe longitudinal velocity deficit that permanently corrupts downstream dead-reckoning position.

---

## 2. Empirical Verification: Cold Start vs. Warm Start at Outage Entry

To prove this mechanism quantitatively, we evaluated the identical physical interval ($t_0 = 4,600.0\text{ s}$) under two conditions:
1. **Cold Start**: Buffer cleared with zeros at $t_0$.
2. **Warm Start**: Buffer pre-warmed using causal historical samples strictly preceding the outage ($t \in [t_0 - 3.0\text{ s}, t_0)$).

### Step-by-Step Telemetry Comparison (First 10 Samples of Blackout):
| Outage Step ($k$) | Time ($t$) | CAN Ground Truth Speed | Cold-Start Prediction | Cold-Start Error | Warm-Start Prediction | Warm-Start Error |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 4600.1 s | 11.006 m/s | **2.380 m/s** | **-8.626 m/s** | **11.101 m/s** | **+0.095 m/s** |
| 1 | 4600.2 s | 11.006 m/s | **2.985 m/s** | **-8.021 m/s** | **10.727 m/s** | **-0.279 m/s** |
| 2 | 4600.3 s | 10.994 m/s | **3.078 m/s** | **-7.916 m/s** | **10.934 m/s** | **-0.060 m/s** |
| 3 | 4600.4 s | 10.983 m/s | **4.222 m/s** | **-6.761 m/s** | **10.925 m/s** | **-0.058 m/s** |
| 4 | 4600.5 s | 10.956 m/s | **3.305 m/s** | **-7.651 m/s** | **10.187 m/s** | **-0.769 m/s** |
| 5 | 4600.6 s | 10.961 m/s | **4.207 m/s** | **-6.754 m/s** | **10.430 m/s** | **-0.531 m/s** |
| 6 | 4600.7 s | 10.958 m/s | **4.593 m/s** | **-6.365 m/s** | **10.344 m/s** | **-0.614 m/s** |
| 7 | 4600.8 s | 10.967 m/s | **4.984 m/s** | **-5.983 m/s** | **10.438 m/s** | **-0.529 m/s** |
| 8 | 4600.9 s | 10.939 m/s | **5.229 m/s** | **-5.710 m/s** | **10.682 m/s** | **-0.257 m/s** |
| 9 | 4601.0 s | 10.922 m/s | **5.243 m/s** | **-5.679 m/s** | **9.766 m/s** | **-1.156 m/s** |

### Key Findings:
1. **Instantaneous Entry Accuracy**: Under warm-start, entry speed error is **$0.095\text{ m/s}$ ($<1\%$)**, confirming that Model F is highly accurate when provided with valid temporal context.
2. **Cumulative Integrated Distance Deficit**: Over the first 3.0 seconds, the cold-start deficit integrates to:
   $$\Delta d_{\text{cold}} = \int_0^{3.0} \left(v_{\text{CAN}}(t) - \hat{v}_{\text{cold}}(t)\right) dt \approx 20.4\text{ m}$$
   This explains why Phase 5 and Phase 6 benchmarks reported an immediate $38 - 53\text{ m}$ position error in a short 10-second blackout!

---

## 3. Operational Realism vs. Artificial Benchmarking

In a real vehicle navigation system:
- The smartphone navigation app runs continuously while GNSS is available.
- When the car enters a tunnel or underpass, the IMU rolling buffer is **already full and warm**.
- Resetting the buffer to zero at the instant of GNSS loss is non-physical and misrepresents operational performance.

---

## 4. Formal Benchmark Protocol Separation

To establish scientific rigor, all subsequent evaluations will enforce two distinct, clearly labeled protocols:

### Benchmark Mode A: `WARM_START` (Primary Operational Benchmark)
- **Definition**: The AI streaming engine is pre-buffered using $W=30$ causal historical IMU frames from $t \in [t_0 - 30 \Delta t, t_0)$.
- **Causality Guarantee**: All pre-buffering data strictly satisfies $t < t_0$. Zero data from $t \ge t_0$ is used in warming.
- **Purpose**: Evaluates true operational dead reckoning during sudden GNSS outage.

### Benchmark Mode B: `COLD_START` (Secondary Stress Benchmark)
- **Definition**: The AI streaming engine starts with an empty zero-padded buffer at $t_0$.
- **Purpose**: Evaluates worst-case system resilience during a cold app launch inside a tunnel with zero prior history.

These two modes will never be mixed or averaged together without explicit labeling.
