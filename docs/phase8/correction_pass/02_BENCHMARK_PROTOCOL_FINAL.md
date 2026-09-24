# Phase 8 Authoritative Benchmark Protocol

**Document**: `docs/phase8/correction_pass/02_BENCHMARK_PROTOCOL_FINAL.md`  
**Purpose**: Specification of the standardized Phase 8 testing and evaluation protocol.

---

## 1. Window Definitions
Every outage benchmark window duration $T_{\text{window}}$ must satisfy:
$$T_{\text{window}} = T_{\text{pre}} + T_{\text{outage}} + T_{\text{post}}$$
Where:
- $T_{\text{pre}} \ge 15.0$ seconds (ensures filter convergence before blackout)
- $T_{\text{outage}} \in \{10.0, 30.0, 60.0, 120.0\}$ seconds
- $T_{\text{post}} \ge 20.0$ seconds ($30.0$s for 120s blackout)

## 2. Evaluation Metrics
1. **Drift Percentage**:
   $$\text{Drift (\%)} = \frac{\text{Endpoint Error}}{\text{Total Distance Traveled in Outage}} \times 100$$
2. **Along-Track / Cross-Track Errors**: Computed by projecting position errors onto road centerline tangent and normal vectors from Phase 7 OSM digital map.
3. **Trajectory Discontinuity Metrics**:
   - Max Position Step: $\Delta p = \max \|\mathbf{p}^+ - \mathbf{p}^-\|$
   - Max Velocity Step: $\Delta v = \max \|\mathbf{v}^+ - \mathbf{v}^-\|$
   - Pseudo-Acceleration Spike: $a_{\text{pseudo}} = \frac{\Delta v}{\Delta t}$
   - 95% Settling Time: $t_{95}$ elapsed time until recovery discrepancy $\le 5\%$.

## 3. Project Engineering Thresholds (Not unsourced ISRO claims)
- Position step threshold: $\le 3.5$ m
- Velocity step threshold: $\le 1.0$ m/s
- Pseudo-acceleration threshold: $\le 2.5$ m/s$^2$
- Heading step threshold: $\le 3.0^\circ$
