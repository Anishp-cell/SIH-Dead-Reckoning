# Phase 7 Candidate Model: Multi-Hypothesis Generation & Scoring

**Project**: Smart India Hackathon 2026 — Problem Statement SIH26168  
**Sponsoring Agency**: Indian Space Research Organisation (ISRO)  
**System**: AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation  
**Document**: `docs/phase7/PHASE_7_CANDIDATE_MODEL.md`  

---

## 1. Motivation: Why Single-Hypothesis Matching Fails

A naive approach to map matching is "nearest-road snapping" (greedy 1-NN projection): find the road segment with the minimum Euclidean distance to the current estimated position and snap the vehicle to it.

In real-world urban and highway environments, greedy single-hypothesis matching fails catastrophically under several common geometric conditions:
1. **Parallel Road Corridors**: Highway main lanes and adjacent service/frontage roads often run parallel within $5 - 15\text{ m}$ of each other. A small lateral IMU bias can cause a greedy matcher to oscillate violently between the highway and the frontage road.
2. **Highway Exits & Splits**: As an exit ramp bifurcates from a highway, the physical distance between the two centerlines starts at $0\text{ m}$ and widens gradually. A single-hypothesis tracker cannot maintain uncertainty during the divergence period.
3. **Complex Multi-Way Intersections**: Roundabouts, 5-way junctions, and diamond interchanges present multiple roads intersecting within a $20\text{ m}$ radius with diverse headings.
4. **Bidirectional Centerlines**: Single-carriageway roads carry traffic in both directions. Without rigorous heading discrimination, a vehicle driving east could be matched to the westbound lane, inverting its heading pseudo-measurement.

To guarantee robust operation, Phase 7 implements a **probabilistic multi-hypothesis candidate generation and scoring model**.

---

## 2. Dynamic Search Radius Formulation

When the navigation filter has high certainty (e.g., immediately after an accurate fix), the search radius should be tight to eliminate distant spurious roads and minimize computation. Conversely, during an extended GNSS blackout, the position uncertainty grows, requiring a broader search horizon to prevent losing track of the true road corridor.

### 2.1 Formulation
Let $P_{\text{pos}} \in \mathbb{R}^{2 \times 2}$ be the horizontal position error covariance block from the 15-state ESKF:
$$P_{\text{pos}} = \begin{bmatrix} P_{ee} & P_{en} \\ P_{ne} & P_{nn} \end{bmatrix}$$

The major semi-axis of the position uncertainty ellipse is:
$$\sigma_p = \sqrt{\frac{P_{ee} + P_{nn}}{2} + \sqrt{\left(\frac{P_{ee} - P_{nn}}{2}\right)^2 + P_{en}^2}}$$

The candidate search radius $d_{\text{max}}$ is dynamically adjusted:
$$d_{\text{max}} = \text{clamp}\left(k_\sigma \, \sigma_p + d_{\min}, \, d_{\min}, \, d_{\text{upper}}\right)$$

Where parameter calibrations are:
- $d_{\min} = 15.0\text{ m}$: Minimum search radius, accommodating road half-width ($3.5 - 7.0\text{ m}$) plus standard GPS mapping inaccuracies.
- $k_\sigma = 2.5$: Corresponds to a $98.7\%$ statistical containment radius under a 2D Gaussian distribution.
- $d_{\text{upper}} = 60.0\text{ m}$: Hard ceiling to prevent excessive computational explosion in dense road meshes during long dead-reckoning intervals.

---

## 3. Candidate Generation Pipeline

At each map-matching cycle ($10\text{ Hz}$):

```
       ESKF State (p, v, q, P)
                 │
                 ▼
     Dynamic Radius Calculation
         d_max = f(P_pos)
                 │
                 ▼
      Spatial Index Query (k-d Tree)
   Find segments within d_max of p
                 │
                 ▼
    Bidirectional & Heading Filter
   Prune segments with |Δψ| > 90°
                 │
                 ▼
   Orthogonal Geometric Projection
   Compute d_perp, s_parallel, t_proj
                 │
                 ▼
       Candidate Scoring Engine
   Evaluate P_dist, P_head, P_vel
                 │
                 ▼
       Candidate Pruning & Rank
      Retain top K_max candidates
```

### 3.1 Bidirectional Road Disambiguation
For undivided bidirectional roads, the OSM ingestion engine creates two distinct directed segments:
- Forward segment $s_{\text{fwd}}$ with heading $\psi_s$.
- Reverse segment $s_{\text{rev}}$ with heading $\psi_s + \pi$.

When the vehicle navigates with estimated yaw $\psi_v$, we compute the angular difference for both candidates:
$$\Delta\psi_{\text{fwd}} = |\text{wrap}_\pi(\psi_v - \psi_s)|$$
$$\Delta\psi_{\text{rev}} = |\text{wrap}_\pi(\psi_v - (\psi_s + \pi))|$$

Any candidate whose heading error exceeds $90^\circ$ ($\frac{\pi}{2}\text{ rad}$) is immediately pruned:
$$\text{If } \Delta\psi > \frac{\pi}{2} \implies \Lambda(\mathbf{z}_t \mid c) = 0$$
This guarantees that a car driving eastbound is never matched to a westbound lane, completely eliminating 180° heading inversion artifacts.

### 3.2 Candidate Data Structure
Every surviving candidate is encapsulated in a comprehensive metadata object:
```python
@dataclass
class MapCandidate:
    segment: RoadSegment          # Matched road segment reference
    projected_enu: np.ndarray     # [e_proj, n_proj] (m)
    cross_track_dist: float       # Signed orthogonal distance d_perp (m)
    along_track_dist: float       # Distance along segment s_parallel (m)
    t_proj: float                 # Normalized projection parameter [0, 1]
    heading_diff_rad: float       # Yaw discrepancy Δψ (rad)
    distance_likelihood: float    # P_dist in [0, 1]
    heading_likelihood: float     # P_head in [0, 1]
    speed_likelihood: float       # P_vel in [0, 1]
    composite_likelihood: float   # Λ(z_t | c)
    prior_belief: float           # B_t^-(c)
    posterior_belief: float       # B_t(c)
```

---

## 4. Multi-Hypothesis Scoring & Emission Likelihoods

Candidate ranking combines three independent physical metrics into a composite emission likelihood:
$$\Lambda(\mathbf{z}_t \mid c) = P_{\text{dist}}(c) \cdot P_{\text{head}}(c) \cdot P_{\text{vel}}(c)$$

### 4.1 Gaussian Distance Likelihood
$$P_{\text{dist}}(c) = \exp\left(-\frac{d(c)^2}{2\sigma_d^2}\right)$$
- Calibration: $\sigma_d = 4.0\text{ m}$.
- At $d = 0\text{ m}$ (exact centerline), $P_{\text{dist}} = 1.0$.
- At $d = 4.0\text{ m}$ (adjacent lane), $P_{\text{dist}} = 0.606$.
- At $d = 8.0\text{ m}$ (off-road / sidewalk), $P_{\text{dist}} = 0.135$.
- At $d = 12.0\text{ m}$ (adjacent street), $P_{\text{dist}} = 0.011$.

### 4.2 Heading Direction Likelihood
$$P_{\text{head}}(c) = \exp\left(-\frac{\Delta\psi(c)^2}{2\sigma_\psi^2}\right)$$
- Calibration: $\sigma_\psi = 25^\circ \approx 0.4363\text{ rad}$.
- At $\Delta\psi = 0^\circ$ (perfect alignment), $P_{\text{head}} = 1.0$.
- At $\Delta\psi = 25^\circ$ (slight drift / curved road entry), $P_{\text{head}} = 0.606$.
- At $\Delta\psi = 50^\circ$ (sharp fork), $P_{\text{head}} = 0.135$.
- At $\Delta\psi \ge 90^\circ$ (perpendicular / reverse), $P_{\text{head}} = 0.0$.

### 4.3 Dynamic Speed Consistency Likelihood
Vehicles traveling on highways typically move at $20 - 35\text{ m/s}$ ($70 - 120\text{ km/h}$), whereas vehicles on residential alleys move at $5 - 10\text{ m/s}$. If the vehicle's dead-reckoning speed $v_{\text{fwd}}$ vastly exceeds the legal speed limit of candidate segment $s$:
$$\Delta v = \max(0, \, v_{\text{fwd}} - (v_{\text{max}} + \Delta v_{\text{tol}}))$$
$$P_{\text{vel}}(c) = \exp\left(-\frac{\Delta v^2}{2\sigma_v^2}\right)$$
- Parameters: $\Delta v_{\text{tol}} = 5.0\text{ m/s}$ ($18\text{ km/h}$ buffer), $\sigma_v = 3.0\text{ m/s}$.

---

## 5. Candidate Pruning & Belief Normalization

To maintain constant bounded computation:
1. **Hypothesis Cap**: The candidate set is restricted to the top $K_{\text{max}} = 10$ candidates ranked by emission likelihood.
2. **Probability Pruning**: Any candidate with posterior belief $B_t(c) < 10^{-4}$ is pruned.
3. **Belief Normalization**: The posterior probabilities of all surviving hypotheses are normalized to sum to unity:
   $$\sum_{i=1}^{K} B_t(c_i) = 1.0$$
4. **Primary Candidate Selection**: The candidate with the highest posterior belief is chosen as the MAP estimate:
   $$c^*_t = \arg\max_{c_i} B_t(c_i)$$
