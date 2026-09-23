# Phase 7 Topology Model: Directed Graph Connectivity & Causal Bayesian Tracking

**Project**: Smart India Hackathon 2026 — Problem Statement SIH26168  
**Sponsoring Agency**: Indian Space Research Organisation (ISRO)  
**System**: AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation  
**Document**: `docs/phase7/PHASE_7_TOPOLOGY_MODEL.md`  

---

## 1. Introduction: Spatial Geometry vs. Network Topology

Spatial proximity alone is insufficient for reliable vehicle localization. Consider two common driving scenarios:
1. **Parallel Road Corridors**: A major 4-lane arterial road running parallel to a narrow residential access road separated by a $6\text{ m}$ grass median. A car driving along the arterial road will frequently have its estimated GPS/DR position fluctuate within $2 - 4\text{ m}$ of both centerlines.
2. **Highway Overpasses / Underpasses**: An overpass crosses directly above a surface road at identical $(x, y)$ coordinates. Spatial distance between the vehicle position and both centerlines is nearly zero.

In both cases, Euclidean distance is ambiguous. However, **network topology** resolves this ambiguity:
- A vehicle on the arterial road cannot transfer onto the residential road without passing through a physical junction or intersection.
- A vehicle on the overpass cannot transition to the surface road without taking an on-ramp.

Phase 7 implements a **directed graph topology model** that couples spatial emission likelihoods with physical network connectivity via a causal first-order Hidden Markov Model (HMM).

---

## 2. Directed Road Network Graph Representation

The road network is modeled as a directed graph $G = (V, E)$:
- **Vertices $V$**: Road intersections, cul-de-sacs, and geometric inflection nodes:
  $$V = \{ n_i = (e_i, n_i) \in \mathbb{R}^2 \}$$
- **Directed Edges $E$**: Traversable directed road segments connecting nodes:
  $$E = \{ s_{ij} = (n_i \to n_j) \}$$

### 2.1 Forward & Reverse Segment Construction
- **One-Way Roads**: Ingested as a single directed edge $s_{ij} = (n_i \to n_j)$.
- **Bidirectional Roads**: Ingested as two distinct antiparallel edges:
  $$s_{ij} = (n_i \to n_j) \quad \text{and} \quad s_{ji} = (n_j \to n_i)$$
  These edges have opposite tangent vectors ($\mathbf{t}_{ji} = -\mathbf{t}_{ij}$) and opposite headings ($\psi_{ji} = \text{wrap}_\pi(\psi_{ij} + \pi)$).

### 2.2 Successor Adjacency Structure
For each segment $s_{ij}$, its set of topologically valid downstream successors is defined by all segments originating at vertex $n_j$:
$$\text{Successors}(s_{ij}) = \{ s_{jk} \in E \mid \text{start}(s_{jk}) = \text{end}(s_{ij}) \}$$

This lookup is indexed in a hash map for $\mathcal{O}(1)$ query time during runtime navigation.

---

## 3. Transition Probability Matrix $P(c_t \mid c_{t-1})$

Let $c_{t-1}$ be the candidate segment occupied at time $t-1$, and $c_t$ be a candidate segment at time $t$. The transition probability encodes the physical likelihood of moving from $c_{t-1}$ to $c_t$ over time interval $\Delta t = 0.1\text{ s}$.

### 3.1 Mathematical Formulation
The transition probability is partitioned into three distinct topological regimes:

$$P(c_t \mid c_{t-1}) = \begin{cases}
p_{\text{stay}} & \text{if } c_t = c_{t-1} \\
p_{\text{conn}} \cdot \Phi_{\text{disp}}(c_t, c_{t-1}) & \text{if } c_t \in \text{Successors}(c_{t-1}) \\
p_{\text{jump}} & \text{if } c_t \notin \{c_{t-1}\} \cup \text{Successors}(c_{t-1})
\end{cases}$$

#### 1. Staying on Same Segment ($c_t = c_{t-1}$)
At $10\text{ Hz}$, a vehicle traveling at $15\text{ m/s}$ moves $1.5\text{ m}$ per time step. For an average segment length of $30 - 100\text{ m}$, the vehicle remains on the same segment for tens or hundreds of consecutive cycles:
$$p_{\text{stay}} = 0.85$$

#### 2. Transitioning to Connected Successor ($c_t \in \text{Successors}(c_{t-1})$)
When approaching the end of a segment ($t_{\text{proj}} \to 1.0$), the vehicle transitions onto an adjacent downstream segment:
$$p_{\text{conn}} = 0.14$$

This transition is modulated by displacement consistency $\Phi_{\text{disp}}$:
$$\Phi_{\text{disp}}(c_t, c_{t-1}) = \exp\left(-\frac{|\Delta s_{\text{veh}} - \Delta s_{\text{topo}}|}{\sigma_{\text{topo}}}\right)$$
where:
- $\Delta s_{\text{veh}} = \int_{t-1}^t v_{\text{fwd}} \, d\tau \approx v_{\text{fwd}} \Delta t$ is the dead-reckoning distance traveled.
- $\Delta s_{\text{topo}} = (L_{t-1} - s_{\parallel, t-1}) + s_{\parallel, t}$ is the shortest path distance along the road network.
- $\sigma_{\text{topo}} = 3.0\text{ m}$ is the displacement tolerance.

#### 3. Topological Disconnection Penalty ($p_{\text{jump}}$)
If $c_t$ is not connected to $c_{t-1}$, transitioning between them represents an illegal teleportation across buildings, barriers, or medians:
$$p_{\text{jump}} = 10^{-4} = 0.0001$$
This heavy penalty ($8500\times$ lower than $p_{\text{stay}}$) acts as an immovable topological barrier.

---

## 4. Causal Recursive Forward Bayesian Tracking

Unlike post-processing map-matching algorithms (e.g., standard Viterbi matching) that require a full driving trajectory and backtrack from the future, an online navigation system must be strictly **causal** (zero future frames).

### 4.1 Recursive Forward Algorithm
At every time step $t$, the belief distribution over candidate hypotheses is updated:

1. **Prediction Step (Prior Belief)**:
   $$B_t^-(c_t) = \sum_{c_{t-1} \in \mathcal{C}_{t-1}} P(c_t \mid c_{t-1}) \, B_{t-1}(c_{t-1})$$

2. **Measurement Update (Posterior Belief)**:
   $$B_t(c_t) = \frac{\Lambda(\mathbf{z}_t \mid c_t) \, B_t^-(c_t)}{\sum_{c' \in \mathcal{C}_t} \Lambda(\mathbf{z}_t \mid c') \, B_t^-(c')}$$
   where $\Lambda(\mathbf{z}_t \mid c_t)$ is the composite emission likelihood (combining distance, heading, and speed).

3. **Maximum A Posteriori (MAP) Selection**:
   $$c^*_t = \arg\max_{c_t \in \mathcal{C}_t} B_t(c_t)$$

### 4.2 Belief Entropy & Map Confidence
To quantify the ambiguity of the current map match, we evaluate the Shannon entropy of the normalized belief distribution:
$$H_t = -\sum_{i=1}^K B_t(c_i) \ln\left(B_t(c_i) + \epsilon\right)$$

- When navigating along a single unobstructed road: $B_t(c_1) \approx 0.99 \implies H_t \approx 0.05\text{ nats}$ (high certainty).
- At a 3-way highway split: $B_t(c_1) \approx 0.35, B_t(c_2) \approx 0.33, B_t(c_3) \approx 0.32 \implies H_t \approx 1.10\text{ nats}$ (high ambiguity).

The continuous map confidence metric $c_{\text{map}}$ scales inversely with ambiguity and directly with the primary candidate's geometric match quality:
$$c_{\text{map}} = B_t(c^*_t) \cdot \exp\left(-\frac{d(c^*_t)^2}{2\sigma_d^2}\right) \cdot \cos^2(\Delta\psi(c^*_t))$$

---

## 5. Topological Disambiguation Mechanisms

### 5.1 Resolving Parallel Roads
Consider vehicle driving on Highway A ($y = 0\text{ m}$) with a parallel Frontage Road B ($y = 10\text{ m}$).
- At $t=0$, the vehicle starts on Highway A: $B_0(A) = 1.0, B_0(B) = 0.0$.
- At $t=10\text{ s}$, lateral drift places the estimated position at $y = 6\text{ m}$ (closer to Frontage Road B than Highway A!).
  - Euclidean Distance: $d_A = 6\text{ m}, d_B = 4\text{ m}$.
  - Emission Likelihood: $\Lambda(B) = \exp(-4^2/32) = 0.606 > \Lambda(A) = \exp(-6^2/32) = 0.325$.
  - Topological Prior:
    $$B^-(A) = P(A \mid A) B(A) = 0.85 \times 1.0 = 0.85$$
    $$B^-(B) = P(B \mid A) B(A) = 0.0001 \times 1.0 = 0.0001$$
  - Posterior:
    $$B(A) \propto 0.325 \times 0.85 = 0.27625$$
    $$B(B) \propto 0.606 \times 0.0001 = 0.00006$$
  - Normalizing: $B(A) = 99.98\%$, $B(B) = 0.02\%$.

**Conclusion**: The topological penalty completely prevents jumping to the parallel road, preserving the correct highway track.

### 5.2 Intersection Multi-Branch Tracking
At an intersection with multiple outgoing branches:
1. As the vehicle enters the intersection node, all connected successor edges $s_1, s_2, s_3$ receive equal topological transition probability $p_{\text{conn}} = 0.14$.
2. The belief splits across the branches according to heading alignment.
3. The filter maintains all three hypotheses simultaneously.
4. As the vehicle executes its turn and advances $10\text{ m}$ into branch $s_2$, the heading and distance likelihoods of branches $s_1$ and $s_3$ collapse to zero.
5. Belief concentrates onto branch $s_2$ without any discontinuous trajectory jumping.
