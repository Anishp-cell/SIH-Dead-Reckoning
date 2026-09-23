# Phase 7 Map Data Specification: Offline OpenStreetMap Road Network

**Project**: Smart India Hackathon 2026 — Problem Statement SIH26168  
**Sponsoring Agency**: Indian Space Research Organisation (ISRO)  
**System**: AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation  
**Document**: `docs/phase7/PHASE_7_MAP_DATA_SPECIFICATION.md`  

---

## 1. Executive Summary & Purpose

Phase 7 integrates offline OpenStreetMap (OSM) data to constrain dead-reckoning drift during GNSS outages. Because land vehicles are constrained to physical road corridors, map data provides geometric and topological bounds on position and heading. 

This document defines the complete specification for:
1. **OSM Data Schema & Ingestion**: Format of raw offline extracts, tag parsing, and drivable road filtering.
2. **Geodetic Coordinate Transformation**: Rigorous WGS84 to local East-North-Up (ENU) projection.
3. **Directed Road Segment Representation**: Data structures for centerline geometry, heading vectors, and one-way constraints.
4. **Spatial Indexing & Query Architecture**: Fast $k$-d tree spatial indexing for $\mathcal{O}(\log N)$ nearest-road lookups.
5. **Offline Storage & Resource Budget**: Storage footprints, memory consumption, and verification of zero-network runtime requirements.

---

## 2. OpenStreetMap Ingestion & Schema

### 2.1 Raw OSM Structure
The offline map layer consumes standard Overpass/OSM JSON extracts. The dataset contains two primary geometric primitives:
- **Nodes**: Point features defined by a 64-bit integer ID and geographic coordinates:
  $$\text{Node} = \{ \text{id} \in \mathbb{Z}^+, \, \lambda \in [-180, 180]^\circ, \, \phi \in [-90, 90]^\circ \}$$
- **Ways**: Ordered sequences of node references representing linear features, decorated with key-value semantic tags:
  $$\text{Way} = \{ \text{id} \in \mathbb{Z}^+, \, \text{nodes}: [n_1, n_2, \dots, n_k], \, \text{tags}: \{\text{key}: \text{val}\} \}$$

### 2.2 Drivable Highway Filtering
Non-navigable features (pedestrian footways, cycle paths, steps, bridleways) are strictly pruned during ingestion to prevent false candidate projections.

A way is retained if and only if:
1. It contains the key `"highway"`.
2. The highway tag value belongs to the approved drivable set:
   ```python
   DRIVABLE_HIGHWAYS = {
       "motorway", "trunk", "primary", "secondary", "tertiary",
       "unclassified", "residential", "service",
       "motorway_link", "trunk_link", "primary_link",
       "secondary_link", "tertiary_link", "living_street"
   }
   ```
3. Excluded tags include: `footway`, `cycleway`, `path`, `pedestrian`, `steps`, `track`, `bridleway`.

### 2.3 One-Way & Bidirectional Semantic Rules
Road directionality dictates valid vehicle heading and topological transitions:
- If `oneway == "yes"` or `oneway == "1"`: The road is strictly one-way in the order of node indexing ($n_1 \to n_2 \to \dots \to n_k$).
- If `oneway == "-1"`: The road is one-way in reverse order ($n_k \to n_{k-1} \to \dots \to n_1$).
- If `junction == "roundabout"` or `highway == "motorway"`: Treated as one-way by default unless explicitly tagged otherwise.
- Otherwise: The road is bidirectional. Ingestion creates two independent directed segments for every node pair: forward ($n_i \to n_{i+1}$) and reverse ($n_{i+1} \to n_i$).

### 2.4 Speed Limit Parsing
Speed limits inform dynamic velocity candidate scoring and anomaly detection:
- Standard format: `maxspeed = "30 mph"` or `maxspeed = "50"`.
- Conversion: Strings are parsed and normalized into SI units ($\text{m/s}$). Default urban speed limit is set to $13.41\text{ m/s}$ ($30\text{ mph}$ / $48.3\text{ km/h}$) if untagged.

---

## 3. Geodetic Coordinate Transformation (WGS84 $\to$ Local ENU)

To maintain numerical precision and mathematical consistency with the Phase 5/6 ESKF navigation frame, all geographic coordinates $(\phi, \lambda, h)$ are mapped into a local Cartesian East-North-Up (ENU) tangent plane.

### 3.1 Projection Origin
The tangent plane origin $(\phi_0, \lambda_0, h_0)$ matches the reference geodetic datum established in Phase 1 for the Coventry S1 driving route:
$$\phi_0 = 52.401660^\circ \text{ N}, \quad \lambda_0 = -1.505290^\circ \text{ E}, \quad h_0 = 147.50\text{ m}$$

### 3.2 Closed-Form Mathematical Formulation
1. **WGS84 Ellipsoid Parameters**:
   - Semi-major axis: $a = 6378137.0\text{ m}$
   - Flattening: $f = \frac{1}{298.257223563}$
   - First eccentricity squared: $e^2 = 2f - f^2 = 6.69437999014 \times 10^{-3}$

2. **Prime Vertical Radius of Curvature**:
   $$N(\phi) = \frac{a}{\sqrt{1 - e^2 \sin^2\phi}}$$

3. **Geodetic to Earth-Centered Earth-Fixed (ECEF)**:
   $$\begin{bmatrix} X \\ Y \\ Z \end{bmatrix} = \begin{bmatrix} (N(\phi) + h)\cos\phi\cos\lambda \\ (N(\phi) + h)\cos\phi\sin\lambda \\ (N(\phi)(1 - e^2) + h)\sin\phi \end{bmatrix}$$

4. **ECEF to Local ENU Tangent Plane**:
   Let $\Delta\mathbf{r}_{\text{ECEF}} = \mathbf{r}_{\text{ECEF}} - \mathbf{r}_{0, \text{ECEF}}$. The local ENU coordinates $\mathbf{p}_{\text{ENU}} = [e, n, u]^T$ are obtained by rotation:
   $$\begin{bmatrix} e \\ n \\ u \end{bmatrix} = \begin{bmatrix} -\sin\lambda_0 & \cos\lambda_0 & 0 \\ -\sin\phi_0\cos\lambda_0 & -\sin\phi_0\sin\lambda_0 & \cos\phi_0 \\ \cos\phi_0\cos\lambda_0 & \cos\phi_0\sin\lambda_0 & \sin\phi_0 \end{bmatrix} \begin{bmatrix} \Delta X \\ \Delta Y \\ \Delta Z \end{bmatrix}$$

5. **Inverse Transformation (ENU to WGS84)**:
   The inverse rotation maps $[e, n, u]^T \to \Delta\mathbf{r}_{\text{ECEF}}$, followed by Bowring's closed-form algorithm for geodetic latitude $\phi$, longitude $\lambda$, and ellipsoidal height $h$.

---

## 4. Directed Road Segment Representation

Every drivable way is decomposed into a set of contiguous, directed 2D line segments in the local ENU plane.

### 4.1 Segment Data Structure
Each directed segment $s_i$ contains:
```python
@dataclass(frozen=True)
class RoadSegment:
    segment_id: str          # Unique ID, e.g. "way12345_seg2_fwd"
    way_id: int              # Parent OSM way identifier
    start_node_id: int       # Source node ID
    end_node_id: int         # Target node ID
    start_enu: np.ndarray    # [e1, n1] 2D coordinates (m)
    end_enu: np.ndarray      # [e2, n2] 2D coordinates (m)
    length: float            # ||end_enu - start_enu|| (m)
    heading_rad: float       # Azimuth in radians [-pi, pi], 0=East, pi/2=North
    tangent_vector: np.ndarray # Unit vector [te, tn]
    normal_vector: np.ndarray  # Unit vector [-tn, te] (left-pointing)
    speed_limit_mps: float   # Legal speed limit (m/s)
    highway_type: str        # e.g., "primary", "secondary"
    oneway: bool             # True if one-way
```

### 4.2 Segment Geometry Calculations
For a segment from $\mathbf{p}_1 = [e_1, n_1]^T$ to $\mathbf{p}_2 = [e_2, n_2]^T$:
- Displacement vector: $\Delta\mathbf{p} = \mathbf{p}_2 - \mathbf{p}_1 = [\Delta e, \Delta n]^T$
- Segment length: $L = \|\Delta\mathbf{p}\|_2 = \sqrt{\Delta e^2 + \Delta n^2}$
- Tangent heading angle (standard ENU convention):
  $$\psi_{\text{road}} = \text{atan2}(\Delta n, \Delta e) \in [-\pi, \pi]$$
- Unit tangent vector: $\mathbf{t} = \frac{\Delta\mathbf{p}}{L} = [\cos\psi_{\text{road}}, \, \sin\psi_{\text{road}}]^T$
- Unit normal vector: $\mathbf{n} = [-\sin\psi_{\text{road}}, \, \cos\psi_{\text{road}}]^T$

---

## 5. Spatial Indexing & Query Architecture

Evaluating distance to all road segments in a metropolitan network ($>50,000$ segments) at $10\text{ Hz}$ would introduce unacceptable computational overhead ($\approx 100\text{ ms}$). To achieve sub-millisecond candidate retrieval, a spatial indexing structure is required.

### 5.1 $k$-d Tree Discretization
1. **Interpolated Waypoint Sampling**:
   Every road segment is sampled along its centerline at regular intervals:
   $$\Delta s_{\text{sample}} \le 10.0\text{ m}$$
   For a segment of length $L$, $M = \max(2, \lceil L / 10 \rceil + 1)$ points are generated:
   $$\mathbf{x}_k = \mathbf{p}_1 + \left(\frac{k}{M - 1}\right) (\mathbf{p}_2 - \mathbf{p}_1), \quad k \in \{0, 1, \dots, M-1\}$$

2. **Index Map**:
   Each sampled point $\mathbf{x}_k \in \mathbb{R}^2$ is mapped to its parent `RoadSegment` pointer.

3. **Spatial Search (`scipy.spatial.cKDTree`)**:
   A balanced 2D $k$-d tree is constructed over all sampled points. 
   - Construction time: $\mathcal{O}(N \log N)$ (executed once at startup).
   - Range query complexity: $\mathcal{O}(\log N + K)$, where $K$ is the number of points within search radius $d_{\text{max}}$.
   - Candidate segments are gathered as the unique set of parent segments corresponding to all points within $d_{\text{max}}$.

---

## 6. Offline Storage, Footprint & Resource Profile

### 6.1 Dataset Profile (Coventry S1 Test Area)
The test area encompasses the urban, suburban, and rural transit corridors of the Coventry S1 reference dataset:

| Parameter | Value |
| :--- | :--- |
| Geographic Bounding Box | Lat: $[52.380^\circ, 52.440^\circ]$, Lon: $[-1.560^\circ, -1.450^\circ]$ |
| Local ENU Coverage | East: $[-5.0\text{ km}, +5.0\text{ km}]$, North: $[-3.5\text{ km}, +4.5\text{ km}]$ |
| Total OSM Ways Ingested | 6,447 ways |
| Total OSM Nodes Ingested | 30,521 nodes |
| Total Directed Road Segments | 58,334 segments |
| Total Road Network Length | 1,057.5 km |
| Spatial Index Sampled Points | 194,022 points |
| Raw JSON File Size | 17.5 MB |
| In-Memory Database RAM | 24.8 MB |
| Cold Ingestion & Index Time | 2.15 seconds |
| Runtime Query Latency ($r=30\text{ m}$) | **0.062 ms** |

### 6.2 Offline Compliance Verification
1. **Network Disconnection**: The navigation engine operates with all network sockets disabled.
2. **Deterministic Lookups**: Road lookups rely exclusively on local memory structures.
3. **No Dynamic Tile Fetching**: Zero external HTTP/Overpass requests during execution.
