# Phase 4 Data Splitting & Validation Protocol

**Project**: Smart India Hackathon 2026 — Problem Statement SIH26168  
**Sponsoring Agency**: Indian Space Research Organisation (ISRO)  
**System**: AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation  
**Phase**: Phase 4 — AI Motion Intelligence  
**Document Code**: `SIH26168-DATA-SPLIT-01`  
**Classification**: PROTOCOL SPECIFICATION & VALIDATION  

---

## 1. Sequence Overview & Integrity Analysis

### Primary Sequence S1 (Coventry, UK)
- **Total Duration**: $5,174.6\text{ seconds}$ ($86.24\text{ minutes}$)
- **Total Synchronized Samples**: $51,746\text{ rows}$ at exactly $10.00\text{ Hz}$
- **Total Cumulative Traveled Distance**: $37,245.4\text{ meters}$ ($37.25\text{ km}$)
- **Environmental Diversity**: 9 urban roundabouts, highway cruising stretches, traffic light stop-and-go intervals, dynamic braking, and urban road conditions.

---

## 2. Ground-Truth Target Provenance & Unit Reconciliation

### Analysis of Speed Data Sources in IO-VNBD S1
1. **Vehicle CAN Indicated Speed (`v_speed_mps`)**:
   - Source: Ford Fiesta onboard powertrain CAN bus (`V-Dataset/V-S1.csv`).
   - Units: Derived from `v_speed_kmh / 3.6` in meters per second (m/s).
   - Speed range: $0.00\text{ m/s}$ to $18.91\text{ m/s}$ ($68.09\text{ km/h}$).
   - Integrated distance: $38,351.8\text{ meters}$ ($102.9\%$ of geodesic ground truth).
   - Characteristics: Extremely high resolution, zero satellite occlusion degradation, continuous across tunnels and underpasses.
   - **Status**: **Primary Supervised Training Target**.

2. **Racelogic VBOX DGPS Speed (`vbox_mps = v_gps_speed_kmh / 3.6`)**:
   - Source: Roof-mounted dual-antenna high-precision differential GPS/INS.
   - Units: Meters per second (m/s).
   - Speed range: $0.00\text{ m/s}$ to $26.06\text{ m/s}$ ($93.83\text{ km/h}$).
   - Integrated distance: $37,961.2\text{ meters}$ ($101.9\%$ of geodesic ground truth).
   - Correlation with CAN speed: $r = 0.9968$ (median difference $\Delta v = 0.075\text{ m/s}$).
   - **Status**: **Secondary Reference Target / High-Speed Benchmark**.

3. **Smartphone GPS Speed (`gps_speed_mps` in `S-S1.csv`)**:
   - Found to be improperly scaled by $3.6$ in historical archives (yielding max $5.23\text{ m/s}$).
   - **Status**: **REJECTED** as a training target. Model targets must come strictly from vehicle-side instrumentation (`v_speed_mps`).

---

## 3. Temporal Block Split Architecture with Purge Gaps

To prevent data leakage caused by window overlap ($W=30, S=5$), the dataset is partitioned chronologically into three non-overlapping contiguous time blocks separated by $5.0\text{ s}$ ($50\text{ sample}$) purge buffers.

```text
Sequence S1 (51,746 samples = 86.2 min)
├───────────────────────────────┬───────┬───────────────────┬───────┬───────────────────┤
│          TRAINING             │ PURGE │    VALIDATION     │ PURGE │       TEST        │
│          (70.0%)              │ GAP 1 │      (14.9%)      │ GAP 2 │      (14.9%)      │
│   Samples: 0 -> 36,220        │ 50 smp│36,270 -> 43,980   │ 50 smp│44,030 -> 51,745   │
│   Duration: 0.0 -> 3622.0s    │  5.0s │3627.0 -> 4398.0s  │  5.0s │4403.0 -> 5174.5s  │
│   Windows: ~7,239             │EXCLUDE│ Windows: ~1,542   │EXCLUDE│ Windows: ~1,543   │
└───────────────────────────────┴───────┴───────────────────┴───────┴───────────────────┘
```

### Mathematical Proof of Independence
For window $i$ ending at $t_i$ and window $j$ starting at $t_j$:
$$\text{Overlap Condition}: t_i \ge t_j$$
With purge gap $G = 50\text{ samples}$ between blocks:
$$t_{\text{val, start}} - t_{\text{train, end}} = 50 > W = 30\text{ samples}$$
Therefore:
$$\text{Shared raw IMU measurements} \equiv \emptyset$$
Zero raw accelerometer or gyroscope samples are shared between Train, Val, and Test partitions.

---

## 4. Speed Distribution Across Partitions

| Partition | Duration | Distance | Mean Speed | Max Speed | Standstill % ($v < 0.1$) | Cruising % ($v > 5.0$) |
|-----------|----------|----------|------------|-----------|--------------------------|------------------------|
| **Train** | $3,622.0\text{ s}$ | $26.12\text{ km}$ | $7.21\text{ m/s}$ | $18.91\text{ m/s}$ | $18.4\%$ | $58.2\%$ |
| **Validation** | $771.0\text{ s}$ | $5.94\text{ km}$ | $7.70\text{ m/s}$ | $18.91\text{ m/s}$ | $16.8\%$ | $62.1\%$ |
| **Test** | $741.5\text{ s}$ | $6.29\text{ km}$ | $8.48\text{ m/s}$ | $18.91\text{ m/s}$ | $14.2\%$ | $67.4\%$ |

Every partition exhibits a balanced distribution of standstill, urban driving, and higher-speed road segments.
