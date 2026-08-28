# Phase 1 Dataset Report — IO-VNBD

## SIH26168: AI-ML based Intelligent Dead Reckoning System for Seamless Navigation

---

## 1. What is IO-VNBD?

IO-VNBD (Inertial and Odometry Vehicle Navigation Benchmark Dataset) is the first large-scale, public dataset for vehicle positioning using inertial and odometry sensors. Created by researchers at Coventry University, it contains synchronized smartphone and vehicle CAN-bus sensor data collected on public roads in the UK, Nigeria, and France. It is designed for developing and benchmarking GPS-denied navigation algorithms.

## 2. Repository Structure

```
IO-VNBD-master/
├── Synchronised V abd S datasets/
│   ├── Categorised IOVNB Dataset/   (organized by driver)
│   └── Uncategorised IOVNB Dataset/  (flat S-Dataset/ and V-Dataset/)
├── Unsynchronised V and S Dataset/
├── README.md
├── README_1.pdf (full paper)
└── .gitattributes (LFS tracking)
```

## 3. Dataset Types & Sensors

- **Smartphone (S-)**: 24 columns — GPS, accelerometer, gyroscope, magnetometer, gravity, orientation
- **Vehicle (V-)**: 29 columns — VBOX GPS, CAN-bus wheel speeds, steering, yaw rate, acceleration, engine, brakes
- **Synchronized**: Smartphone + vehicle data collected simultaneously from same vehicle
- **Unsynchronized**: Collected independently

## 4. Sampling Rates

### Smartphone
- Records: 51746
- Duration: 86.2 minutes
- Effective frequency: 10.00 Hz
- Mean dt: 100.0 ms (expected 100 ms)
- Jitter (std): 0.8 ms
- Duplicate timestamps: 0
- Large gaps (>0.5s): 0

### Vehicle
- Records: 51746
- Duration: 86.2 minutes
- Effective frequency: 10.00 Hz
- Mean dt: 100.0 ms (expected 100 ms)
- Jitter (std): 0.0 ms
- Duplicate timestamps: 0
- Large gaps (>0.5s): 0

## 5. Selected Sequence

- **Name**: S1
- **Driver**: A
- **Location**: Coventry, UK
- **Description**: Dynamic driving in Coventry including B-Roads, 9 Roundabouts, Reverse maneuvers, and Hard Braking.

## 6. Data Quality

- Total null values across all columns: 0
- Constant columns: v_clutch

## 7. GPS Trajectory & Local Coordinates

GPS trajectory has been converted to local East-North-Up (ENU) metric coordinates using WGS84 → ECEF → ENU transformation with the first valid GPS point as origin.

See plots: `gps_trajectory.png`, `enu_trajectory.png`

### Simple Explanation of Coordinate Transformation (Understandable to a 10-Year-Old)

**What are we trying to calculate?**  
We are converting curved globe coordinates (latitude and longitude in degrees) into a flat city map measured in ordinary meters (East, North, and Up).

**Why do we need it for our SIH26168 problem?**  
Accelerometers measure acceleration in meters per second squared ($\text{m/s}^2$). If you try to add meters directly to latitude and longitude degrees, the math makes no sense because 1 degree of longitude is over 111,000 meters at the equator, but shrinks to 0 meters at the North Pole! We need flat, metric $(X, Y)$ coordinates to measure distance and vehicle error accurately.

**What does the transformation mean in plain English?**  
1. **WGS84**: Describes where you are on the curved, spinning Earth ellipsoid using latitude (how far North) and longitude (how far East).
2. **ECEF**: Translates those angles into a 3D $(X, Y, Z)$ box measured in meters from the center of the Earth.
3. **ENU**: Sets the starting point of our car ($t=0$) as $(0, 0, 0)$ meters, and measures how many meters the car drives East ($X$), North ($Y$), and Up ($Z$) on a flat tangent sheet touching that point.

**Real-world analogy:**  
Imagine holding a curved basketball. If you want to draw a tiny car driving on the court, drawing on the round ball is awkward. So you take a flat piece of square paper and rest it gently against the ball at your starting spot. Now you can easily measure distances in straight centimeters with a regular school ruler! That flat piece of paper is our ENU coordinate plane.

### Why This Matters for Our Project (SIH26168)

Converting GPS ground truth to the ENU metric frame gives us a flat Euclidean coordinate space where distances, RMSE errors, and velocities can be calculated directly with standard vector math, forming the ground truth reference for all hackathon benchmarks.

---

## 8. Stationary Sensor Analysis

Sensor bias during stationary periods:

| Sensor | Mean | Std | Unit |
|--------|------|-----|------|
| acc_x | -0.037103 | 0.421376 | m/s² |
| acc_y | -0.147456 | 0.459418 | m/s² |
| acc_z | 9.878072 | 0.122867 | m/s² |
| grav_x | 0.000316 | 0.002402 | m/s² |
| grav_y | -0.000261 | 0.005389 | m/s² |
| grav_z | 9.806589 | 3.1e-05 | m/s² |
| gyro_x | -0.000896 | 0.019433 | rad/s |
| gyro_y | -0.014255 | 0.072969 | rad/s |
| gyro_z | 0.001568 | 0.014653 | rad/s |
| mag_x | -7.483371 | 0.568704 | μT |
| mag_y | -26.09713 | 0.456356 | μT |
| mag_z | 31.621526 | 1.258617 | μT |

### Simple Explanation of Stationary Sensor Analysis (Understandable to a 10-Year-Old)

**What are we trying to calculate?**  
We are checking what the sensors read when the car is parked completely still at a red light, so we can discover the hidden errors built into the phone.

**Why do we need it for our SIH26168 problem?**  
When the car is parked, its true acceleration is $0\text{ m/s}^2$ and its turn rate is $0\text{ rad/s}$. Any number that the phone reports during this stopped time is a fake error (sensor bias). If we don't measure and subtract these fake numbers, our navigation will think the car is driving away while parked!

**What do the numbers represent?**  
- On `acc_z`, the sensor reads $9.878\text{ m/s}^2$. That is Earth's gravity pull ($9.81$) plus a small $+0.068\text{ m/s}^2$ factory error.
- On `gyro_y`, the sensor reports $-0.014\text{ rad/s}$ even though the car is not turning at all! If left uncorrected, that fake spin would turn the vehicle $50^\circ$ in one minute.

**Real-world analogy:**  
Imagine looking at a car speedometer while parked in your driveway with the engine off, and the needle points to 5 km/h! If you don't know the needle is stuck, you will think you are rolling down the street when you haven't even touched the gas pedal.

### Why This Matters for Our Project (SIH26168)

Measuring stationary bias proves that smartphone sensors cannot be used raw. These stationary averages become the calibration offsets subtracted in Phase 2 to prevent rapid drift.

---

## 9. GNSS Blackout & Raw Inertial Baseline

### Baseline Results Across Blackout Durations

| Duration (s) | Endpoint Error (m) | RMSE (m) | Max Error (m) | Drift % |
|-------------|-------------------|---------|--------------|---------|
| 10 | 61.8 | 51.2 | 90.0 | 49.2% |
| 30 | 142.4 | 89.6 | 152.7 | 47.4% |
| 60 | 284.9 | 184.1 | 291.1 | 60.2% |
| 120 | 405.9 | 241.7 | 415.2 | 33.0% |

### Simple Explanation of Raw Inertial Dead Reckoning (Understandable to a 10-Year-Old)

**What are we trying to calculate?**  
We are testing how well textbook physics ($v = \int a \, dt, p = \int v \, dt$) works if you take raw smartphone sensor data and blindly add it up during a GPS blackout without any calibration, filtering, or AI.

**Why do we need it for our SIH26168 problem?**  
Before you can claim you have invented a smart AI navigation system, you must establish a baseline to compare against. This baseline shows how badly a simple, naive approach fails.

**What do the results mean in plain English?**  
The table shows that over a 60-second blackout, the uncalibrated double-integration drifts by **284.9 meters** (a **60.2% drift**). That means if the car drove 472 meters, the computer got the final position wrong by more than half the total trip!

**Real-world analogy:**  
Imagine closing your eyes while walking across a playground, trying to guess where you are just by feeling the wind on your face and counting your steps. After 5 steps, you might be close. After 60 steps, you are completely lost and banging into the jungle gym!

### Why This Matters for Our Project (SIH26168)

This 60.2% drift is our official Phase 1 benchmark number. The entire objective of SIH26168 is to bring this drift down below 10% using calibration (Phase 2), signal processing (Phase 3), and intelligent machine learning (Phase 4).

---

## 10. Important Limitations

1. GPS is used as reference trajectory, not a high-precision RTK ground truth.
2. Smartphone GPS accuracy is typically 3-10m.
3. The raw inertial baseline uses a simplified phone-to-ENU rotation (yaw only).
4. No bias estimation, no calibration, no filtering applied yet.
5. Phone mounting orientation may vary between sequences.

## 11. Recommendations for Phase 2

1. **Phone-to-Vehicle Calibration**: Full 3D rotation matrix estimation
2. **ZUPT**: Zero-Velocity Update corrections during detected stops
3. **Vibration Filtering**: Low-pass filter before integration
4. **Bias Estimation**: Use stationary data to estimate and remove accelerometer/gyro bias
5. **AI/ML Velocity Estimation**: Train model to predict speed/velocity from IMU windows
6. **EKF/ESKF**: Implement proper state estimator for sensor fusion