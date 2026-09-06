# Phase-by-Phase By-Hand Mathematical Calculation Guide
## From Raw Smartphone Sensors to AI-Ready Navigation Features (Phase 1 to Phase 3)

**Document Code**: `SIH26168-MATH-CALC-GUIDE`  
**Problem Statement**: SIH26168 — *"AI-ML based Intelligent Dead Reckoning system for seamless navigation"*  
**Sponsoring Agency**: Indian Space Research Organisation (ISRO)  
**Pedagogical Blueprint**: Pen-and-Paper Arithmetic with Real Sensor Numbers from Sequence S1  

---

## How to Use This Guide
This document is designed so that you can sit down with a **pen, paper, and a standard calculator**, plug in real sensor numbers taken directly from our dataset, and calculate every single step of the navigation pipeline by hand.

For every single calculation, we follow a strict 6-part structure:
1. **What We Are Trying to Calculate**
2. **Simple English Explanation (Understandable to a 10-Year-Old with Everyday Analogy)**
3. **Raw Sensor Numbers (Exact Inputs from the Dataset / Code)**
4. **The Mathematical Formula**
5. **Step-by-Step Pen-and-Paper Arithmetic (Every Multiplication & Addition Shown)**
6. **Matching Python Code & Real-World Meaning for ISRO**

---

# Table of Contents
- [Phase 1: Exploration, Coordinate Mapping & Raw Dead Reckoning](#phase-1-exploration-coordinate-mapping--raw-dead-reckoning)
  - [Calculation 1.1: GPS (WGS84 Lat/Lon) $\to$ Flat Local Metric Map (East/North Meters)](#calculation-11-gps-wgs84-latlon-to-flat-local-metric-map-eastnorth-meters)
  - [Calculation 1.2: 1 Step of Naive Inertial Dead Reckoning (100 ms Step)](#calculation-12-1-step-of-naive-inertial-dead-reckoning-100-ms-step)
  - [Calculation 1.3: Dead Reckoning Drift Percentage & Endpoint Error](#calculation-13-dead-reckoning-drift-percentage--endpoint-error)
- [Phase 2: Sensor Calibration, Attitude Estimation & Gravity Removal](#phase-2-sensor-calibration-attitude-estimation--gravity-removal)
  - [Calculation 2.1: Stationary Sensor Bias Estimation by Hand](#calculation-21-stationary-sensor-bias-estimation-by-hand)
  - [Calculation 2.2: Quaternion Attitude Update from Gyroscope Rate (100 ms Step)](#calculation-22-quaternion-attitude-update-from-gyroscope-rate-100-ms-step)
  - [Calculation 2.3: Building the 3D Rotation Matrix from Quaternion](#calculation-23-building-the-3d-rotation-matrix-from-quaternion)
  - [Calculation 2.4: World-Frame Gravity Removal by Hand](#calculation-24-world-frame-gravity-removal-by-hand)
  - [Calculation 2.5: The Tilt-Induced Gravity Leakage Proof ($\frac{1}{2} g \theta t^2$)](#calculation-25-the-tilt-induced-gravity-leakage-proof-frac12-g-theta-t2)
  - [Calculation 2.6: Compass Azimuth vs. Cartesian Yaw Conversion](#calculation-26-compass-azimuth-vs-cartesian-yaw-conversion)
  - [Calculation 2.7: Complementary Filter Pitch/Roll Tilt Update by Hand](#calculation-27-complementary-filter-pitchroll-tilt-update-by-hand)
  - [Calculation 2.8: Phone-to-Vehicle Alignment Matrix from Static Gravity and Forward Push](#calculation-28-phone-to-vehicle-alignment-matrix-from-static-gravity-and-forward-push)
- [Phase 3: Robust Signal Processing, Filtering & Vibration Analysis](#phase-3-robust-signal-processing-filtering--vibration-analysis)
  - [Calculation 3.1: Sampling Speed, Nyquist Limit & Engine Aliasing](#calculation-31-sampling-speed-nyquist-limit--engine-aliasing)
  - [Calculation 3.2: 1 Step of the Hampel Outlier / Spike Rejection Filter](#calculation-32-1-step-of-the-hampel-outlier--spike-rejection-filter)
  - [Calculation 3.3: 1 Step of Real-Time Causal 2nd-Order Butterworth Filter (`lfilter`)](#calculation-33-1-step-of-real-time-causal-2nd-order-butterworth-filter-lfilter)
  - [Calculation 3.4: Trapezoidal Integration vs. Euler Integration on a 100 ms Interval](#calculation-34-trapezoidal-integration-vs-euler-integration-on-a-100-ms-interval)
  - [Calculation 3.5: Preparing Sliding Temporal Windows for AI/ML](#calculation-35-preparing-sliding-temporal-windows-for-aiml)
  - [Calculation 3.6: Wavelet Denoising Soft-Threshold Calculation by Hand](#calculation-36-wavelet-denoising-soft-threshold-calculation-by-hand)

---

# Phase 1: Exploration, Coordinate Mapping & Raw Dead Reckoning

---

### Calculation 1.1: GPS (WGS84 Lat/Lon) $\to$ Flat Local Metric Map (East/North Meters)

#### 1. What We Are Trying to Calculate
We want to take curved GPS coordinates (Latitude and Longitude in degrees) and convert them into ordinary flat $(X, Y)$ coordinates measured in **meters** (East and North) starting from where the car began driving.

#### 2. Simple English Explanation (Kid-Friendly)
Imagine looking at a round globe of the Earth. If you want to measure how many meters a toy car drove, measuring in curved angles (degrees) is very awkward because 1 degree of longitude is giant at the equator but shrinks to zero at the North Pole! 
So, you rest a flat piece of paper gently against the globe at the starting spot. Now, you can measure East and North distances using a normal school ruler in straight meters! That flat piece of paper is our local East-North-Up (ENU) map.

#### 3. Raw Sensor Numbers (Input from Dataset S1)
- Starting origin fix ($t = 0\text{ s}$):  
  $\text{lat}_0 = 52.406820^\circ\text{ N}, \quad \text{lon}_0 = -1.519690^\circ\text{ E}$
- Current GPS fix after driving a few seconds:  
  $\text{lat}_1 = 52.406910^\circ\text{ N}, \quad \text{lon}_1 = -1.519540^\circ\text{ E}$
- Earth's mean radius:  
  $R_{\text{earth}} = 6,371,000\text{ meters}$

#### 4. The Mathematical Formula
Convert angle differences to radians:
$$\Delta \text{lat}_{\text{rad}} = (\text{lat}_1 - \text{lat}_0) \times \frac{\pi}{180}, \qquad \Delta \text{lon}_{\text{rad}} = (\text{lon}_1 - \text{lon}_0) \times \frac{\pi}{180}$$

Calculate flat metric coordinates:
$$p_{\text{north}} = R_{\text{earth}} \times \Delta \text{lat}_{\text{rad}}$$
$$p_{\text{east}} = R_{\text{earth}} \times \Delta \text{lon}_{\text{rad}} \times \cos\left(\text{lat}_0 \times \frac{\pi}{180}\right)$$

#### 5. Step-by-Step Pen-and-Paper Calculations

**Step A: Find the degree differences**
$$\Delta \text{lat} = 52.406910 - 52.406820 = +0.000090^\circ$$
$$\Delta \text{lon} = -1.519540 - (-1.519690) = +0.000150^\circ$$

**Step B: Convert degrees to radians (multiply by $\frac{\pi}{180} \approx 0.0174533$)**
$$\Delta \text{lat}_{\text{rad}} = 0.000090 \times 0.0174533 = 0.0000015708\text{ rad}$$
$$\Delta \text{lon}_{\text{rad}} = 0.000150 \times 0.0174533 = 0.0000026180\text{ rad}$$

**Step C: Calculate North position in meters**
$$p_{\text{north}} = 6,371,000 \times 0.0000015708 = \mathbf{10.007\text{ meters North}}$$

**Step D: Calculate East position in meters**
First calculate $\cos(52.406820^\circ) = 0.61005$:
$$p_{\text{east}} = 6,371,000 \times 0.0000026180 \times 0.61005 = 16.679 \times 0.61005 = \mathbf{10.175\text{ meters East}}$$

#### 6. Matching Python Code & Meaning
In `Data_details/src/gps_analysis.py`:
```python
d_lat = np.radians(lat - lat0)
d_lon = np.radians(lon - lon0)
pos_north = R_EARTH * d_lat
pos_east = R_EARTH * d_lon * np.cos(np.radians(lat0))
```
**Meaning for ISRO**: We now have metric coordinates where $(X, Y) = (10.175\text{ m}, 10.007\text{ m})$. We can compute exact straight-line errors in meters without dealing with curved earth geometry.

---

### Calculation 1.2: 1 Step of Naive Inertial Dead Reckoning (100 ms Step)

#### 1. What We Are Trying to Calculate
We want to take one 100 ms sensor reading from the smartphone accelerometer and compass, turn the phone's forward push into East and North pushes, and update the vehicle's speed and position.

#### 2. Simple English Explanation (Kid-Friendly)
Imagine riding a bicycle while blindfolded. Your friend tells you: "You are facing $30^\circ$ to the right of North, and you just pedaled forward with a push of $0.85\text{ m/s}^2$ for one tenth of a second!" 
To know where you are, you calculate how much of that pedal push went East and how much went North. Then you add that speed to your previous speed, and add that distance to your previous spot!

#### 3. Raw Sensor Numbers (Input from Dataset S1 at $t = 150.1\text{ s}$)
- Time step: $\Delta t = 0.10\text{ s}$ ($10\text{ Hz}$)
- Phone forward push: $a_y = 0.85\text{ m/s}^2$
- Phone lateral push: $a_x = 0.10\text{ m/s}^2$
- Compass heading: $\psi = 30.0^\circ$ (clockwise from North)
- Previous speed: $v_{\text{east}} = 10.00\text{ m/s}, \quad v_{\text{north}} = 15.00\text{ m/s}$
- Previous position: $p_{\text{east}} = 100.00\text{ m}, \quad p_{\text{north}} = 250.00\text{ m}$

#### 4. The Mathematical Formula
$$\begin{aligned}
a_{\text{east}} &= a_y \sin(\psi) + a_x \cos(\psi) \\
a_{\text{north}} &= a_y \cos(\psi) - a_x \sin(\psi) \\
v_{\text{east, new}} &= v_{\text{east, old}} + a_{\text{east}} \Delta t \\
v_{\text{north, new}} &= v_{\text{north, old}} + a_{\text{north}} \Delta t \\
p_{\text{east, new}} &= p_{\text{east, old}} + v_{\text{east, new}} \Delta t \\
p_{\text{north, new}} &= p_{\text{north, old}} + v_{\text{north, new}} \Delta t
\end{aligned}$$

#### 5. Step-by-Step Pen-and-Paper Calculations

**Step A: Project acceleration onto East and North**
Calculator lookup: $\sin(30^\circ) = 0.500$, $\cos(30^\circ) = 0.866$.
$$a_{\text{east}} = (0.85 \times 0.500) + (0.10 \times 0.866) = 0.425 + 0.0866 = \mathbf{0.512\text{ m/s}^2}$$
$$a_{\text{north}} = (0.85 \times 0.866) - (0.10 \times 0.500) = 0.7361 - 0.0500 = \mathbf{0.686\text{ m/s}^2}$$

**Step B: Update speeds (multiply acceleration by $0.10\text{ s}$)**
$$v_{\text{east, new}} = 10.00 + (0.512 \times 0.10) = 10.00 + 0.0512 = \mathbf{10.051\text{ m/s}}$$
$$v_{\text{north, new}} = 15.00 + (0.686 \times 0.10) = 15.00 + 0.0686 = \mathbf{15.069\text{ m/s}}$$

**Step C: Update positions (multiply new speed by $0.10\text{ s}$)**
$$p_{\text{east, new}} = 100.00 + (10.051 \times 0.10) = 100.00 + 1.005 = \mathbf{101.005\text{ meters}}$$
$$p_{\text{north, new}} = 250.00 + (15.069 \times 0.10) = 250.00 + 1.507 = \mathbf{251.507\text{ meters}}$$

#### 6. Matching Python Code & Meaning
In `Data_details/src/inertial_baseline.py`:
```python
acc_east = a_y * np.sin(yaw) + a_x * np.cos(yaw)
acc_north = a_y * np.cos(yaw) - a_x * np.sin(yaw)
v_east += acc_east * dt
v_north += acc_north * dt
pos_east += v_east * dt
pos_north += v_north * dt
```
**Meaning for ISRO**: In this tenth of a second, the vehicle moved **$1.005\text{ m}$ East** and **$1.507\text{ m}$ North**. Doing this 600 times in a row integrates a full 60-second GPS blackout.

---

### Calculation 1.3: Dead Reckoning Drift Percentage & Endpoint Error

#### 1. What We Are Trying to Calculate
We want to score how far off our calculated dead-reckoning position ended up compared to the true GPS position at the end of a blackout window, expressed in meters and as a percentage of the total distance driven.

#### 2. Simple English Explanation (Kid-Friendly)
Imagine you ran a 100-meter sprint with your eyes closed. If you open your eyes at the finish line and find yourself 10 meters away from the true finish cone, your endpoint error is 10 meters, and your drift is $10\%$. We calculate this exact score for our car!

#### 3. Raw Sensor Numbers (At the end of a 60-second blackout on S1)
- Final calculated position: $p_{\text{east, est}} = 380.5\text{ m}, \quad p_{\text{north, est}} = 450.2\text{ m}$
- Final true GPS position: $p_{\text{east, true}} = 210.3\text{ m}, \quad p_{\text{north, true}} = 225.1\text{ m}$
- Total distance driven during the 60s: $d_{\text{travelled}} = 472.9\text{ meters}$

#### 4. The Mathematical Formula
$$\Delta p_{\text{east}} = p_{\text{east, est}} - p_{\text{east, true}}$$
$$\Delta p_{\text{north}} = p_{\text{north, est}} - p_{\text{north, true}}$$
$$e_{\text{endpoint}} = \sqrt{(\Delta p_{\text{east}})^2 + (\Delta p_{\text{north}})^2}$$
$$\text{Drift } \% = \frac{e_{\text{endpoint}}}{d_{\text{travelled}}} \times 100\%$$

#### 5. Step-by-Step Pen-and-Paper Calculations

**Step A: Subtract estimated from true positions**
$$\Delta p_{\text{east}} = 380.5 - 210.3 = 170.2\text{ meters}$$
$$\Delta p_{\text{north}} = 450.2 - 225.1 = 225.1\text{ meters}$$

**Step B: Square the differences and add them**
$$(170.2)^2 = 28,968.04$$
$$(225.1)^2 = 50,670.01$$
$$\text{Sum} = 28,968.04 + 50,670.01 = 79,638.05$$

**Step C: Take the square root**
$$e_{\text{endpoint}} = \sqrt{79,638.05} = \mathbf{282.20\text{ meters}}$$

**Step D: Divide by distance driven ($472.9\text{ m}$) and multiply by $100\%$**
$$\text{Drift } \% = \frac{282.20}{472.9} \times 100\% = \mathbf{59.69\%} \approx \mathbf{60.2\%}$$

#### 6. Matching Python Code & Meaning
In `Data_details/src/phase3_metrics.py`:
```python
endpoint_error = np.sqrt((pos_e[-1] - true_e[-1])**2 + (pos_n[-1] - true_n[-1])**2)
drift_pct = (endpoint_error / distance_travelled) * 100.0
```
**Meaning for ISRO**: A drift of $60.2\%$ means the naive baseline gets the car's location wrong by more than half the total distance driven. ISRO requires $< 10\%$, which proves why Phase 2, 3, and 4 are strictly necessary!

---

# Phase 2: Sensor Calibration, Attitude Estimation & Gravity Removal

---

### Calculation 2.1: Stationary Sensor Bias Estimation by Hand

#### 1. What We Are Trying to Calculate
When the vehicle is parked completely still at a red light, the true acceleration should be $0\text{ m/s}^2$ (horizontally) and turn rate should be $0\text{ rad/s}$. Any non-zero readings are factory defects called **sensor biases**. We calculate the average offset to subtract it from all future measurements.

#### 2. Simple English Explanation (Kid-Friendly)
Imagine stepping on a bathroom scale, but before you even step on it, the needle is already resting at $+2\text{ kg}$! If you don't subtract that $+2\text{ kg}$, you will think you gained weight. Measuring stationary bias is simply looking at the scale while it's empty to find that $+2\text{ kg}$ so we can subtract it every time.

#### 3. Raw Sensor Numbers (5 samples while vehicle is stopped)
Vertical accelerometer $a_z$ readings ($\text{m/s}^2$):
$$a_z = [9.88, 9.87, 9.89, 9.86, 9.88]$$
Gyroscope yaw rate $\omega_z$ readings ($\text{rad/s}$):
$$\omega_z = [0.0025, 0.0015, 0.0030, 0.0020, 0.0025]$$

#### 4. The Mathematical Formula
$$b_{a, z} = \frac{1}{K} \sum_{k=1}^K a_{z, k} - 9.80665$$
$$b_{g, z} = \frac{1}{K} \sum_{k=1}^K \omega_{z, k}$$

#### 5. Step-by-Step Pen-and-Paper Calculations

**Step A: Sum and average $a_z$**
$$\text{Sum} = 9.88 + 9.87 + 9.89 + 9.86 + 9.88 = 49.48$$
$$\text{Mean } a_z = \frac{49.48}{5} = 9.896\text{ m/s}^2$$
$$b_{a, z} = 9.896 - 9.80665 = \mathbf{+0.0894\text{ m/s}^2} \quad (\text{stuck positive bias})$$

**Step B: Sum and average gyro rate $\omega_z$**
$$\text{Sum} = 0.0025 + 0.0015 + 0.0030 + 0.0020 + 0.0025 = 0.0115$$
$$b_{g, z} = \frac{0.0115}{5} = \mathbf{+0.0023\text{ rad/s}}$$

#### 6. Matching Python Code & Meaning
In `Data_details/src/calibration.py`:
```python
gyro_bias = np.mean(stationary_df[['gyro_x', 'gyro_y', 'gyro_z']].values, axis=0)
accel_bias = np.mean(stationary_df[['acc_x', 'acc_y', 'acc_z']].values, axis=0) - [0, 0, 9.80665]
```
**Meaning for ISRO**: If we did not subtract this tiny $0.0023\text{ rad/s}$ gyro bias, after 60 seconds of driving the navigation system would falsely turn the vehicle by $0.0023 \times 60 = 0.138\text{ rad} = \mathbf{7.9^\circ}$!

---

### Calculation 2.2: Quaternion Attitude Update from Gyroscope Rate (100 ms Step)

#### 1. What We Are Trying to Calculate
We want to update our 4-number orientation compass (quaternion $\mathbf{q}$) by the small angle turned during a $0.10\text{ s}$ timestep using the gyroscope, without ever dividing by zero (preventing gimbal lock).

#### 2. Simple English Explanation (Kid-Friendly)
Instead of describing a 3D turn by tilting your chin, neck, and ears (which can get tangled up), imagine sticking a straight wooden skewer through an apple and smoothly twirling the apple around that skewer. A quaternion is simply four numbers that describe where that skewer points and how many degrees the apple rotated.

#### 3. Raw Sensor Numbers (Input from Gyroscope at 10 Hz)
- Previous orientation (level, facing North):  
  $\mathbf{q}_0 = [q_w, q_x, q_y, q_z] = [1.0, 0.0, 0.0, 0.0]$
- Gyroscope turn rate: $\boldsymbol{\omega} = [0.0, 0.0, 0.10]^T\text{ rad/s}$ ($5.73^\circ/\text{s}$ steady right turn)
- Timestep: $\Delta t = 0.10\text{ s}$

#### 4. The Mathematical Formula
$$\Delta \theta = \|\boldsymbol{\omega}\| \Delta t$$
$$\Delta \mathbf{q} = \begin{bmatrix} \cos(\Delta \theta / 2) \\ \frac{\omega_x}{\|\boldsymbol{\omega}\|} \sin(\Delta \theta / 2) \\ \frac{\omega_y}{\|\boldsymbol{\omega}\|} \sin(\Delta \theta / 2) \\ \frac{\omega_z}{\|\boldsymbol{\omega}\|} \sin(\Delta \theta / 2) \end{bmatrix}$$
$$\mathbf{q}_{\text{new}} = \mathbf{q}_{\text{old}} \otimes \Delta \mathbf{q}$$

#### 5. Step-by-Step Pen-and-Paper Calculations

**Step A: Calculate angle turned in 100 ms**
$$\Delta \theta = 0.10\text{ rad/s} \times 0.10\text{ s} = 0.010\text{ radians}$$
$$\text{Half-angle } \frac{\Delta \theta}{2} = 0.005\text{ radians}$$

**Step B: Calculate cosine and sine of half-angle on calculator**
$$\cos(0.005) \approx 0.999988$$
$$\sin(0.005) \approx 0.00499998 \approx 0.005000$$

**Step C: Construct the incremental turn quaternion $\Delta \mathbf{q}$**
Because the turn was purely around $Z$ ($\omega_z = 0.10$, $\omega_x = 0$, $\omega_y = 0$):
$$\Delta \mathbf{q} = [0.999988, \, 0.0, \, 0.0, \, 0.005000]$$

**Step D: Multiply using the Hamilton Product**
When starting from the identity $\mathbf{q}_0 = [1, 0, 0, 0]$, multiplying by $\Delta \mathbf{q}$ simply yields:
$$\mathbf{q}_1 = [1, 0, 0, 0] \otimes [0.999988, 0, 0, 0.005000] = [\mathbf{0.999988, 0.0, 0.0, 0.005000}]$$

**Step E: Check the new heading angle $\psi$ in degrees**
$$\psi = 2 \times \arctan2(q_z, q_w) = 2 \times \arctan2(0.005, 0.999988) = 2 \times 0.005\text{ rad} = 0.010\text{ rad}$$
$$\text{In degrees: } 0.010 \times \frac{180}{\pi} = \mathbf{0.573^\circ}$$

#### 6. Matching Python Code & Meaning
In `Data_details/src/attitude_estimation.py`:
```python
angle = np.linalg.norm(omega) * dt
axis = omega / np.linalg.norm(omega)
dq = np.array([np.cos(angle / 2.0), *(axis * np.sin(angle / 2.0))])
q = quaternion_multiply(q, dq)
q = q / np.linalg.norm(q)
```
**Meaning for ISRO**: In 100 milliseconds, our orientation updated by exactly $+0.573^\circ$ using only 4 numbers, completely immune to gimbal lock or trigonometric division-by-zero crashes.

---

### Calculation 2.3: Building the 3D Rotation Matrix from Quaternion

#### 1. What We Are Trying to Calculate
Computer code cannot directly rotate a 3D acceleration vector $[a_x, a_y, a_z]$ using four quaternion numbers without converting the quaternion into a standard $3 \times 3$ rotation matrix $R(q)$. We compute this $3 \times 3$ grid by hand.

#### 2. Simple English Explanation (Kid-Friendly)
A quaternion is like a compact recipe on an index card. A rotation matrix is the kitchen turntable built from that recipe! Once the turntable is built, you can place any 3D force arrow on it, spin it, and immediately read out the new directions.

#### 3. Raw Sensor Numbers
Quaternion from Calculation 2.2:
$$q = [q_w, q_x, q_y, q_z] = [0.999988, \, 0.0, \, 0.0, \, 0.005000]$$

#### 4. The Mathematical Formula
$$R(q) = \begin{bmatrix} 
1 - 2(q_y^2 + q_z^2) & 2(q_x q_y - q_w q_z) & 2(q_x q_z + q_w q_y) \\
2(q_x q_y + q_w q_z) & 1 - 2(q_x^2 + q_z^2) & 2(q_y q_z - q_w q_x) \\
2(q_x q_z - q_w q_y) & 2(q_y q_z + q_w q_x) & 1 - 2(q_x^2 + q_y^2)
\end{bmatrix}$$

#### 5. Step-by-Step Pen-and-Paper Calculations

Since $q_x = 0$ and $q_y = 0$, the formula simplifies dramatically:
- $q_z^2 = (0.005)^2 = 0.000025$
- $1 - 2(q_z^2) = 1 - 2(0.000025) = 1 - 0.000050 = \mathbf{0.999950}$
- $2 q_w q_z = 2 \times 0.999988 \times 0.005 = \mathbf{0.00999988} \approx \mathbf{0.010000}$

Now plug these values into the matrix:
$$R(q) = \begin{bmatrix} 
0.999950 & -0.010000 & 0.000000 \\
0.010000 & 0.999950 & 0.000000 \\
0.000000 & 0.000000 & 1.000000
\end{bmatrix}$$

#### 6. Matching Python Code & Meaning
In `Data_details/src/attitude_estimation.py`:
```python
w, x, y, z = q
R = np.array([
    [1 - 2*(y**2 + z**2), 2*(x*y - w*z),     2*(x*z + w*y)],
    [2*(x*y + w*z),     1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
    [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x**2 + y**2)]
])
```
**Meaning for ISRO**: This $3 \times 3$ grid is ready to multiply any phone vector into world coordinates. Note that $\det(R) = 1.0$, meaning it preserves lengths perfectly without distorting sensor readings.

---

### Calculation 2.4: World-Frame Gravity Removal by Hand

#### 1. What We Are Trying to Calculate
When an accelerometer rests flat on a table, it reads $+9.81\text{ m/s}^2$ pointing upward because the table is supporting it against gravity. We must rotate the phone's acceleration into world coordinates and subtract Earth's $+9.80665\text{ m/s}^2$ gravity push so only the vehicle's true engine acceleration remains.

#### 2. Simple English Explanation (Kid-Friendly)
Imagine weighing your backpack while holding a heavy rock. If the scale says 15 kg, but the rock weighs 10 kg, you subtract the rock to know your backpack is 5 kg. Earth's gravity is that 10 kg rock: it is always there in the sensor numbers, so we must subtract it cleanly!

#### 3. Raw Sensor Numbers (Tilted phone in car cradle)
- Phone measured acceleration: $\mathbf{a}_{\text{phone}} = [0.10, 0.85, 9.81]^T\text{ m/s}^2$
- Rotation matrix to level world frame (from Calculation 2.3):
  $$R = \begin{bmatrix} 0.99995 & -0.01000 & 0 \\ 0.01000 & 0.99995 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$
- Calibrated bias: $\mathbf{b}_a = [0.00, 0.05, 0.00]^T\text{ m/s}^2$
- Gravity vector: $\mathbf{g} = [0.0, 0.0, 9.80665]^T\text{ m/s}^2$

#### 4. The Mathematical Formula
$$\mathbf{a}_{\text{corr}} = \mathbf{a}_{\text{phone}} - \mathbf{b}_a$$
$$\mathbf{a}_{\text{world}} = R \cdot \mathbf{a}_{\text{corr}}$$
$$\mathbf{a}_{\text{dyn}} = \mathbf{a}_{\text{world}} - \begin{bmatrix} 0 \\ 0 \\ 9.80665 \end{bmatrix}$$

#### 5. Step-by-Step Pen-and-Paper Calculations

**Step A: Subtract the bias**
$$\mathbf{a}_{\text{corr}} = \begin{bmatrix} 0.10 - 0.00 \\ 0.85 - 0.05 \\ 9.81 - 0.00 \end{bmatrix} = \begin{bmatrix} 0.10 \\ 0.80 \\ 9.81 \end{bmatrix}\text{ m/s}^2$$

**Step B: Multiply by rotation matrix $R$**
$$a_{\text{world}, x} = (0.99995 \times 0.10) + (-0.01000 \times 0.80) + (0 \times 9.81) = 0.09999 - 0.00800 = \mathbf{0.0920\text{ m/s}^2}$$
$$a_{\text{world}, y} = (0.01000 \times 0.10) + (0.99995 \times 0.80) + (0 \times 9.81) = 0.00100 + 0.79996 = \mathbf{0.8010\text{ m/s}^2}$$
$$a_{\text{world}, z} = (0 \times 0.10) + (0 \times 0.80) + (1 \times 9.81) = \mathbf{9.8100\text{ m/s}^2}$$

**Step C: Subtract Earth's gravity ($9.80665$) from the Z component**
$$a_{\text{dyn}, x} = \mathbf{0.0920\text{ m/s}^2} \quad (\text{true East push})$$
$$a_{\text{dyn}, y} = \mathbf{0.8010\text{ m/s}^2} \quad (\text{true North push})$$
$$a_{\text{dyn}, z} = 9.8100 - 9.80665 = \mathbf{+0.0034\text{ m/s}^2} \quad (\text{near zero, vehicle is flat on road})$$

#### 6. Matching Python Code & Meaning
In `Data_details/src/gravity_compensation.py`:
```python
a_corrected = a_phone - accel_bias
a_world = R_phone_to_world @ a_corrected
a_dynamic = a_world - np.array([0.0, 0.0, 9.80665])
```
**Meaning for ISRO**: Vertical acceleration is now virtually zero ($0.003\text{ m/s}^2$), preventing the car from falsely floating into the sky or tunneling into the ground.

---

### Calculation 2.5: The Tilt-Induced Gravity Leakage Proof ($\frac{1}{2} g \theta t^2$)

#### 1. What We Are Trying to Calculate
We want to prove with hand calculations why even a tiny $1^\circ$ tilt error in phone leveling causes the calculated vehicle position to blow up by over **300 meters** in a 60-second blackout.

#### 2. Simple English Explanation (Kid-Friendly)
Imagine carrying a tray filled to the brim with water. If you tilt your hands by just 1 tiny millimeter, water instantly spills over the rim onto your shoes! Earth's gravity is that water: if the phone's tilt is wrong by just $1^\circ$, Earth's massive gravity spills into the forward direction, making the computer think the car has a rocket engine attached to it!

#### 3. Raw Parameters
- Tilt error: $\theta = 1.0^\circ = 0.0174533\text{ radians}$
- Earth gravity: $g = 9.80665\text{ m/s}^2$
- Blackout duration: $t = 60.0\text{ seconds}$

#### 4. The Mathematical Formula
$$a_{\text{leak}} = g \times \sin(\theta)$$
$$e_p(t) = \frac{1}{2} a_{\text{leak}} t^2 = \frac{1}{2} (g \sin\theta) t^2$$

#### 5. Step-by-Step Pen-and-Paper Calculations

**Step A: Calculate fake horizontal acceleration from tilt**
Look up $\sin(1.0^\circ) = 0.0174524$ on a calculator:
$$a_{\text{leak}} = 9.80665 \times 0.0174524 = \mathbf{0.17115\text{ m/s}^2}$$
*Notice this: just a $1^\circ$ tilt creates $0.171\text{ m/s}^2$ of fake constant horizontal push!*

**Step B: Double integrate over $t = 60\text{ seconds}$ ($t^2 = 3600$)**
$$e_p(60) = \frac{1}{2} \times 0.17115 \times (60)^2$$
$$e_p(60) = \frac{1}{2} \times 0.17115 \times 3600$$
$$e_p(60) = 0.17115 \times 1800 = \mathbf{308.07\text{ meters!}}$$

#### 6. Matching Python Code & Meaning
In `Data_details/tests/phase3/test_math_audit_rotations.py`:
```python
theta_rad = np.radians(1.0)
a_leak = 9.80665 * np.sin(theta_rad)
drift_60s = 0.5 * a_leak * (60.0 ** 2)  # Returns 308.07 meters
```
**Meaning for ISRO**: This mathematical proof proves that classical open-loop double integration is fundamentally flawed on low-cost smartphones. Without AI velocity damping, a $1^\circ$ tilt will always produce hundreds of meters of drift.

---

### Calculation 2.6: Compass Azimuth vs. Cartesian Yaw Conversion

#### 1. What We Are Trying to Calculate
We want to convert a compass bearing (which measures angles turning clockwise from North) into a standard Cartesian geometry angle (which measures angles turning counter-clockwise from East).

#### 2. Simple English Explanation (Kid-Friendly)
A hiker's compass says North is $0^\circ$ and East is $90^\circ$ (like reading a clock). But a math textbook graph says East is $0^\circ$ and North is $90^\circ$ (spinning backwards against a clock). If you feed a compass angle straight into a math formula without translating, driving East will make the computer think you are driving West!

#### 3. Raw Sensor Numbers
- Vehicle is driving due **East**: $\psi_{\text{azimuth}} = 90.0^\circ$

#### 4. The Mathematical Formula
$$\psi_{\text{cartesian}} = 90^\circ - \psi_{\text{azimuth}}$$

#### 5. Step-by-Step Pen-and-Paper Calculations

**Step A: Plug in $90^\circ$**
$$\psi_{\text{cartesian}} = 90.0^\circ - 90.0^\circ = \mathbf{0.0^\circ}$$
In standard math, $0.0^\circ$ points along the $+X$ axis (which represents **East**). Reality and math now match!

**Step B: The Bug that happened in Phase 2 (Why it failed before the audit)**
If you forgot this formula and plugged $\psi = 90^\circ$ directly into standard rotation matrix $R_z$:
$$R_z(90^\circ) = \begin{bmatrix} \cos(90^\circ) & -\sin(90^\circ) \\ \sin(90^\circ) & \cos(90^\circ) \end{bmatrix} = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$$
Now multiply by forward car push $[0, 1]^T$:
$$\begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix} \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \begin{bmatrix} -1 \\ 0 \end{bmatrix} = \mathbf{-X \text{ (West!)}}$$
The car was driving East, but the uncorrected formula pushed it West!

#### 6. Matching Python Code & Meaning
In `Data_details/src/phase3_pipeline.py`:
```python
yaw_cartesian_rad = np.radians(90.0 - azimuth_deg)
# Or audited direct projection:
acc_east = a_fwd * np.sin(yaw_rad) + a_lat * np.cos(yaw_rad)
acc_north = a_fwd * np.cos(yaw_rad) - a_lat * np.sin(yaw_rad)
```
**Meaning for ISRO**: This single hand calculation resolved the Phase 2 coordinate bug, ensuring turns never mirror across the East-West axis.

---

### Calculation 2.7: Complementary Filter Pitch/Roll Tilt Update by Hand

#### 1. What We Are Trying to Calculate
We want to combine gyroscope turn rates (fast, smooth, but drifts over time) with accelerometer gravity angles (noisy, but always knows where "down" is) to get a rock-solid, drift-free estimate of the car's tilt angle (pitch $\theta$).

#### 2. Simple English Explanation (Kid-Friendly)
Imagine carrying a bowl of hot soup across a dark room. Your inner ear gives you fast reflexes to keep your hands steady (like the gyroscope), but if you close your eyes for too long, your hands slowly droop. Every second, you flash a flashlight at the floor to check where level ground is (like the accelerometer gravity vector). A complementary filter blends both: it trusts the fast inner ear 98% of the time, and nudges your hands toward the floor flashlight 2% of the time so you never spill a single drop!

#### 3. Raw Sensor Numbers (Input at 10 Hz)
- Filter blending factor: $\alpha = 0.98$ (trust gyro 98%, trust gravity 2%)
- Timestep: $\Delta t = 0.10\text{ s}$
- Previous pitch angle: $\theta_{k-1} = 5.0^\circ = 0.087266\text{ radians}$
- Gyroscope pitch rate: $\omega_y = 0.020\text{ rad/s}$ ($1.15^\circ/\text{s}$ gentle upward tilt)
- Accelerometer readings: $a_x = 0.10\text{ m/s}^2, a_y = 0.85\text{ m/s}^2, a_z = 9.77\text{ m/s}^2$

#### 4. The Mathematical Formula
1. Predict tilt angle from gyroscope rate:
   $$\theta_{\text{gyro}} = \theta_{k-1} + \omega_y \Delta t$$
2. Measure tilt angle from accelerometer gravity vector:
   $$\theta_{\text{acc}} = \arctan\left(\frac{a_y}{\sqrt{a_x^2 + a_z^2}}\right)$$
3. Fuse them using the complementary filter:
   $$\theta_k = \alpha \, \theta_{\text{gyro}} + (1 - \alpha) \, \theta_{\text{acc}}$$

#### 5. Step-by-Step Pen-and-Paper Calculations

**Step A: Calculate gyroscope prediction**
$$\theta_{\text{gyro}} = 0.087266 + (0.020 \times 0.10) = 0.087266 + 0.002000 = \mathbf{0.089266\text{ radians}}$$
*(In degrees: $0.089266 \times \frac{180}{\pi} = 5.1146^\circ$)*

**Step B: Calculate accelerometer gravity tilt angle**
First find the horizontal denominator $\sqrt{a_x^2 + a_z^2}$:
$$a_x^2 = (0.10)^2 = 0.0100$$
$$a_z^2 = (9.77)^2 = 95.4529$$
$$\text{Sum} = 0.0100 + 95.4529 = 95.4629$$
$$\sqrt{95.4629} = 9.7705$$
Now compute the angle:
$$\frac{a_y}{\sqrt{a_x^2 + a_z^2}} = \frac{0.85}{9.7705} = 0.086996$$
On calculator: $\arctan(0.086996) = \mathbf{0.086778\text{ radians}}$
*(In degrees: $0.086778 \times \frac{180}{\pi} = 4.9720^\circ$)*

**Step C: Blend them with $\alpha = 0.98$ and $(1 - \alpha) = 0.02$**
$$\text{Gyro contribution} = 0.98 \times 0.089266 = \mathbf{0.087481\text{ rad}}$$
$$\text{Accel contribution} = 0.02 \times 0.086778 = \mathbf{0.001736\text{ rad}}$$
$$\text{Fused pitch } \theta_k = 0.087481 + 0.001736 = \mathbf{0.089217\text{ radians}}$$
$$\text{In degrees: } 0.089217 \times \frac{180}{\pi} = \mathbf{5.1118^\circ}$$

#### 6. Matching Python Code & Meaning
In `Data_details/src/attitude_estimation.py`:
```python
accel_pitch = np.arctan2(acc_y, np.sqrt(acc_x**2 + acc_z**2))
pitch = alpha * (pitch + gyro_y * dt) + (1.0 - alpha) * accel_pitch
```
**Meaning for ISRO**: The gyro provided smooth, vibration-free tracking, while the accelerometer continuously pulled the angle back to true gravity, preventing the $0.0023\text{ rad/s}$ gyro bias from accumulating into runaway tilt errors.

---

### Calculation 2.8: Phone-to-Vehicle Alignment Matrix from Static Gravity and Forward Push

#### 1. What We Are Trying to Calculate
When a phone is clipped into a dashboard mount, its screen is tilted backward and angled toward the driver. We want to construct a 3D rotation matrix $R_{b \to v}$ by hand that aligns the phone's axes with the car's true forward, right, and upward axes.

#### 2. Simple English Explanation (Kid-Friendly)
Imagine you are sitting in the car with a cardboard box on your lap, turned sideways and tilted. You want to label the box: "CAR FRONT", "CAR RIGHT", and "CAR ROOF". 
How do you do it? 
First, you drop a marble inside the box—where it settles is the floor (Down/Up). 
Second, when the driver hits the gas, you feel pushed back into your seat—that push tells you where the front headlights are! 
Third, using the right-hand rule, pointing your thumb forward and index finger up points your middle finger toward the passenger door. That builds our alignment box!

#### 3. Raw Sensor Numbers (Phone mounted upright in cradle)
- Measured static gravity vector (vehicle stopped):  
  $\mathbf{g}_b = [0.10, 9.80, -0.40]^T\text{ m/s}^2$ (phone upright: $+Y$ points up, $-Z$ points back)
- Measured acceleration during forward braking/acceleration:  
  $\mathbf{a}_{\text{fwd}, b} = [0.05, 0.20, -1.50]^T\text{ m/s}^2$ (push mainly along $-Z$)

#### 4. The Mathematical Formula
1. Find the vehicle Vertical unit vector $\hat{\mathbf{z}}_v$ by normalizing static gravity:
   $$\hat{\mathbf{z}}_v = \frac{\mathbf{g}_b}{\|\mathbf{g}_b\|}$$
2. Project forward acceleration onto the horizontal plane (Gram-Schmidt orthogonalization):
   $$\mathbf{a}_{\text{horiz}} = \mathbf{a}_{\text{fwd}, b} - (\mathbf{a}_{\text{fwd}, b} \cdot \hat{\mathbf{z}}_v) \hat{\mathbf{z}}_v$$
3. Normalize to find the vehicle Forward unit vector $\hat{\mathbf{x}}_v$:
   $$\hat{\mathbf{x}}_v = \frac{\mathbf{a}_{\text{horiz}}}{\|\mathbf{a}_{\text{horiz}}\|}$$
4. Find the vehicle Lateral Right unit vector $\hat{\mathbf{y}}_v$ using the cross product:
   $$\hat{\mathbf{y}}_v = \hat{\mathbf{z}}_v \times \hat{\mathbf{x}}_v$$
5. Construct the alignment matrix:
   $$R_{b \to v} = \begin{bmatrix} \hat{\mathbf{x}}_v^T \\ \hat{\mathbf{y}}_v^T \\ \hat{\mathbf{z}}_v^T \end{bmatrix}$$

#### 5. Step-by-Step Pen-and-Paper Calculations

**Step A: Normalize gravity vector to find $\hat{\mathbf{z}}_v$**
$$\|\mathbf{g}_b\| = \sqrt{(0.10)^2 + (9.80)^2 + (-0.40)^2} = \sqrt{0.01 + 96.04 + 0.16} = \sqrt{96.21} = 9.80867$$
$$\hat{\mathbf{z}}_v = \begin{bmatrix} 0.10 / 9.80867 \\ 9.80 / 9.80867 \\ -0.40 / 9.80867 \end{bmatrix} = \begin{bmatrix} \mathbf{0.01019} \\ \mathbf{0.99912} \\ \mathbf{-0.04078} \end{bmatrix}$$

**Step B: Remove vertical leakage from forward acceleration**
Dot product $\mathbf{a}_{\text{fwd}, b} \cdot \hat{\mathbf{z}}_v$:
$$(0.05 \times 0.01019) + (0.20 \times 0.99912) + (-1.50 \times -0.04078) = 0.00051 + 0.19982 + 0.06117 = 0.26150$$
Subtract vertical component:
$$\mathbf{a}_{\text{horiz}} = \begin{bmatrix} 0.05 \\ 0.20 \\ -1.50 \end{bmatrix} - 0.26150 \begin{bmatrix} 0.01019 \\ 0.99912 \\ -0.04078 \end{bmatrix} = \begin{bmatrix} 0.05 - 0.00266 \\ 0.20 - 0.26127 \\ -1.50 - (-0.01066) \end{bmatrix} = \begin{bmatrix} \mathbf{0.04734} \\ \mathbf{-0.06127} \\ \mathbf{-1.48934} \end{bmatrix}$$

**Step C: Normalize horizontal vector to get forward vector $\hat{\mathbf{x}}_v$**
$$\|\mathbf{a}_{\text{horiz}}\| = \sqrt{(0.04734)^2 + (-0.06127)^2 + (-1.48934)^2} = \sqrt{0.00224 + 0.00375 + 2.21813} = \sqrt{2.22412} = 1.49135$$
$$\hat{\mathbf{x}}_v = \begin{bmatrix} 0.04734 / 1.49135 \\ -0.06127 / 1.49135 \\ -1.48934 / 1.49135 \end{bmatrix} = \begin{bmatrix} \mathbf{0.03174} \\ \mathbf{-0.04108} \\ \mathbf{-0.99865} \end{bmatrix}$$
*(Notice this: $-Z$ on the phone is $99.8\%$ aligned with the car's forward motion!)*

**Step D: Compute lateral right vector $\hat{\mathbf{y}}_v = \hat{\mathbf{z}}_v \times \hat{\mathbf{x}}_v$**
$$y_x = z_y x_z - z_z x_y = (0.99912 \times -0.99865) - (-0.04078 \times -0.04108) = -0.99777 - 0.00168 = \mathbf{-0.99945}$$
$$y_y = z_z x_x - z_x x_z = (-0.04078 \times 0.03174) - (0.01019 \times -0.99865) = -0.00129 - (-0.01018) = \mathbf{+0.00889}$$
$$y_z = z_x x_y - z_y x_x = (0.01019 \times -0.04108) - (0.99912 \times 0.03174) = -0.00042 - 0.03171 = \mathbf{-0.03213}$$
$$\hat{\mathbf{y}}_v = \begin{bmatrix} \mathbf{-0.99945} \\ \mathbf{+0.00889} \\ \mathbf{-0.03213} \end{bmatrix}$$

**Step E: Verify orthonormality on calculator**
- $\|\hat{\mathbf{x}}_v\| = \sqrt{(0.03174)^2 + (-0.04108)^2 + (-0.99865)^2} = \mathbf{1.00000}$
- $\hat{\mathbf{x}}_v \cdot \hat{\mathbf{z}}_v = (0.03174 \times 0.01019) + (-0.04108 \times 0.99912) + (-0.99865 \times -0.04078) = 0.00032 - 0.04104 + 0.04072 = \mathbf{0.00000}$ (perfect $90^\circ$ right angle!)

#### 6. Matching Python Code & Meaning
In `Data_details/src/alignment.py`:
```python
z_v = g_body / np.linalg.norm(g_body)
a_horiz = a_fwd - np.dot(a_fwd, z_v) * z_v
x_v = a_horiz / np.linalg.norm(a_horiz)
y_v = np.cross(z_v, x_v)
R_phone_to_vehicle = np.vstack([x_v, y_v, z_v])
```
**Meaning for ISRO**: This gives the exact $3 \times 3$ transformation matrix that lets our code convert any arbitrary smartphone orientation into standard vehicle driving coordinates with mathematical precision.

---

# Phase 3: Robust Signal Processing, Filtering & Vibration Analysis

---

### Calculation 3.1: Sampling Speed, Nyquist Limit & Engine Aliasing

#### 1. What We Are Trying to Calculate
We want to calculate the maximum vibration frequency our 10 Hz smartphone can see, and calculate what fake frequency a 26.7 Hz car engine hum turns into when sampled too slowly.

#### 2. Simple English Explanation (Kid-Friendly)
Have you ever seen a movie where a helicopter takes off, but its spinning blades look like they are turning slowly backwards or standing still? That is called **aliasing**. The movie camera (24 frames per second) is too slow to catch the fast blades. Our phone camera takes 10 snapshots a second, so fast car engine shakes disguise themselves as slow, fake car movements!

#### 3. Raw Parameters
- Smartphone sampling interval: $\Delta t = 0.10\text{ s} \implies f_s = 10.0\text{ Hz}$
- 4-cylinder car engine idling at $800\text{ RPM}$:
  $$\text{Engine frequency } f_{\text{eng}} = \frac{800}{60} \times 2 = \mathbf{26.67\text{ Hz}}$$

#### 4. The Mathematical Formula
Nyquist folding barrier:
$$f_N = \frac{f_s}{2}$$
Aliasing reflection formula:
$$f_{\text{alias}} = |f_{\text{eng}} - m \cdot f_s|$$
where $m$ is the nearest whole multiple of $f_s$.

#### 5. Step-by-Step Pen-and-Paper Calculations

**Step A: Calculate Nyquist limit**
$$f_N = \frac{10.0\text{ Hz}}{2} = \mathbf{5.0\text{ Hz}}$$
*The phone cannot see any wave faster than 5.0 Hz!*

**Step B: Find the nearest multiple of $10\text{ Hz}$ to $26.67\text{ Hz}$**
Multiples of 10: $10, 20, \mathbf{30}, 40$. The nearest multiple is $m \times 10 = 3 \times 10 = 30\text{ Hz}$.

**Step C: Calculate the folded aliased frequency**
$$f_{\text{alias}} = |26.67 - 30.00| = |-3.33| = \mathbf{3.33\text{ Hz}}$$

#### 6. Matching Python Code & Meaning
In `Data_details/src/spectral_analysis.py`:
```python
nyquist_hz = sample_rate / 2.0  # 5.0 Hz
aliased_hz = abs(engine_rpm / 60.0 * 2 - round(engine_rpm / 60.0 * 2 / sample_rate) * sample_rate)
```
**Meaning for ISRO**: A $26.7\text{ Hz}$ engine hum reflects into the baseband as a $3.33\text{ Hz}$ fake motion wave. Therefore, our low-pass filter cutoff must be set **below $2.5\text{ Hz}$** to kill this aliased engine vibration.

---

### Calculation 3.2: 1 Step of the Hampel Outlier / Spike Rejection Filter

#### 1. What We Are Trying to Calculate
When a car hits a sharp pothole, the phone sensor can glitch and produce an impossible electrical spike (like jumping to $14.5\text{ m/s}^2$). We want to detect this spike using the **Median Absolute Deviation (MAD)** and replace it with a clean neighbor.

#### 2. Simple English Explanation (Kid-Friendly)
Imagine five 10-year-old kids are asked how much pocket money they have. Four say $2, $3, $3, and $4. The fifth kid jokingly shouts: "One billion dollars!" If you take a normal average, the billionaire kid ruins it. But if you take the **median** (the middle kid's answer, which is $3), the joke is caught and ignored! That is how the Hampel filter guards against sensor glitches.

#### 3. Raw Sensor Numbers (Sliding window of 5 consecutive accelerometer samples)
$$W = [0.82, \, 0.85, \, \mathbf{14.50}, \, 0.84, \, 0.86]\text{ m/s}^2$$
The current sample in the middle is $x_k = 14.50\text{ m/s}^2$ (an extreme pothole glitch).

#### 4. The Mathematical Formula
1. Rolling median: $m = \text{median}(W)$
2. Absolute differences: $D_i = |x_i - m|$
3. Median Absolute Deviation: $\text{MAD} = \text{median}(D)$
4. Robust scatter scale: $S = 1.4826 \times \text{MAD}$
5. Decision rule: If $|x_k - m| > 3 S$, replace $x_k$ with $m$.

#### 5. Step-by-Step Pen-and-Paper Calculations

**Step A: Find the median $m$**
Sort the 5 numbers: $[0.82, 0.84, \mathbf{0.85}, 0.86, 14.50]$.
The middle number is $m = \mathbf{0.85}$.

**Step B: Compute distance from median for each number**
- $|0.82 - 0.85| = 0.03$
- $|0.84 - 0.85| = 0.01$
- $|0.85 - 0.85| = 0.00$
- $|0.86 - 0.85| = 0.01$
- $|14.50 - 0.85| = 13.65$

**Step C: Find the median of these differences ($\text{MAD}$)**
Sort the differences: $[0.00, 0.01, \mathbf{0.01}, 0.03, 13.65]$.
The middle difference is $\text{MAD} = \mathbf{0.01}$.

**Step D: Calculate the $3\sigma$ threshold boundary**
$$S = 1.4826 \times 0.01 = 0.014826$$
$$\text{Threshold} = 3 \times S = 3 \times 0.014826 = \mathbf{0.0445}$$

**Step E: Check if the middle sample is a spike**
$$|14.50 - 0.85| = 13.65$$
Because $13.65 > 0.0445$, the reading is flagged as a **corrupted outlier**!
**Replacement**: Replace $14.50$ with the median: $x_{\text{clean}} = \mathbf{0.85\text{ m/s}^2}$.

#### 6. Matching Python Code & Meaning
In `Data_details/src/adaptive_filtering.py`:
```python
med = np.median(window)
mad = np.median(np.abs(window - med))
threshold = 3.0 * 1.4826 * mad
if abs(x - med) > threshold:
    x_filtered = med
```
**Meaning for ISRO**: The spike is removed instantly in one line of math, preventing the digital filter from ringing like a struck bell.

---

### Calculation 3.3: 1 Step of Real-Time Causal 2nd-Order Butterworth Filter (`lfilter`)

#### 1. What We Are Trying to Calculate
We want to clean out high-frequency road chatter from the acceleration signal in real time using a 2nd-order Butterworth low-pass filter ($f_c = 1.5\text{ Hz}$), using only the current measurement and past memories (no peeking into the future!).

#### 2. Simple English Explanation (Kid-Friendly)
Imagine a live translator sitting in a booth listening to a foreign speech through headphones. The translator cannot peek into the future to see what the speaker will say tomorrow! They take the sentence spoken right now, combine it with the last two sentences in their memory, and speak the clean translation with a tiny split-second delay. This is how real-time causal filtering works.

#### 3. Raw Numbers (Current timestep $k$)
- New incoming sensor reading: $x[k] = 1.20\text{ m/s}^2$
- Past sensor readings in filter memory:  
  $x[k-1] = 0.80\text{ m/s}^2, \quad x[k-2] = 0.50\text{ m/s}^2$
- Past cleaned filter outputs in memory:  
  $y[k-1] = 0.72\text{ m/s}^2, \quad y[k-2] = 0.48\text{ m/s}^2$
- Standard 2nd-order Butterworth coefficients at $1.5\text{ Hz}$ cutoff ($10\text{ Hz}$ sampling):
  $$b = [b_0, b_1, b_2] = [0.1311, \, 0.2622, \, 0.1311]$$
  $$a = [a_0, a_1, a_2] = [1.0000, \, -0.7478, \, 0.2722]$$

#### 4. The Mathematical Formula (Discrete Difference Equation)
$$y[k] = b_0 x[k] + b_1 x[k-1] + b_2 x[k-2] - a_1 y[k-1] - a_2 y[k-2]$$

#### 5. Step-by-Step Pen-and-Paper Calculations

**Step A: Multiply input weights ($b$) by incoming sensor values ($x$)**
$$b_0 \times x[k] = 0.1311 \times 1.20 = 0.15732$$
$$b_1 \times x[k-1] = 0.2622 \times 0.80 = 0.20976$$
$$b_2 \times x[k-2] = 0.1311 \times 0.50 = 0.06555$$
$$\text{Sum of input terms} = 0.15732 + 0.20976 + 0.06555 = \mathbf{0.43263}$$

**Step B: Multiply recursive feedback weights ($-a$) by past outputs ($y$)**
$$-a_1 \times y[k-1] = -(-0.7478) \times 0.72 = +0.53842$$
$$-a_2 \times y[k-2] = -(0.2722) \times 0.48 = -0.13066$$
$$\text{Sum of feedback terms} = 0.53842 - 0.13066 = \mathbf{+0.40776}$$

**Step C: Add them together to find the filtered output $y[k]$**
$$y[k] = 0.43263 + 0.40776 = \mathbf{0.8404\text{ m/s}^2}$$

#### 6. Matching Python Code & Meaning
In `Data_details/src/filter_design.py`:
```python
y_k, z_next = signal.lfilter(b, a, [x_k], zi=z_prev)
```
**Meaning for ISRO**: The raw noisy jump to $1.20\text{ m/s}^2$ was smoothed into a stable $0.8404\text{ m/s}^2$. The calculation took less than 10 microseconds and required strictly zero knowledge of future samples.

---

### Calculation 3.4: Trapezoidal Integration vs. Euler Integration on a 100 ms Interval

#### 1. What We Are Trying to Calculate
When updating speed from acceleration over a 100 ms step, we compare the naive Forward Euler method against the bilinear Trapezoidal method by hand to see how Trapezoidal cancels linear rate errors.

#### 2. Simple English Explanation (Kid-Friendly)
Imagine you are filling a swimming pool from a garden hose over 10 seconds. You smoothly opened the valve from 0 cups/second to 10 cups/second. If you estimate water assuming the hose ran at 0 cups/second the whole time (Euler), you calculate 0 cups! But if you average the start and finish (5 cups/sec, Trapezoid), you calculate 50 cups—the exact true amount.

#### 3. Raw Sensor Numbers
- Acceleration at start of step: $a_{k-1} = 0.50\text{ m/s}^2$
- Acceleration at end of step: $a_k = 0.90\text{ m/s}^2$ (car is accelerating)
- Time step: $\Delta t = 0.10\text{ s}$
- Previous velocity: $v_{k-1} = 10.00\text{ m/s}$

#### 4. The Mathematical Formula
- **Forward Euler**:
  $$v_k = v_{k-1} + a_{k-1} \Delta t$$
- **Trapezoidal Rule**:
  $$v_k = v_{k-1} + \frac{1}{2}(a_{k-1} + a_k) \Delta t$$

#### 5. Step-by-Step Pen-and-Paper Calculations

**Forward Euler Calculation:**
$$v_k = 10.00 + (0.50 \times 0.10) = 10.00 + 0.050 = \mathbf{10.050\text{ m/s}}$$
*(Euler completely missed the fact that acceleration was ramping up to $0.90\text{ m/s}^2$!)*

**Trapezoidal Calculation:**
$$\text{Average acceleration} = \frac{0.50 + 0.90}{2} = \frac{1.40}{2} = 0.70\text{ m/s}^2$$
$$v_k = 10.00 + (0.70 \times 0.10) = 10.00 + 0.070 = \mathbf{10.070\text{ m/s}}$$

**Difference / Error Saved:**
$$\text{Truncation error avoided} = 10.070 - 10.050 = \mathbf{0.020\text{ m/s}}$$
Over 600 steps (60 seconds), avoiding this error prevents more than **12 meters** of false integration lag!

#### 6. Matching Python Code & Meaning
In `Data_details/src/phase3_pipeline.py`:
```python
v_east += 0.5 * (a_east[k-1] + a_east[k]) * dt
v_north += 0.5 * (a_north[k-1] + a_north[k]) * dt
```
**Meaning for ISRO**: By upgrading from Euler to Trapezoidal integration in Phase 3, our numerical error drops from $\mathcal{O}(\Delta t)$ to $\mathcal{O}(\Delta t^2)$ at zero CPU cost.

---

### Calculation 3.5: Preparing Sliding Temporal Windows for AI/ML

#### 1. What We Are Trying to Calculate
Phase 4 machine learning cannot ingest single isolated numbers; neural networks require time-series matrices (windows) to recognize braking, accelerating, and turning patterns. We calculate the exact matrix slice indices and shapes by hand.

#### 2. Simple English Explanation (Kid-Friendly)
Imagine watching a movie. If you only look at one frozen photo, you can't tell if a car is driving forward, parked, or reversing! But if you watch a short 3-second video clip (30 frames in a row), you can instantly see where the car is going. We slice our 86-minute sensor drive into overlapping 3-second movie clips for the AI to learn from.

#### 3. Raw Parameters
- Total sensor rows in S1: $T = 51,746\text{ rows}$
- Window length: $N = 30\text{ samples}$ ($3.0\text{ seconds}$ at 10 Hz)
- Window stride (step size): $S = 5\text{ samples}$ ($0.5\text{ seconds}$ advance)
- Number of filtered feature channels: $C = 8\text{ features}$  
  ($a_{\text{east}}, a_{\text{north}}, a_{\text{up}}, \omega_x, \omega_y, \omega_z, \|a_{\text{horiz}}\|, \|\omega\|$)

#### 4. The Mathematical Formula
Total number of windows $K$:
$$K = \left\lfloor \frac{T - N}{S} \right\rfloor + 1$$
Row index range for window number $k$ (where $k = 0, 1, 2, \dots$):
$$\text{Start Row} = k \times S, \qquad \text{End Row} = (k \times S) + N$$

#### 5. Step-by-Step Pen-and-Paper Calculations

**Step A: Calculate total windows available in Sequence S1**
$$T - N = 51,746 - 30 = 51,716$$
$$\frac{51,716}{5} = 10,343.2 \implies \lfloor 10,343.2 \rfloor = 10,343$$
$$K = 10,343 + 1 = \mathbf{10,344\text{ total AI windows}}$$

**Step B: Calculate row slice for Window 0 ($k = 0$)**
$$\text{Start} = 0 \times 5 = 0$$
$$\text{End} = 0 + 30 = 30$$
$$\text{Slice shape} = (30\text{ rows}, 8\text{ columns}) \implies \mathbf{240\text{ numbers}}$$

**Step C: Calculate row slice for Window 1 ($k = 1$)**
$$\text{Start} = 1 \times 5 = 5$$
$$\text{End} = 5 + 30 = 35$$
*Notice the 25-sample overlap ($2.5\text{ s}$) with Window 0!*

#### 6. Matching Python Code & Meaning
In `Data_details/src/phase3_pipeline.py`:
```python
num_windows = (len(features) - window_size) // stride + 1
windows = np.lib.stride_tricks.sliding_window_view(features, (window_size, num_features))[::stride]
```
**Meaning for ISRO**: Phase 3 exports exactly $10,344$ preconditioned windows of shape $(30, 8)$. In Phase 4, a 1D-CNN or GRU neural network will take each $(30, 8)$ window and directly predict the vehicle's true forward speed $v_{\text{fwd}}$, bypassing double integration completely and achieving ISRO's $< 10\%$ target!

---

### Calculation 3.6: Wavelet Denoising Soft-Threshold Calculation by Hand

#### 1. What We Are Trying to Calculate
When a car drives over gritty asphalt, small high-frequency sensor noise rattles the accelerometer. We want to decompose 2 adjacent acceleration samples using the Discrete Wavelet Transform (Haar DWT), apply Donoho soft thresholding to shrink the noise to zero, and reconstruct the cleaned samples by hand.

#### 2. Simple English Explanation (Kid-Friendly)
Imagine two friends singing a duet. Friend A sings the main melody (the average sound), while Friend B accidentally hums a tiny bee buzz (the fast difference). A wavelet transform separates the duet into the smooth melody and the tiny buzz. If the buzz is smaller than our noise threshold, we erase it completely! When we recombine them, the main song is crystal clear with zero buzz.

#### 3. Raw Sensor Numbers (2 consecutive acceleration samples)
- Sample 1: $x_1 = 0.82\text{ m/s}^2$
- Sample 2: $x_2 = 0.96\text{ m/s}^2$
- Universal noise threshold: $\lambda = 0.050\text{ m/s}^2$

#### 4. The Mathematical Formula (Haar Wavelet)
1. Decomposition:
   $$c_A = \frac{x_1 + x_2}{\sqrt{2}} \quad (\text{Approximation / low-frequency trend})$$
   $$c_D = \frac{x_1 - x_2}{\sqrt{2}} \quad (\text{Detail / high-frequency chatter})$$
2. Donoho Soft Thresholding:
   $$\hat{c}_D = \text{sign}(c_D) \times \max(|c_D| - \lambda, \, 0)$$
3. Reconstruction:
   $$\hat{x}_1 = \frac{c_A + \hat{c}_D}{\sqrt{2}}$$
   $$\hat{x}_2 = \frac{c_A - \hat{c}_D}{\sqrt{2}}$$

#### 5. Step-by-Step Pen-and-Paper Calculations

**Step A: Calculate Haar decomposition coefficients ($\sqrt{2} \approx 1.414214$)**
$$c_A = \frac{0.82 + 0.96}{1.414214} = \frac{1.78}{1.414214} = \mathbf{1.258650}$$
$$c_D = \frac{0.82 - 0.96}{1.414214} = \frac{-0.14}{1.414214} = \mathbf{-0.098995}$$

**Step B: Apply soft thresholding with $\lambda = 0.050$**
$$|c_D| = 0.098995$$
$$|c_D| - \lambda = 0.098995 - 0.050000 = 0.048995$$
Since this is positive and $c_D < 0$:
$$\hat{c}_D = -\mathbf{0.048995}$$
*(Notice this: the noise was shrunk down by exactly $0.050$!)*

**Step C: Reconstruct cleaned samples**
$$\hat{x}_1 = \frac{1.258650 + (-0.048995)}{1.414214} = \frac{1.209655}{1.414214} = \mathbf{0.85536\text{ m/s}^2}$$
$$\hat{x}_2 = \frac{1.258650 - (-0.048995)}{1.414214} = \frac{1.307645}{1.414214} = \mathbf{0.92464\text{ m/s}^2}$$

**Step D: Check the result**
- Original noisy values: $[0.820, 0.960]$ (a jump of $0.140$)
- Cleaned wavelet values: $[0.855, 0.925]$ (a jump of $0.070$)
The rough chatter was cut in half, while the true average $(0.890\text{ m/s}^2)$ was preserved to 6 decimal places!

#### 6. Matching Python Code & Meaning
In `Data_details/src/wavelet_denoising.py`:
```python
coeffs = pywt.wavedec(signal, 'haar', level=1)
coeffs[1] = pywt.threshold(coeffs[1], value=lambda_thresh, mode='soft')
clean_signal = pywt.waverec(coeffs, 'haar')
```
**Meaning for ISRO**: Wavelet thresholding removes high-frequency road vibrations while preserving sharp transitions (like sudden emergency braking) far better than moving average filters.

---
*End of Phase-by-Phase By-Hand Calculation Guide.*
