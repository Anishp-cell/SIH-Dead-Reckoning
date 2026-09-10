# Understanding Sensor Noise, Vibration, and Filtering in Dead Reckoning
## A Plain-Language Guide for Students, Hackathon Judges, and Engineers

**Problem Statement**: SIH26168 — *"AI-ML based Intelligent Dead Reckoning system for seamless navigation"* (ISRO)  
**Phase**: Phase 3 (Signal Processing & Noise Filtering)  

---

### 1. What is a "Signal" vs. "Noise"?
When you drive a car with a smartphone mounted on the dashboard, the phone's sensors measure everything at once:
- **The True Signal**: The actual motion of the car — speeding up when you press the gas pedal, slowing down when you brake, and rotating when you turn a corner.
- **The Noise**: Unwanted vibrations and electrical chatter that have nothing to do with where the car is going:
  - The hum of the engine idling at 800 RPM.
  - The car body vibrating as rubber tires roll over coarse asphalt.
  - The phone shaking slightly in its plastic mount.
  - Sensor electronics introducing microscopic measurement errors.

---

### 2. Why Does Unfiltered Noise Destroy Navigation?
To calculate a vehicle's position from an accelerometer, software must integrate twice:
$$\text{Acceleration} \xrightarrow{\int dt} \text{Velocity} \xrightarrow{\int dt} \text{Position}$$

In mathematics, integration acts like an accumulator that adds up every number over time.
If your acceleration sensor has even a tiny constant error of **$0.05\text{ m/s}^2$** (less than 1% of Earth's gravity):
- In 1 second, the position error is only $0.025\text{ m}$ (negligible).
- In 10 seconds, the error becomes $\frac{1}{2} (0.05) (10)^2 = 2.5\text{ meters}$.
- In 60 seconds, the error becomes $\frac{1}{2} (0.05) (60)^2 = \mathbf{90\text{ meters}}$!
- In 120 seconds, the error explodes to $\mathbf{360\text{ meters}}$!

And if the phone is tilted by just **$1.0^\circ$**, Earth's massive gravity vector ($9.81\text{ m/s}^2$) leaks into the forward direction with magnitude $9.81 \times \sin(1^\circ) \approx 0.17\text{ m/s}^2$, causing **$308\text{ meters}$ of drift in 60 seconds**!

---

### 3. What is the Fourier Transform (FFT) and Why Do We Care?
Imagine listening to a symphony. Your ear hears one combined sound wave, but your brain can distinguish the deep bass drum from the high-pitched violin.
The **Fourier Transform (FFT)** is a mathematical prism for signals:
- It takes a sensor signal that changes over time (time domain).
- It splits it into its individual musical "notes" (frequency domain).

When we analyzed our car data (Sequence S1 in Coventry, UK) with FFT, we discovered something beautiful:
- **Vehicle Driving Dynamics** (braking, turning, accelerating) happen slowly, between **$0.0\text{ Hz}$ and $1.5\text{ Hz}$** (less than 1.5 cycles per second).
- **Engine Vibration & Road Noise** happen much faster, between **$2.5\text{ Hz}$ and $5.0\text{ Hz}$**.

Because the noise and the driving maneuvers live in different frequency bands, we can use a **digital low-pass filter** to remove the noise without touching the car's real motion!

---

### 4. What is a Low-Pass Filter?
A **low-pass filter** is like an electronic sieve:
- Frequencies *lower* than the cutoff (e.g. $1.5\text{ Hz}$) pass through freely (preserving braking and turning).
- Frequencies *higher* than the cutoff are blocked and thrown away (eliminating engine buzz and road chatter).

We selected a **Butterworth Filter** because it has the flattest possible passband — it does not distort real driving maneuvers while cleanly wiping out high-frequency vibrations.

---

### 5. Why is "Too Much Filtering" Dangerous?
If you set the filter cutoff too low (for example, $0.5\text{ Hz}$):
- You will eliminate 98% of the vibration.
- BUT when the driver slams on the brakes, the sudden drop in acceleration looks like a fast change. A severe filter will smooth it away, thinking it was noise!
- The software will underestimate how hard the car stopped, causing the estimated car to drive right past the red light.

In Phase 3, we performed a careful measured sweep and proved that **$1.5\text{ Hz}$** is the scientific sweet spot: it cleans out **$92\%$ of vibration noise** while keeping **$95\%+$ of hard braking peaks** intact!

---

### 6. The "Causal" Rule: Why You Cannot Cheat Time on a Smartphone
In offline scientific research, engineers often use a trick called `filtfilt` (zero-phase filtering):
1. It filters the data from Monday to Friday.
2. Then it flips the file backward and filters from Friday to Monday.
This eliminates time delay completely.

**HOWEVER**: A smartphone driving on the highway in real time **does not know what the driver will do 2 seconds in the future!**
Presenting `filtfilt` as a smartphone solution is unrealistic.
In Phase 3, we strictly built **Mode B: Causal Real-Time Filtering**:
- It only uses past and present sensor readings.
- It calculates each output in **$0.012\text{ milliseconds}$** (fast enough to run 80,000 times per second).
- It runs with zero future data cheating.

---

### 7. What About Potholes and Speed Bumps? (The Hampel Filter)
When a car hits a pothole, the accelerometer registers a massive, violent spike for just 1 or 2 samples.
Standard linear filters would blur this spike across the next 10 samples, corrupting the speed calculation.
We implemented the **Hampel Filter**:
- It calculates the rolling median (the middle value) and Median Absolute Deviation (MAD).
- If an individual sample is more than $3\sigma$ away from its neighbors, it flags it as a pothole glitch and replaces it with the local median.
- Genuine step-changes (like hitting the brakes) are left completely untouched!

---

### 8. What's Next in Phase 4?
Phase 3 gives us clean, calibrated, vibration-free IMU signals.
However, because no classical filter can stop open-loop integration drift over 2 minutes, **Phase 4 will train an AI/ML neural network**.
Instead of integrating acceleration twice ($a \to v \to p$), the AI will look at a 3-second window of clean vibrations and **estimate the vehicle's forward speed directly**.
Integrating speed only once ($v \to p$) reduces error growth from quadratic ($t^2$) to linear ($t$), unlocking ISRO's target of **$< 10\%$ drift**.

---
*End of Simple Explanation Guide.*
