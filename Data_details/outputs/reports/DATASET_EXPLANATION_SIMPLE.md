# IO-VNBD Dataset — Simple Explanation

## What is this dataset?

IO-VNBD is a collection of driving data recorded from real cars on real roads. Researchers drove cars around cities in England, Nigeria, and France while two systems were recording data at the same time:

1. **A smartphone** strapped to the car dashboard (like you'd mount for Google Maps)
2. **A professional car data logger** connected to the car's computer (CAN bus)

Think of it as: the smartphone is the 'cheap sensor' we want to build our navigation system with, and the car's professional equipment gives us 'reference answers' to compare against.

---

## What does the smartphone record?

The smartphone records 24 measurements, 10 times every second:

- **GPS**: Where the phone thinks it is (latitude, longitude, altitude, speed, direction)
- **Accelerometer**: How much the phone is being pushed/pulled in 3 directions (forward-backward, left-right, up-down). This includes Earth's gravity.
- **Gyroscope**: How fast the phone is rotating (turning left, tilting forward, rolling sideways)
- **Magnetometer**: Compass readings in 3 directions (helps figure out which way is North)
- **Gravity Sensor**: Android's estimate of which direction gravity is pulling (helps separate gravity from actual vehicle movement)
- **Orientation**: The phone's best guess of its tilt angles (yaw/heading, pitch, roll)

---

## What does the car's computer record?

The car logger records 29 measurements, also 10 times per second:

- **GPS** (from a roof-mounted antenna — more accurate than the phone's GPS)
- **Wheel speeds** for each of the 4 wheels
- **Steering wheel angle**
- **Vehicle speed** from the speedometer
- **Yaw rate**: how fast the car is turning (from the car's own IMU sensor)
- **Acceleration**: forward/backward and side-to-side forces
- **Brake pressure and pedal position**
- **Engine speed (RPM), gear, clutch, accelerator pedal**
- **Battery voltage, coolant temp, air temperature**

---

## Why do we need this dataset?

Our SIH project needs to keep tracking a vehicle's position when GPS stops working (in tunnels, parking garages, under bridges, in urban canyons between tall buildings).

The idea is: when GPS disappears, use the smartphone's accelerometer and gyroscope to estimate 'where did the car go since GPS was last available?' This is called **dead reckoning** — like a sailor estimating position by speed and direction when they can't see the stars.

The problem: smartphone sensors are noisy and inaccurate. Even tiny errors in acceleration measurements get amplified when you integrate them to get velocity, and amplified again when you integrate velocity to get position. Errors grow very fast.

---

## What did we find in Phase 1?

### The dataset is large and diverse
- About 98 hours of driving, 5,700 km total
- 8 different drivers with different driving styles
- Many driving scenarios: roundabouts, hard braking, rain, motorways, bumpy roads, hills

### The sensors work at ~10 Hz as claimed
- The smartphone records a measurement roughly every 100 milliseconds (10 times/second)
- There is slight timing jitter but no major gaps in our selected test sequence

### Raw dead reckoning fails badly
- When we tried the simplest approach (just integrate acceleration → velocity → position), the estimated position drifts far from the true path within seconds.
- Even in a 60-second GPS blackout, the position error can reach hundreds of meters.
- The drift percentage is typically **far above the SIH target of <10%**.
- This confirms that **AI/ML and sophisticated algorithms are essential** — simple integration is nowhere near good enough.

### Why does raw dead reckoning fail?
1. **Sensor bias**: The accelerometer has a tiny constant offset. When integrated twice, this causes quadratic error growth (error grows as time²).
2. **Gravity leakage**: It's hard to perfectly separate Earth's gravity from vehicle motion acceleration. Any residual gravity directly becomes a large false acceleration.
3. **Phone orientation**: The phone is mounted at a slightly unknown angle. We don't perfectly know which direction each sensor axis points relative to the car.
4. **Noise amplification**: Random sensor noise, when integrated twice, turns into ever-growing random drift.
5. **Vibration**: Car vibrations (engine, bumps, potholes) add noise that looks like acceleration but isn't actual vehicle motion.

---

## What's next?

Phase 1 has shown us exactly *how bad* the raw approach is and *why*. This tells us exactly what Phase 2+ needs to fix:

1. **Calibrate the phone orientation** relative to the car
2. **Filter out vibration and noise** before integrating
3. **Estimate and remove sensor bias** using stationary periods
4. **Train an AI model** that learns to predict vehicle speed/motion from IMU patterns (much more robust than raw integration)
5. **Add physics constraints**: cars can't fly, can't slide sideways, have maximum possible speeds and turn rates
6. **Add map matching**: keep the estimated position on actual roads
7. **Fuse everything together** using a mathematical state estimator (Kalman Filter)

The SIH target of <10% drift (<5m error per 50m, <100m per 1km) is achievable with proper engineering, but requires all of these components working together.