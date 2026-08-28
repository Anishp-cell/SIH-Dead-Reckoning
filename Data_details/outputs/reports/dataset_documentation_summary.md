# IO-VNBD Dataset Documentation Summary

## Source: Repository Facts

- **Full Name**: IO-VNBD — Inertial and Odometry Vehicle Navigation Benchmark Dataset
- **Repository**: https://github.com/onyekpeu/IO-VNBD
- **Folder Structure**:
  - `Synchronised V abd S datasets/` — Smartphone (S-) and Vehicle (V-) datasets collected simultaneously
    - `Categorised IOVNB Dataset/` — Organized by driver
    - `Uncategorised IOVNB Dataset/` — Flat S-Dataset/ and V-Dataset/ folders
  - `Unsynchronised V and S Dataset/` — Data collected separately (no time alignment)
  - `README.md` — Brief description
  - `README_1.pdf` — Full dataset paper

## Source: Paper Facts (IO-VNBD Paper)

### Equipment
- **Vehicle Data Logger**: Racelogic VBOX Video HD2 CAN-Bus data logger (10 Hz)
- **GPS Antenna**: Racelogic VBOX Video HD2 GPS Antenna (10 Hz), placed centrally on roof
- **Vehicle**: Ford Fiesta Titanium (front-wheel drive) for CAN-bus data
- **Smartphones**: Huawei P20 Pro, Motorola Moto G7 Power, BlackBerry Priv
- **Smartphone App**: AndroSensor (10 Hz sampling, GPS update ~1 Hz)

### Data Volume
- **Smartphone Data**: ~58 hours, ~4,400 km, ~2.2M records × 24 columns
- **Vehicle Data**: ~40 hours, ~1,300 km, ~1.4M records × 29 columns
- **Total**: ~98 hours, ~5,700 km across 8 drivers

### Collection Locations
- United Kingdom (Coventry, Nuneaton, etc.)
- Nigeria
- France

### Smartphone Sensor Columns (24 fields)
| # | Column | Unit |
|---|--------|------|
| 1 | GPS Latitude | degrees |
| 2 | GPS Longitude | degrees |
| 3 | GPS Altitude | m |
| 4 | GPS Speed | km/h |
| 5 | GPS Accuracy | m |
| 6 | GPS Orientation | degrees |
| 7 | GPS Satellites In Range | count |
| 8 | Time Since Start | ms |
| 9 | Date | YYYY-MO-DD HH-MI-SS_SSS |
| 10 | Accelerometer X | m/s² |
| 11 | Accelerometer Y | m/s² |
| 12 | Accelerometer Z | m/s² |
| 13 | Gravity X | m/s² |
| 14 | Gravity Y | m/s² |
| 15 | Gravity Z | m/s² |
| 16 | Gyroscope Yaw | rad/s |
| 17 | Gyroscope Pitch | rad/s |
| 18 | Gyroscope Roll | rad/s |
| 19 | Magnetic Field X | μT |
| 20 | Magnetic Field Y | μT |
| 21 | Magnetic Field Z | μT |
| 22 | Orientation Yaw/Azimuth | degrees |
| 23 | Orientation Pitch | degrees |
| 24 | Orientation Roll | degrees |

### Vehicle CAN-bus Columns (29 fields)
| # | Column | Unit |
|---|--------|------|
| 1 | No of GPS Satellites | count |
| 2 | Time Since Start of Day | seconds |
| 3 | Latitude | degrees |
| 4 | Longitude | degrees |
| 5 | Velocity | km/h |
| 6 | Heading | degrees |
| 7 | Height | km |
| 8 | Vertical Velocity | km/h |
| 9 | Sample Period | seconds |
| 10 | Steering Angle | degrees |
| 11-14 | Wheel Speeds (FL, FR, RL, RR) | rad/s |
| 15 | Yaw Rate | deg/s |
| 16 | Indicated Vehicle Speed | km/h |
| 17 | Longitudinal Acceleration | g |
| 18 | Lateral Acceleration | g |
| 19 | Handbrake | 0 or 1 |
| 20-21 | Gear Requested / Gear | 1-5 |
| 22 | Engine Speed | rev/min |
| 23 | Coolant Temperature | °C |
| 24 | Clutch Position | 0 or 1 |
| 25 | Brake Pressure | PSI |
| 26 | Brake Position | 0 or 1 |
| 27 | Battery Voltage | volts |
| 28 | Air Temperature | °C |
| 29 | Accelerator Pedal Position | % activation |

### Driving Scenarios
Hard braking, roundabouts, sharp turns, rain, hills, motorway, town centre, traffic, bumps, potholes, mud roads, parking, stationary, drifts, zig-zag, winding roads, U-turns, varying tyre pressures, and more.

### Drivers
| Driver | Style |
|--------|-------|
| A | Defensive |
| B | Defensive |
| C | Defensive |
| D | Defensive |
| E | Aggressive |
| F | Defensive |
| G | Defensive |
| H | Defensive |

### Synchronized vs Unsynchronized
- **Synchronized**: V- and S- datasets collected simultaneously from the same vehicle, manually time-aligned, stored in `Synchronised V abd S datasets/`.
- **Unsynchronized**: V- or S- datasets collected independently (different times, vehicles, or locations), stored in `Unsynchronised V and S Dataset/`.
- Not every V-file has a corresponding S-file and vice versa.

### Important Notes (from paper)
- Phone vibration interferes with acceleration measurement precision.
- Gravity readings are provided to help correct measured acceleration.
- GPS communication difficulties were encountered at times (documented in txt files).
- Direction of travel is positive X on the phone.