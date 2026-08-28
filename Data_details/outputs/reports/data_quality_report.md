# Data Quality Report — IO-VNBD Phase 1

## Smartphone Data Quality

- **Total columns**: 27
- **Total null values**: 0 (0.00%)

### Columns with Extreme Outliers (>5σ from mean)

| Column | Outlier Count |
|--------|--------------|
| gps_acc_m | 91 |
| acc_x | 28 |
| acc_y | 37 |
| acc_z | 159 |
| grav_x | 193 |
| grav_y | 196 |
| grav_z | 212 |
| gyro_x | 173 |
| gyro_y | 8 |
| gyro_z | 112 |
| dt | 96 |

## Vehicle Data Quality

- **Total columns**: 32
- **Total null values**: 0 (0.00%)
- **Constant columns** (no variation): v_clutch

### Columns with Extreme Outliers (>5σ from mean)

| Column | Outlier Count |
|--------|--------------|
| v_vert_vel_kmh | 61 |
| v_sample_period_s | 2 |
| v_steering_deg | 470 |
| v_yaw_rate_degs | 12 |
| v_long_acc_g | 8 |
| v_lat_acc_g | 10 |
| v_handbrake | 311 |
| v_gear | 1423 |
| v_coolant_temp_c | 200 |
| v_brake_psi | 197 |
| v_battery_v | 2 |
| v_accel_pedal | 20 |
| dt | 2 |