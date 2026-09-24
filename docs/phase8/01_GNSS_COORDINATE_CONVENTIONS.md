# Phase 8 Mathematical Foundation: GNSS Coordinate Conventions & Transformations

**Project**: Smart India Hackathon 2026 — Problem Statement SIH26168  
**Sponsoring Agency**: Indian Space Research Organisation (ISRO)  
**System**: AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation  
**Document**: `docs/phase8/01_GNSS_COORDINATE_CONVENTIONS.md`  

---

## 1. Executive Summary

This document establishes the authoritative coordinate transformation chain for GNSS position, velocity vectors, and heading measurements in Phase 8. All spatial transformations strictly map raw geodetic satellite telemetry (WGS84) into the local Cartesian East-North-Up (ENU) navigation frame used by the 15-state ESKF.

---

## 2. The WGS84 Reference Ellipsoid

The World Geodetic System 1984 (WGS84) reference ellipsoid is defined by:
- Semi-major axis (equatorial radius): $a = 6,378,137.0\text{ m}$
- Flattening: $f = \frac{1}{298.257223563}$
- Semi-minor axis (polar radius): $b = a(1 - f) \approx 6,356,752.3142\text{ m}$
- First eccentricity squared:
  $$e^2 = 2f - f^2 = \frac{a^2 - b^2}{a^2} \approx 6.69437999014 \times 10^{-3}$$

---

## 3. Geodetic (LLA) to Earth-Centered Earth-Fixed (ECEF)

Given geodetic latitude $\phi$, longitude $\lambda$, and ellipsoidal height $h$:

The prime vertical radius of curvature $N(\phi)$ is:
$$N(\phi) = \frac{a}{\sqrt{1 - e^2 \sin^2\phi}}$$

The Cartesian coordinates in the Earth-Centered Earth-Fixed (ECEF) frame are:
$$\begin{aligned}
X_{\text{ECEF}} &= (N(\phi) + h) \cos\phi \cos\lambda \\
Y_{\text{ECEF}} &= (N(\phi) + h) \cos\phi \sin\lambda \\
Z_{\text{ECEF}} &= \left(N(\phi)(1 - e^2) + h\right) \sin\phi
\end{aligned}$$

---

## 4. ECEF to Local Cartesian East-North-Up (ENU)

Let the local navigation tangent plane origin be defined by geodetic coordinates $(\phi_0, \lambda_0, h_0)$ with corresponding ECEF coordinates $(X_0, Y_0, Z_0)$.

The displacement vector in ECEF is:
$$\Delta\mathbf{r}_{\text{ECEF}} = \begin{bmatrix} X - X_0 \\ Y - Y_0 \\ Z - Z_0 \end{bmatrix}$$

The transformation matrix from ECEF to local ENU, $\mathbf{R}_{\text{enu/ecef}}$, is given by:
$$\mathbf{R}_{\text{enu/ecef}} = \begin{bmatrix}
-\sin\lambda_0 & \cos\lambda_0 & 0 \\
-\sin\phi_0 \cos\lambda_0 & -\sin\phi_0 \sin\lambda_0 & \cos\phi_0 \\
\cos\phi_0 \cos\lambda_0 & \cos\phi_0 \sin\lambda_0 & \sin\phi_0
\end{bmatrix}$$

The local ENU position vector $\mathbf{p}^n = [p_E, p_N, p_U]^T$ is:
$$\mathbf{p}^n = \mathbf{R}_{\text{enu/ecef}} \Delta\mathbf{r}_{\text{ECEF}}$$

### Official Coventry S1 Tangent Origin:
- $\phi_0 = 52.401660^\circ\text{ N}$
- $\lambda_0 = -1.505290^\circ\text{ E}$
- $h_0 = 147.50\text{ m}$

---

## 5. GNSS Velocity Transformation

When the GNSS receiver provides speed $v_{\text{horiz}}$, vertical velocity $v_U$, and ground course $\psi_{\text{GPS}}$:

$$\begin{aligned}
v_E &= v_{\text{horiz}} \sin(\psi_{\text{GPS}}) \\
v_N &= v_{\text{horiz}} \cos(\psi_{\text{GPS}}) \\
v_U &= v_U
\end{aligned}$$

The 3D velocity vector in the navigation frame is:
$$\mathbf{v}^n = \begin{bmatrix} v_E \\ v_N \\ v_U \end{bmatrix} \in \mathbb{R}^3$$

When the receiver outputs 3D velocity in ECEF $(\dot{X}, \dot{Y}, \dot{Z})$, the transformation is exact:
$$\mathbf{v}^n = \mathbf{R}_{\text{enu/ecef}} \begin{bmatrix} \dot{X} \\ \dot{Y} \\ \dot{Z} \end{bmatrix}$$

---

## 6. Heading Angle Conventions & Conversions

The system enforces an unambiguous mathematical conversion between geographic track azimuth and Cartesian ENU yaw:

1. **Geographic Azimuth ($\psi_{\text{GPS}}$)**:
   - Measured clockwise from True North:
     - North = $0^\circ = 0\text{ rad}$
     - East = $90^\circ = \frac{\pi}{2}\text{ rad}$
     - South = $180^\circ = \pi\text{ rad}$
     - West = $270^\circ = \frac{3\pi}{2}\text{ rad}$
   - Domain: $[0, 2\pi)$.

2. **Cartesian ENU Yaw ($\psi_{\text{ENU}}$)**:
   - Measured counter-clockwise from East:
     - East = $0\text{ rad}$
     - North = $+\frac{\pi}{2}\text{ rad}$
     - West = $\pm\pi\text{ rad}$
     - South = $-\frac{\pi}{2}\text{ rad}$
   - Domain: $(-\pi, +\pi]$.

3. **Authoritative Conversion Formulas**:
   $$\psi_{\text{ENU}} = \text{wrap}\left(\frac{\pi}{2} - \psi_{\text{GPS}}\right)$$
   $$\psi_{\text{GPS}} = \text{wrap}_{2\pi}\left(\frac{\pi}{2} - \psi_{\text{ENU}}\right)$$

---

## 7. Numerical Invariance & Round-Trip Validation

For any coordinate $\mathbf{p} = (\phi, \lambda, h)$, the round-trip conversion satisfies:
$$\|\text{enu\_to\_geodetic}(\text{geodetic\_to\_enu}(\mathbf{p})) - \mathbf{p}\| < 10^{-10}\text{ m}$$
Ensuring zero numerical drift across repeated coordinate transformations.
