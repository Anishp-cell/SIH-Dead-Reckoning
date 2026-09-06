# Phase 3 Research Notes & Scientific References
## Digital Signal Processing, Inertial Navigation, and Machine-Learned Dead Reckoning

**Problem Statement**: SIH26168 (ISRO)  
**Track**: Track A / B Synthesis  

---

## 1. Classical Inertial Navigation & Error Propagation

1. **Titterton, D. H., & Weston, J. L. (2004).**  
   *Strapdown Inertial Navigation Technology (2nd ed.).* The Institution of Engineering and Technology.  
   - **Key takeaway**: Authoritative treatise on strapdown INS mechanization in navigation frames (ECEF/ENU). Sections 3.4–3.6 provide the analytical basis for specific-force gravity compensation and error growth equations ($e_p \propto \frac{1}{2} b_a t^2$).
   - **Relevance to SIH26168**: Governs our coordinate transformation pipeline from Body $\to$ Navigation frame and proves why open-loop inertial navigation inevitably requires external damping.

2. **Groves, P. D. (2013).**  
   *Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems (2nd ed.).* Artech House.  
   - **Key takeaway**: Detailed mathematical models for low-cost MEMS sensor stochastic errors, including temperature-dependent bias drift, in-run bias stability, and angular random walk (ARW).
   - **Relevance to SIH26168**: Direct foundation for our IMU generative measurement model in Module 4.

3. **Woodman, O. J. (2007).**  
   *An introduction to inertial navigation.* University of Cambridge, Computer Laboratory, Technical Report UCAM-CL-TR-696.  
   - **Key takeaway**: Clear tutorial derivation of orientation representation (Euler angles vs rotation vectors vs quaternions) and numerical integration pitfalls on low-cost consumer sensors.

---

## 2. Digital Signal Processing & Spectral Analysis

4. **Oppenheim, A. V., & Schafer, R. W. (2009).**  
   *Discrete-Time Signal Processing (3rd ed.).* Pearson.  
   - **Key takeaway**: Foundational theory on the Nyquist-Shannon sampling theorem, aliasing, bilinear transform, and Butterworth filter design.
   - **Relevance to SIH26168**: Governs our discrete Butterworth IIR filter implementations (`filter_design.py`) and explains why 10 Hz sampling creates aliased engine vibration bands.

5. **Welch, P. D. (1967).**  
   *The use of fast Fourier transform for the estimation of power spectra: A method based on time averaging over short, modified periodograms.* IEEE Transactions on Audio and Electroacoustics, 15(2), 70–73.  
   - **Key takeaway**: Derives the variance reduction properties of overlapping windowed segments for consistent power spectral density estimation.
   - **Relevance to SIH26168**: Direct implementation in `spectral_analysis.py` for characterization of vehicle driving regimes.

6. **Donoho, D. L., & Johnstone, I. M. (1994).**  
   *Ideal spatial adaptation by wavelet shrinkage.* Biometrika, 81(3), 425–455.  
   - **Key takeaway**: Formulates the universal threshold $\lambda = \hat{\sigma} \sqrt{2 \ln N}$ and proves asymptotic minimax optimality for denoising non-stationary signals via wavelet coefficient thresholding.
   - **Relevance to SIH26168**: Direct foundation for our PyWavelets multiresolution module (`wavelet_denoising.py`).

7. **Hampel, F. R. (1974).**  
   *The influence curve and its role in robust estimation.* Journal of the American Statistical Association, 69(346), 383–393.  
   - **Key takeaway**: Introduces the robust Median Absolute Deviation (MAD) dispersion scale factor $1.4826 \times \text{MAD}$ for outlier rejection resilient to 50% breakdown contamination.
   - **Relevance to SIH26168**: Deployed in `adaptive_filtering.py` to eradicate pothole sensor bit errors without blurring step maneuvers.

---

## 3. Learned Inertial Odometry & Vehicle Dead Reckoning

8. **Onyekpe, U., et al. (2021).**  
   *IO-VNBD: Inertial and Odometry Vehicle Navigation Benchmark Dataset.* Data in Brief / IEEE.  
   - **Key takeaway**: The official reference paper for our underlying dataset. Documents the Huawei P20 Pro smartphone mount, Racelogic VBOX DGPS ground truth, and Ford Fiesta CAN-bus ECU synchronization.

9. **Chen, C., et al. (2018).**  
   *IONet: Learning to Cure the Curse of Drift in Inertial Odometry.* IEEE Conference on Computer Vision and Pattern Recognition (CVPR).  
   - **Key takeaway**: First major paper proving that deep neural networks (Bi-LSTM) trained on sliding windows of IMU data can estimate displacement steps $\Delta \mathbf{p}$, reducing open-loop drift from $> 100\%$ to $< 5\%$.
   - **Relevance to SIH26168**: Strongly validates our Phase 4 target strategy (Target A and Target C).

10. **Brossard, M., Bonnabel, S., & Condomines, J. P. (2020).**  
    *AI-IMU Dead-Reckoning: Neural Network Augmented Inertial Odometry for Autonomous Driving.* IEEE Transactions on Robotics, 36(3), 661–675.  
    - **Key takeaway**: Proves that training a 1D Convolutional Neural Network (CNN) to predict dynamic covariance and velocity constraints directly eliminates dead-reckoning divergence during GNSS outages.
    - **Relevance to SIH26168**: Key architectural reference for our Phase 4 implementation.

---
*End of Research Notes.*
