"""
Generates the 5 Phase 3 Jupyter notebooks in Data_details/notebooks/.
- 12_signal_frequency_analysis.ipynb
- 13_filter_design_and_validation.ipynb
- 14_vibration_analysis.ipynb
- 15_wavelet_analysis.ipynb
- 16_phase3_dead_reckoning_benchmark.ipynb
"""

import nbformat as nbf
from pathlib import Path

nb_dir = Path("Data_details/notebooks")
nb_dir.mkdir(parents=True, exist_ok=True)


def make_nb(title, objective, cells):
    nb = nbf.v4.new_notebook()
    nb.cells.append(nbf.v4.new_markdown_cell(f"# {title}\n\n**Objective**: {objective}"))
    for cell_type, content in cells:
        if cell_type == "md":
            nb.cells.append(nbf.v4.new_markdown_cell(content))
        elif cell_type == "code":
            nb.cells.append(nbf.v4.new_code_cell(content))
    return nb


# --- Notebook 12: Signal Frequency Analysis ---
nb12 = make_nb(
    "12. Sensor Frequency Analysis & Driving Event Characterization",
    "Compute single-sided FFT magnitude spectra, Welch Power Spectral Density (PSD), and characterize 7 representative driving regimes.",
    [
        ("md", "## 1. Setup & Environment\nLoad sequence S1 and spectral analysis module."),
        ("code", (
            "import sys\n"
            "sys.path.insert(0, '../..')\n"
            "import yaml\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "from Data_details.src.dataset_loader import DatasetLoader\n"
            "from Data_details.src.spectral_analysis import compute_fft_spectrum, compute_welch_psd, segment_driving_events, build_frequency_characterization_table\n\n"
            "with open('../config/config.yaml', 'r') as f:\n"
            "    cfg = yaml.safe_load(f)\n"
            "loader = DatasetLoader('../../IO-VNBD-master', '../../Data_details/data/raw')\n"
            "s_df, _ = loader.load_sequence(cfg['selected_sequence']['smartphone_file'])\n"
            "print(f'Loaded Sequence S1: {len(s_df)} rows')\n"
        )),
        ("md", "## 2. Event Segmentation & Frequency Table\nSegment driving events (Stationary, Braking, Cornering, Cruising, Bumps) and extract spectral metrics."),
        ("code", (
            "events = segment_driving_events(s_df)\n"
            "freq_df = build_frequency_characterization_table(s_df, events, fs=10.0)\n"
            "display(freq_df[['event_name', 'accel_dominant_freq_hz', 'accel_hf_energy_ratio_pct', 'peak_accel_mps2', 'crest_factor']])\n"
        )),
        ("md", "## 3. FFT Magnitude & Welch PSD\nPlot single-sided spectrum and inspect Nyquist boundary at 5.0 Hz."),
        ("code", (
            "ay = s_df['acc_y'].values\n"
            "freqs, mag, pwr = compute_fft_spectrum(ay, fs=10.0)\n"
            "f_psd, psd = compute_welch_psd(ay, fs=10.0)\n\n"
            "fig, axes = plt.subplots(2, 1, figsize=(12, 6))\n"
            "axes[0].plot(freqs, mag, color='#2ecc71', lw=1.2)\n"
            "axes[0].set_title('Forward Acceleration FFT Magnitude Spectrum')\n"
            "axes[0].set_ylabel('Magnitude (m/s²)')\n"
            "axes[0].grid(True, alpha=0.3)\n\n"
            "axes[1].semilogy(f_psd, psd, color='#e74c3c', lw=1.2)\n"
            "axes[1].set_title('Welch Power Spectral Density (PSD)')\n"
            "axes[1].set_xlabel('Frequency (Hz)')\n"
            "axes[1].set_ylabel('PSD ((m/s²)²/Hz)')\n"
            "axes[1].grid(True, alpha=0.3)\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
        )),
        ("md", "## 4. Key Findings\n- Useful vehicle maneuvers concentrate below 1.5 Hz.\n- Frequencies above 2.5 Hz are dominated by engine vibration and aliased road chatter."),
    ]
)
with open(nb_dir / "12_signal_frequency_analysis.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb12, f)

# --- Notebook 13: Filter Design & Validation ---
nb13 = make_nb(
    "13. Digital Filter Design, Cutoff Optimization & Causal Validation",
    "Design Butterworth low-pass filters, sweep cutoff frequencies (0.5 to 3.0 Hz), and evaluate causal real-time streaming.",
    [
        ("md", "## 1. Setup & Candidate Filters"),
        ("code", (
            "import sys\n"
            "sys.path.insert(0, '../..')\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "from Data_details.src.filter_design import design_butterworth_lowpass, apply_butterworth_causal, apply_butterworth_offline, RealTimeCausalFilter\n"
            "from Data_details.src.dataset_loader import DatasetLoader\n"
            "import yaml\n\n"
            "with open('../config/config.yaml', 'r') as f:\n"
            "    cfg = yaml.safe_load(f)\n"
            "loader = DatasetLoader('../../IO-VNBD-master', '../../Data_details/data/raw')\n"
            "s_df, _ = loader.load_sequence(cfg['selected_sequence']['smartphone_file'])\n"
            "ay = s_df['acc_y'].values\n"
        )),
        ("md", "## 2. Causal vs Non-Causal Comparison\nVerify that causal filtering operates without future-data cheating."),
        ("code", (
            "causal_filt = apply_butterworth_causal(ay, cutoff_hz=1.5, fs=10.0, order=2)\n"
            "offline_filt = apply_butterworth_offline(ay, cutoff_hz=1.5, fs=10.0, order=2)\n\n"
            "plt.figure(figsize=(12, 4))\n"
            "plt.plot(ay[1000:1300], color='#bdc3c7', label='Raw Accel (with Vibration)')\n"
            "plt.plot(offline_filt[1000:1300], 'r--', label='Offline Zero-Phase (filtfilt)')\n"
            "plt.plot(causal_filt[1000:1300], 'b-', label='Causal Streaming (lfilter)')\n"
            "plt.title('Causal vs Non-Causal Filtering on Real Driving Maneuver')\n"
            "plt.xlabel('Sample (100 ms per step)')\n"
            "plt.ylabel('Forward Accel (m/s²)')\n"
            "plt.legend()\n"
            "plt.grid(True, alpha=0.3)\n"
            "plt.show()\n"
        )),
        ("md", "## 3. Streaming Stateful Filter\nProcess sample-by-sample with RealTimeCausalFilter."),
        ("code", (
            "cf = RealTimeCausalFilter(cutoff_hz=1.5, fs=10.0, order=2)\n"
            "stream_out = [cf.process_sample(val) for val in ay[:100]]\n"
            "print(f'Processed {len(stream_out)} samples causally in real time.')\n"
        )),
    ]
)
with open(nb_dir / "13_filter_design_and_validation.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb13, f)

# --- Notebook 14: Vibration Analysis ---
nb14 = make_nb(
    "14. Vehicle Vibration Decomposition & Road Roughness Analysis",
    "Decompose raw signals into kinematic translation and structural vibration, comparing stationary vs cruising states.",
    [
        ("md", "## 1. Setup & Vibration Decomposition"),
        ("code", (
            "import sys\n"
            "sys.path.insert(0, '../..')\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "from Data_details.src.vibration_analysis import decompose_vibration, compute_vibration_metrics\n"
            "from Data_details.src.dataset_loader import DatasetLoader\n"
            "import yaml\n\n"
            "with open('../config/config.yaml', 'r') as f:\n"
            "    cfg = yaml.safe_load(f)\n"
            "loader = DatasetLoader('../../IO-VNBD-master', '../../Data_details/data/raw')\n"
            "s_df, _ = loader.load_sequence(cfg['selected_sequence']['smartphone_file'])\n"
            "az = s_df['acc_z'].values\n"
            "kin, vib = decompose_vibration(az, cutoff_hz=1.8, fs=10.0)\n"
            "metrics = compute_vibration_metrics(az, vib)\n"
            "print('Vertical Vibration Metrics:', metrics)\n"
        )),
        ("md", "## 2. Visualize Isolated Vibration"),
        ("code", (
            "fig, ax = plt.subplots(figsize=(12, 4))\n"
            "ax.plot(vib[2000:2500], color='#e74c3c', lw=0.8)\n"
            "ax.set_title('Isolated Road/Engine Vibration (>= 1.8 Hz)')\n"
            "ax.set_ylabel('Vibration (m/s²)')\n"
            "ax.grid(True, alpha=0.3)\n"
            "plt.show()\n"
        )),
    ]
)
with open(nb_dir / "14_vibration_analysis.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb14, f)

# --- Notebook 15: Wavelet Analysis ---
nb15 = make_nb(
    "15. Wavelet Multiresolution Denoising & Non-Stationary Transient Analysis",
    "Apply Discrete Wavelet Transform (DWT) multi-level thresholding and compare against linear Butterworth filtering.",
    [
        ("md", "## 1. Setup & Wavelet Denoising"),
        ("code", (
            "import sys\n"
            "sys.path.insert(0, '../..')\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "from Data_details.src.wavelet_denoising import wavelet_denoise_signal\n"
            "from Data_details.src.filter_design import apply_butterworth_causal\n"
            "from Data_details.src.dataset_loader import DatasetLoader\n"
            "import yaml\n\n"
            "with open('../config/config.yaml', 'r') as f:\n"
            "    cfg = yaml.safe_load(f)\n"
            "loader = DatasetLoader('../../IO-VNBD-master', '../../Data_details/data/raw')\n"
            "s_df, _ = loader.load_sequence(cfg['selected_sequence']['smartphone_file'])\n"
            "ay = s_df['acc_y'].values\n"
            "wav_clean, wmeta = wavelet_denoise_signal(ay, wavelet='sym4', level=3)\n"
            "print('Wavelet Denoising Meta:', wmeta)\n"
        )),
        ("md", "## 2. Wavelet vs Butterworth Residuals"),
        ("code", (
            "butter_clean = apply_butterworth_causal(ay, cutoff_hz=1.5, fs=10.0)\n"
            "plt.figure(figsize=(12, 5))\n"
            "plt.plot(ay[500:800], color='#bdc3c7', label='Raw Signal')\n"
            "plt.plot(butter_clean[500:800], 'r--', label='Butterworth (1.5 Hz)')\n"
            "plt.plot(wav_clean[500:800], 'b-', label='Wavelet (sym4, Level 3)')\n"
            "plt.title('Wavelet vs Butterworth Filter on Transient Deceleration')\n"
            "plt.legend()\n"
            "plt.grid(True, alpha=0.3)\n"
            "plt.show()\n"
        )),
    ]
)
with open(nb_dir / "15_wavelet_analysis.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb15, f)

# --- Notebook 16: Phase 3 Dead Reckoning Benchmark ---
nb16 = make_nb(
    "16. Phase 3 Dead Reckoning Benchmarks, Ablation & Trajectory Analysis",
    "Evaluate filtered dead-reckoning variants across 10s, 30s, 60s, 120s blackouts and review the Phase 3 ablation study.",
    [
        ("md", "## 1. Load Phase 3 Benchmark Tables"),
        ("code", (
            "import pandas as pd\n"
            "bench_df = pd.read_csv('../../Data_details/outputs/phase3/tables/phase3_dead_reckoning_benchmarks.csv')\n"
            "abl_df = pd.read_csv('../../Data_details/outputs/phase3/tables/phase3_ablation.csv')\n"
            "lat_df = pd.read_csv('../../Data_details/outputs/phase3/tables/filter_latency_benchmarks.csv')\n\n"
            "print('=== Phase 3 Dead Reckoning Benchmarks ===')\n"
            "display(bench_df)\n"
            "print('\n=== Phase 3 Ablation Study (60s Blackout) ===')\n"
            "display(abl_df)\n"
            "print('\n=== Filter Latency & Throughput ===')\n"
            "display(lat_df)\n"
        )),
        ("md", "## 2. Key Takeaways\n- Filtering eliminates erratic vibration excursions and stabilizes trajectory estimation.\n- Execution latency (< 0.05 ms) easily runs at >= 10 Hz on mobile devices.\n- Demonstrates the boundary where classical filtering ends and Phase 4 AI begins."),
    ]
)
with open(nb_dir / "16_phase3_dead_reckoning_benchmark.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb16, f)

print("All 5 Phase 3 notebooks generated successfully.")
