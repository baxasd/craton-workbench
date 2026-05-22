# Workbench

![Python](https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/streamlit-1.57.0-FF4B4B?logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/plotly-6.7.0-3F4F75?logo=plotly&logoColor=white)

Workbench is designed for biomechanical gait analysis and radar signal processing, extension for the Project Craton, to analyse the recordings of mmWave radar and RealSense x MediaPipe Pose recordings.    
Developed at the **University of Roehampton**

---

## Workflow


### DSP Pipeline
Ensure your raw motion data is research-ready before analysis.
*   **Teleport Removal**: Automatically detect and purge signal jumps based on Euclidean distance thresholds.
*   **Linear Interpolation**: Repair tracking losses and data gaps
*   **Signal Smoothing**: Apply center-aligned rolling mean filters to reduce high-frequency noise.
*   **Export**: Download the cleaned dataset as a CSV for external use.

### Gait Analysis
Generate postural and kinematic reports
*   **Saggital Trunk Lean**: Accurate Saggital lean calculation using hip/shoulder midpoints to cancel reciprocal twisting.
*   **Joint Kinematics**: High-fidelity angle calculations for Knee, Hip, Shoulder, and Elbow flexion.
*   **Dynamic ROM**: 30-frame rolling Peak-to-Peak analysis for Range of Motion evaluation.
*   **Temporal Grouping**: Export metrics at Frame-by-Frame, Second-by-Second, or Minute-by-Minute resolutions.

### Micro-Doppler
Process complex frequency-modulated continuous-wave (FMCW) radar signals.
*   **Spectrogram Generation**: Advanced FFT-based Micro-Doppler visualization with noise-floor clutter removal.
*   **Centroid Tracking**: Automatic extraction of human motion centroids from binary radar data.
*   **Gait Metrics**: Detection of Steps, Cadence (SPM), Symmetry Asymmetry, and Path Drift.
*   **Oscillation Dynamics**: Visualization of AC-coupled velocity components during gait.

---

## 🛠️ Technical Stack

| Component | Technology 
| :--- | :--- |
| **Engine** | Python 3.12
| **Frontend** | Streamlit
| **Visualization**| Plotly 
| **DSP** | SciPy
| **Data** | Pandas / PyArrow

---

## Installation

### Setup Environment
```bash
# Clone the repository
git clone https://github.com/baxasd/Workbench.git
cd Workbench

# Install dependencies
pip install -r requirements.txt
```

### Launch Workbench
Run the main entry point to start the local server:
```bash
python app.py
```
*The application is self-contained. Theme settings, server restrictions (localhost), and browser controls are managed automatically via the launcher.*

---

## ⚖️ License & Credits
Developed at the **University of Roehampton**.
Distributed under the MIT License. See `LICENSE.md` for details.
