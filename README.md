# Workbench

Version: v0.1.0

[![Python](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.57.0-FF4B4B.svg?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-6.7.0-3F4F75.svg?style=flat&logo=plotly&logoColor=white)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blueviolet.svg?style=flat)](/LICENSE)

Workbench is a specialized analytics studio for biomechanical gait analysis and radar signal processing. It serves as the primary analysis suite for data captured via the Craton Vision and Craton Radar pipelines, providing research-grade visualizations and kinematic extraction.

## Architecture

The workbench is structured to separate data processing logic from the interactive visualization layer:

*   **`core/logic.py`**: The central math engine. Handles DSP pipelines, biomechanical kinematic calculations, and radar centroid tracking.
*   **`core/main.py`**: The Streamlit-based dashboard logic. Manages state, interactive widgets, and Plotly visualizations.
*   **`core/config.cfg`**: Configuration storage for radar parameters and analysis thresholds.
*   **`app.py`**: Desktop entry point launcher. Manages the Streamlit server lifecycle, environment paths (`libs`), and automatic browser launching.

## Analysis Capabilities

### 1. DSP Pipeline
Pre-process raw motion data to ensure research-grade quality:
*   **Teleport Removal**: Purge signal jumps using Euclidean distance thresholds.
*   **Linear Interpolation**: Fill gaps in tracking data for continuous analysis.
*   **Signal Smoothing**: Center-aligned rolling mean filters for noise reduction.

### 2. Biomechanical Gait Analysis
Extract high-fidelity kinematics from vision-based skeletal data:
*   **Sagittal Trunk Lean**: Normalized calculation using shoulder/hip midpoints.
*   **Joint Kinematics**: Flexion/Extension tracking for Knee, Hip, and Shoulder.
*   **Dynamic ROM**: 30-frame rolling Peak-to-Peak Range of Motion evaluation.

### 3. Radar Dynamics
Process complex FMCW radar signals from Craton Radar:
*   **Micro-Doppler Spectrograms**: FFT-based visualization with clutter removal.
*   **Centroid Extraction**: Automatic motion tracking from binary radar data.
*   **Gait Metrics**: Extraction of cadence, step frequency, and symmetry.

## Installation & Usage

### Setup

1.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Execution

Run the workbench locally:
```bash
python app.py
```

### Build

To package the workbench as a standalone `x64` executable:

**Windows**:
```bash
pyinstaller --clean tools/windows/build.spec
```

**Linux**:
```bash
pyinstaller --clean tools/linux/build.spec
```

## Contribution & License

*   **License**: Distributed under the [Apache 2.0 License](/LICENSE).
*   **Contributions**: Pull requests are welcome. For major changes, please open an issue first. PRs containing unreviewed, generated AI content will be closed.
