# ICU Vision Monitor: Non-Contact Physiological Vital Sign Monitoring

An end-to-end computer vision pipeline for non-contact physiological vital sign monitoring using RGB camera feeds and the Intel RealSense D435. The system extracts micro-capillary pulsatile blood volume changes (rPPG) from facial regions to estimate continuous Heart Rate (HR) using the deep temporal neural network architecture **EfficientPhys**.

---

## System Overview

```
RGB Video Stream (*_rgb.mp4)
│
▼
Adaptive YCrCb Skin ROI Segmentation (Forehead / Cheeks)
│
▼
Normalized Frame Differencing: (f_{t+1} - f_t) / (f_{t+1} + f_t)
│
▼
EfficientPhys Neural Network (2D-CNN + Temporal Shift Module)
│
▼
Reconstructed BVP Waveform via Overlap-Add (72-frame chunks)
│
▼
Bandpass Filtering (0.83–2.16 Hz) & Welch PSD Spectral Estimation
│
▼
Predicted Heart Rate (60–100 BPM)
```

---

## Directory Structure

```text
icu-vision-monitor/
├── data/
│   └── raw/
│       └── own/                     # RealSense dataset (.mp4, .csv, .json, .mkv)
├── models/
│   └── EfficientPhys_Baseline.pth   # Pretrained deep learning model weights
├── scripts/
│   ├── evaluate_hr_zero_shot.py     # Main neural HR batch evaluation script
│   └── evaluate_hr_pos.py           # Mathematical POS baseline comparison script
├── venv/                            # Local virtual environment
├── requirements.txt
└── README.md
```

---

## Key Hardware & Recording Formats

Each capture session with the Intel RealSense D435 produces synchronized multi-stream artifacts:

* **`*_rgb.mp4`**: Compressed 8-bit color video stream used for facial skin rPPG signal extraction.
* **`*_synclog_d435.csv`**: Hardware frame timestamps used to derive precise capture frequency ($f_s \approx 29.89\text{ Hz}$).
* **`*_meta.json`**: Session metadata recording camera parameters (exposure: `156`, white balance: `4600`, lighting).
* **`*_depth.mkv`**: 16-bit depth container recording metric distance.

---

## Installation & Setup

### 1. Clone Repository & Setup Environment

```bash
git clone <repository_url>
cd icu-vision-monitor

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install torch torchvision numpy scipy opencv-python pandas
```

### 3. Place Model Weights

Ensure model checkpoints are placed inside the `models/` directory:

```bash
mkdir -p models
# Move EfficientPhys_Baseline.pth into models/
```

---

## Running Inference

### Zero-Shot Deep Learning Heart Rate Evaluation

To run batch evaluation across all RealSense subjects using the `EfficientPhys` model:

```bash
python scripts/evaluate_hr_zero_shot.py
```

* **Compute Acceleration**: Automatically selects Apple Silicon Metal Performance Shaders (`mps`), NVIDIA `cuda`, or fallback `cpu`.
* **Output Range**: Normal resting physiological band ($60\text{--}100\text{ BPM}$).

### Classical Physics Baseline (POS Algorithm)

To run a comparison benchmark using the mathematical Plane-Orthogonal-to-Skin algorithm:

```bash
python scripts/evaluate_hr_pos.py
```

---

## Model Architecture Details

### EfficientPhys

* **Input**: 72-frame sliding window of normalized spatial color difference tensors of shape `[B, 3, 72, 72, 72]`.
* **Temporal Shift Module (TSM)**: Exchanges channel slices across adjacent temporal frames to capture zero-parameter temporal dynamics without 3D convolutions.
* **Attention Mechanism**: Spatial attention masks focus gradient weight on pulsatile micro-vascular skin areas while suppressing illumination shifts and head movements.
* **Post-Processing**: Overlap-add waveform reconstruction with Butterworth bandpass filtering ($50\text{--}130\text{ BPM}$) and high-resolution Welch Power Spectral Density (PSD) peak analysis.

---

## RealSense Evaluation Results

Summary of zero-shot `EfficientPhys` performance across diverse experimental conditions:

| Subject / Condition | Target Physiology | Model Output | Status |
| --- | --- | --- | --- |
| `sub02_upright_rest` | Resting baseline | **65.68 BPM** | Normal resting rate |
| `sub03_upright_rest` | Resting baseline | **71.37 BPM** | Normal resting rate |
| `sub04_upright_rest` | Resting baseline | **66.11 BPM** | Normal resting rate |
| `sub02_post_exercise` | Elevated cardio recovery | **95.89 BPM** | Elevated tracking |
| `sub05_post_exercise` | Elevated cardio recovery | **91.07 BPM** | Elevated tracking |
| `sub03_dark_ir_only` | Non-RGB capture | **Skipped** | Rejection of invalid optical feed |

---

## Requirements

* Python >= 3.10
* PyTorch >= 2.0.0
* OpenCV >= 4.8.0
* NumPy >= 1.24.0
* SciPy >= 1.10.0