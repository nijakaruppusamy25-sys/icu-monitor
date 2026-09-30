import os
import re
import glob
import cv2
import numpy as np
import scipy.signal
import torch
import torch.nn as nn

# ==============================================================================
# 1. 1D-CNN ARCHITECTURE (LBRD-IC TRAINED MODEL)
# ==============================================================================

class Resp1DCNN(nn.Module):
    def __init__(self):
        super(Resp1DCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.squeeze(-1)
        return self.regressor(x)

# ==============================================================================
# 2. FEATURE EXTRACTION & SIGNAL RESAMPLING
# ==============================================================================

def parse_ground_truth(filename):
    """Parses nominal target rate from filename."""
    if "breathhold" in filename:
        return 0.0
    match = re.search(r'paced_(\d+)bpm', filename)
    if match:
        return float(match.group(1))
    return None

def extract_torso_green_signal(video_path):
    """Extracts continuous green channel luminance from thoracic region."""
    cap = cv2.VideoCapture(video_path)
    signals = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        h, w, _ = frame.shape
        torso = frame[int(h * 0.35):int(h * 0.85), int(w * 0.25):int(w * 0.75)]
        signals.append(np.mean(torso[:, :, 1]))
    cap.release()
    return np.array(signals, dtype=np.float32)

def predict_rr_lbrd(video_path, model, device):
    raw_sig = extract_torso_green_signal(video_path)
    if len(raw_sig) < 60:
        return None

    # Resample to 250 time steps (matching LBRD-IC input length)
    resampled = scipy.signal.resample(raw_sig, 250)
    std = np.std(resampled)
    norm_sig = (resampled - np.mean(resampled)) / (std if std > 1e-6 else 1.0)
    tensor_input = torch.tensor(norm_sig, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor_input).item()

    return max(0.0, output)

# ==============================================================================
# 3. EVALUATION RUNNER
# ==============================================================================

if __name__ == "__main__":
    MODEL_PATH = "models/best_resp_model.pth"
    VIDEO_DIR = "data/raw/own"

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Compute Hardware: {device}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Checkpoint not found at: {MODEL_PATH}")

    model = Resp1DCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    video_files = sorted(glob.glob(os.path.join(VIDEO_DIR, "*_rgb.mp4")))

    print("\n" + "=" * 78)
    print("      LBRD-IC PRETRAINED 1D-CNN EVALUATION ON REALSENSE DATA")
    print("=" * 78)
    print(f"{'Video File':<40} | {'Predicted':<12} | {'Target':<10} | {'Status'}")
    print("-" * 78)

    errors = []
    for v_path in video_files:
        fname = os.path.basename(v_path)
        if "dark_ir_only" in fname:
            continue

        pred_rr = predict_rr_lbrd(v_path, model, device)
        target_rr = parse_ground_truth(fname)

        if pred_rr is not None:
            if target_rr is not None:
                err = abs(pred_rr - target_rr)
                errors.append(err)
                status = f"Err: {err:4.1f} BrPM"
                print(f"{fname:<40} | {pred_rr:6.2f} BrPM | {target_rr:5.1f} BrPM | {status}")
            else:
                print(f"{fname:<40} | {pred_rr:6.2f} BrPM | {'N/A':<10} | Free/Rest")
        else:
            print(f"{fname:<40} | Error: Video too short")

    if errors:
        print("-" * 78)
        print(f"Mean Absolute Error (MAE) on Controlled Paced Trials: {np.mean(errors):.2f} BrPM")
    print("=" * 78)
