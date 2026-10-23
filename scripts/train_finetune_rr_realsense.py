"""
scripts/train_finetune_rr_realsense.py
Fine-tunes the 1D-ResNet Respiratory Rate model on Intel RealSense depth recordings.
Transfers weights from best_resp_model.pth and outputs best_realsense_rr_model.pth.
"""

import os
import re
import glob
import cv2
import numpy as np
import scipy.signal
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# 1. HARDWARE SELECTION (Apple Silicon MPS / NVIDIA CUDA / CPU)
# ---------------------------------------------------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("[DEVICE] Accelerated training on Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"[DEVICE] Accelerated training on NVIDIA GPU ({torch.cuda.get_device_name(0)})")
else:
    device = torch.device("cpu")
    print("[DEVICE] Running on CPU")

# ---------------------------------------------------------------------------
# 2. SIGNAL EXTRACTION & GROUND TRUTH PARSING
# ---------------------------------------------------------------------------
def parse_target_rr(filename: str) -> float:
    """Parses ground-truth respiratory rate from the file name."""
    if "breathhold" in filename:
        return 0.0

    match = re.search(r'paced_(\d+)bpm', filename)
    if match:
        seconds = float(match.group(1))
        return 60.0 / seconds if seconds > 0 else 0.0

    if "post_exercise" in filename:
        return 28.0

    if any(k in filename for k in ["upright_rest", "dim_light", "dark_ir", "two_people", "reclined"]):
        return 12.0

    return 12.0

def extract_chest_depth_signal(mkv_path: str) -> np.ndarray:
    """Extracts continuous thoracic millimeter distance z(t) from depth MKV."""
    cap = cv2.VideoCapture(mkv_path)
    timeline = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert depth frame to single channel
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray = frame.astype(np.float32)
            
        h, w = gray.shape

        # Sternal Thoracic ROI: 44% to 65% width, 54% to 82% height
        torso = gray[int(h * 0.54):int(h * 0.82), int(w * 0.44):int(w * 0.65)]
        valid = torso[torso > 5]
        
        if len(valid) > 30:
            timeline.append(float(np.mean(valid)))
        elif len(timeline) > 0:
            timeline.append(timeline[-1])

    cap.release()
    return np.array(timeline, dtype=np.float32)

# ---------------------------------------------------------------------------
# 3. PYTORCH DATASET WITH SLIDING WINDOW AUGMENTATION
# ---------------------------------------------------------------------------
class RealSenseRRDataset(Dataset):
    def __init__(self, data_dir="data/raw/own", window_size=300, step_size=20):
        """
        window_size = 300 frames (10.0 seconds at ~30 FPS)
        step_size   = 20 frames (sliding stride for data augmentation)
        """
        self.samples = []
        self.labels = []

        depth_files = sorted(glob.glob(os.path.join(data_dir, "*_depth.mkv")))
        print(f"[DATASET] Found {len(depth_files)} RealSense depth recordings in {data_dir}...")

        for fpath in depth_files:
            fname = os.path.basename(fpath)
            target_rr = parse_target_rr(fname)
            raw_sig = extract_chest_depth_signal(fpath)

            if len(raw_sig) < window_size:
                continue

            # Linear detrend across the entire recording
            detrended = scipy.signal.detrend(raw_sig, type='linear')

            # Create overlapping 10-second windows
            for start in range(0, len(detrended) - window_size + 1, step_size):
                window = detrended[start:start + window_size]
                
                # Z-score normalization per window
                std = np.std(window)
                std = std if std > 1e-4 else 1.0
                norm_window = (window - np.mean(window)) / std

                self.samples.append(norm_window)
                self.labels.append(target_rr)

        self.samples = np.array(self.samples, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.float32)
        print(f"[DATASET] Generated {len(self.samples)} training slices (window={window_size}, step={step_size}).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Shape: (1, 300) for 1D convolution
        x = torch.from_numpy(self.samples[idx]).unsqueeze(0)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y

# ---------------------------------------------------------------------------
# 4. 1D-RESNET ARCHITECTURE (LBRD-IC BENCHMARK COMPATIBLE)
# ---------------------------------------------------------------------------
class ResidualBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.conv(x))

class RespiratoryNet1D(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2)
        )

        self.layer1 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            ResidualBlock1D(64)
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            ResidualBlock1D(128)
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.global_pool(x).flatten(1)
        return self.regressor(x).squeeze(-1)

# ---------------------------------------------------------------------------
# 5. TRAINING & EVALUATION LOOP
# ---------------------------------------------------------------------------
def train_and_finetune(
    data_dir="data/raw/own",
    pretrained_path="models/best_resp_model.pth",
    output_model_path="models/best_realsense_rr_model.pth",
    epochs=35,
    batch_size=32,
    lr=1e-4
):
    dataset = RealSenseRRDataset(data_dir=data_dir, window_size=300, step_size=20)
    if len(dataset) == 0:
        print("[ERROR] No depth recordings found. Verify data/raw/own directory path.")
        return

    # 80% Train, 20% Validation Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = RespiratoryNet1D().to(device)

    # Transfer pre-trained weights if available
    alt_paths = [pretrained_path, "models/public/rr_tcn_lbrdic.pth", "best_resp_model.pth"]
    loaded = False
    for p in alt_paths:
        if os.path.exists(p):
            print(f"[CHECKPOINT] Loading pretrained weights from {p}...")
            state = torch.load(p, map_location=device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model_dict = model.state_dict()
            matched = {k: v for k, v in state.items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(matched)
            model.load_state_dict(model_dict)
            print(f"[CHECKPOINT] Transferred {len(matched)} / {len(model_dict)} layer tensors.")
            loaded = True
            break

    if not loaded:
        print("[WARN] Pretrained weights not found. Initializing network from scratch.")

    criterion = nn.L1Loss()  # Mean Absolute Error (MAE) loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)

    best_val_mae = float("inf")
    print("\n" + "=" * 65)
    print(f"  STARTING RESPIRATORY RATE FINE-TUNING ({epochs} EPOCHS)")
    print("=" * 65)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            preds = model(x_b)
            loss = criterion(preds, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y_b)

        train_mae = train_loss / len(train_ds)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                preds = model(x_b)
                loss = criterion(preds, y_b)
                val_loss += loss.item() * len(y_b)

        val_mae = val_loss / len(val_ds)
        scheduler.step(val_mae)

        saved_marker = ""
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
            torch.save(model.state_dict(), output_model_path)
            saved_marker = f"--> Saved: {output_model_path}"

        print(f"Epoch {epoch:2d}/{epochs:2d} | Train MAE: {train_mae:5.2f} BrPM | Val MAE: {val_mae:5.2f} BrPM {saved_marker}")

    print("=" * 65)
    print(f"[COMPLETE] Best Fine-Tuned RealSense RR Model: {best_val_mae:.2f} BrPM MAE")
    print(f"[SAVED] Checkpoint saved at: {output_model_path}\n")

if __name__ == "__main__":
    train_and_finetune()