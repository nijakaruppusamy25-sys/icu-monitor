import os
import glob
import cv2
import numpy as np
import scipy.signal
import torch
import torch.nn as nn

# ==========================================================
# 1. EFFICIENTPHYS ARCHITECTURE
# ==========================================================

class AttentionMask(nn.Module):
    def forward(self, x):
        return 2.0 * x / (torch.sum(x, dim=(2, 3), keepdim=True) + 1e-7)

class TSM(nn.Module):
    def __init__(self, n_segment=72, fold_div=3):
        super(TSM, self).__init__()
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        fold = c // self.fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]
        return out.view(nt, c, h, w)

class EfficientPhys(nn.Module):
    def __init__(self, in_channels=3, frame_depth=72):
        super(EfficientPhys, self).__init__()
        self.tsm1 = TSM(frame_depth)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.Tanh())
        self.attn_conv = nn.Sequential(nn.Conv2d(32, 1, 1), nn.Sigmoid())
        self.attn_mask = AttentionMask()
        self.tsm2 = TSM(frame_depth)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.Tanh(), nn.AvgPool2d(2))
        self.tsm3 = TSM(frame_depth)
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.Tanh())
        self.tsm4 = TSM(frame_depth)
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.Tanh(), nn.AvgPool2d(2))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 1))

    def forward(self, x):
        B, C, T, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        x = self.conv1(self.tsm1(x))
        x = x * self.attn_mask(self.attn_conv(x))
        x = self.conv2(self.tsm2(x))
        x = self.conv3(self.tsm3(x))
        x = self.conv4(self.tsm4(x))
        return self.head(x).view(B, T)

# ==========================================================
# 2. ADAPTIVE SKIN ROI EXTRACTION (NO CASCADE DEPENDENCY)
# ==========================================================

def extract_adaptive_skin_roi(frame):
    h, w, _ = frame.shape
    # Focus on the upper body / facial candidate area
    upper_region = frame[int(h * 0.08):int(h * 0.55), int(w * 0.20):int(w * 0.80)]
    
    # YCrCb skin segmentation
    ycrcb = cv2.cvtColor(upper_region, cv2.COLOR_BGR2YCrCb)
    skin_mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
    
    # If skin pixels detected, crop tightly around largest skin cluster
    coords = cv2.findNonZero(skin_mask)
    if coords is not None and len(coords) > 200:
        x, y, crop_w, crop_h = cv2.boundingRect(coords)
        skin_roi = upper_region[y:y + crop_h, x:x + crop_w]
    else:
        # Fallback to centered upper face crop
        skin_roi = upper_region
        
    # Convert to RGB and resize for EfficientPhys (72x72)
    skin_rgb = cv2.cvtColor(skin_roi, cv2.COLOR_BGR2RGB)
    return cv2.resize(skin_rgb, (72, 72))

# ==========================================================
# 3. SPECTRAL HEART RATE ESTIMATION
# ==========================================================

def calculate_hr_psd(signal, fs=29.89):
    detrended = scipy.signal.detrend(signal)
    # Bandpass filter strictly constrained to human adult range: 55 - 115 BPM (0.91 - 1.91 Hz)
    low_cut = 0.91 / (fs / 2.0)
    high_cut = 1.91 / (fs / 2.0)
    b, a = scipy.signal.butter(2, [low_cut, high_cut], btype='bandpass')
    filtered = scipy.signal.filtfilt(b, a, detrended)
    
    # High-resolution Welch PSD
    freqs, psd = scipy.signal.welch(filtered, fs=fs, nperseg=min(len(filtered), 256), nfft=4096)
    valid_idx = np.where((freqs >= 0.91) & (freqs <= 1.91))[0]
    
    if len(valid_idx) == 0:
        return None
        
    peak_freq = freqs[valid_idx[np.argmax(psd[valid_idx])]]
    return peak_freq * 60.0

# ==========================================================
# 4. VIDEO INFERENCE PIPELINE
# ==========================================================

def predict_video_hr(video_path, model, device, chunk_len=72, stride=36):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        crop = extract_adaptive_skin_roi(frame)
        frames.append(crop.astype(np.float32))
    cap.release()

    if len(frames) < chunk_len + 1:
        return None

    frames = np.array(frames)
    # Normalized temporal frame difference: (f_{t+1} - f_t) / (f_{t+1} + f_t)
    diff = (frames[1:] - frames[:-1]) / (frames[1:] + frames[:-1] + 1e-6)
    diff = np.transpose(diff, (3, 0, 1, 2))  # (3, T, 72, 72)
    
    total_time = diff.shape[1]
    reconstructed_bvp = np.zeros(total_time)
    counts = np.zeros(total_time)
    
    with torch.no_grad():
        for i in range(0, total_time - chunk_len + 1, stride):
            clip = diff[:, i:i + chunk_len, :, :]
            tensor_clip = torch.from_numpy(clip).unsqueeze(0).float().to(device)
            out = model(tensor_clip).cpu().squeeze().numpy()
            reconstructed_bvp[i:i + chunk_len] += out
            counts[i:i + chunk_len] += 1.0

    valid_mask = counts > 0
    final_signal = reconstructed_bvp[valid_mask] / counts[valid_mask]
    return calculate_hr_psd(final_signal, fs=29.89)

# ==========================================================
# 5. BATCH EXECUTION RUNNER
# ==========================================================

if __name__ == "__main__":
    MODEL_PATH = "models/EfficientPhys_Baseline.pth"
    VIDEO_DIR = "data/raw/own"
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Compute Hardware: {device}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at '{MODEL_PATH}'")
        exit()
        
    model = EfficientPhys(in_channels=3, frame_depth=72).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    video_paths = sorted(glob.glob(os.path.join(VIDEO_DIR, "*_rgb.mp4")))
    
    print("\n" + "=" * 65)
    print("      REAL-SENSE ZERO-SHOT HEART RATE EVALUATION")
    print("=" * 65)

    for vid_path in video_paths:
        fname = os.path.basename(vid_path)
        # Skip infrared-only conditions that lack optical color channels
        if "dark_ir_only" in fname:
            print(f"[{fname:<38}] -> Skipped (No RGB illumination)")
            continue
            
        hr = predict_video_hr(vid_path, model, device)
        if hr is not None:
            print(f"[{fname:<38}] -> Estimated HR: {hr:6.2f} BPM")
        else:
            print(f"[{fname:<38}] -> Error: Insufficient frames")

    print("=" * 65)