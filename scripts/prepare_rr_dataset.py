import os
import re
import glob
import cv2
import numpy as np
import scipy.signal
import torch

def parse_target_bpm(filename):
    if "breathhold" in filename:
        return 0.0
    match = re.search(r'paced_(\d+)bpm', filename)
    if match:
        return float(match.group(1))
    return None

def extract_raw_green_signal(video_path):
    cap = cv2.VideoCapture(video_path)
    signals = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        h, w, _ = frame.shape
        # Focus strictly on upper chest/diaphragm area (higher SNR)
        torso = frame[int(h * 0.30):int(h * 0.75), int(w * 0.25):int(w * 0.75)]
        signals.append(np.mean(torso[:, :, 1]))
    cap.release()
    return np.array(signals, dtype=np.float32)

if __name__ == "__main__":
    VIDEO_DIR = "data/raw/own"
    OUTPUT_FILE = "data/rr_realsense_dataset.pt"
    
    video_files = sorted(glob.glob(os.path.join(VIDEO_DIR, "*_rgb.mp4")))
    samples, labels = [], []

    # 10-second window (300 frames @ ~30 FPS), 2-second stride (60 frames)
    WINDOW_SIZE = 300
    STRIDE = 60

    print("Extracting sliding-window 1D respiratory signals from RealSense recordings...")
    for v_path in video_files:
        fname = os.path.basename(v_path)
        if "dark_ir_only" in fname:
            continue
            
        target_bpm = parse_target_bpm(fname)
        if target_bpm is None:
            continue  # Skip unpaced trials during training
            
        sig = extract_raw_green_signal(v_path)
        if len(sig) < WINDOW_SIZE:
            continue

        # Extract overlapping 10-second chunks
        video_sample_count = 0
        for start_idx in range(0, len(sig) - WINDOW_SIZE + 1, STRIDE):
            chunk = sig[start_idx:start_idx + WINDOW_SIZE]
            
            # Resample chunk to exactly 250 points
            resampled = scipy.signal.resample(chunk, 250)
            
            # Z-score normalization
            std = np.std(resampled)
            norm_sig = (resampled - np.mean(resampled)) / (std if std > 1e-6 else 1.0)
            
            samples.append(norm_sig)
            labels.append(target_bpm)
            video_sample_count += 1
            
        print(f"Processed: {fname:<35} -> Generated {video_sample_count:2d} training chunks (Target: {target_bpm:4.1f} BrPM)")

    # Save augmented PyTorch dataset
    dataset = {
        "samples": torch.tensor(np.array(samples), dtype=torch.float32).unsqueeze(1),  # [N, 1, 250]
        "labels": torch.tensor(np.array(labels), dtype=torch.float32).unsqueeze(1)     # [N, 1]
    }
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    torch.save(dataset, OUTPUT_FILE)
    print(f"\nSUCCESS: Dataset augmented from ~35 to {len(samples)} samples! Saved to {OUTPUT_FILE}")