import os
import re
import glob
import cv2
import numpy as np
import scipy.signal

def parse_ground_truth(filename):
    """Extracts target BPM from filename (e.g., paced_16bpm -> 16.0, breathhold -> 0.0)"""
    if "breathhold" in filename:
        return 0.0
    match = re.search(r'paced_(\d+)bpm', filename)
    if match:
        return float(match.group(1))
    return None

def extract_depth_respiration_rate(depth_mkv_path, fs=29.89):
    cap = cv2.VideoCapture(depth_mkv_path)
    depth_timeline = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # If frame is 3-channel grayscale representation of depth, use single channel
        if len(frame.shape) == 3:
            depth_map = frame[:, :, 0].astype(np.float32)
        else:
            depth_map = frame.astype(np.float32)

        h, w = depth_map.shape
        # Torso bounding box (center 50% width, mid-to-lower chest height)
        torso_depth = depth_map[int(h * 0.35):int(h * 0.80), int(w * 0.25):int(w * 0.75)]
        
        # Filter out 0 (invalid depth sensor dropouts)
        valid_depths = torso_depth[torso_depth > 0]
        
        if len(valid_depths) > 50:
            # Median depth eliminates edge/shadow noise artifacts
            depth_timeline.append(np.median(valid_depths))
        elif len(depth_timeline) > 0:
            depth_timeline.append(depth_timeline[-1])

    cap.release()

    if len(depth_timeline) < int(fs * 6):
        return None

    z_signal = np.array(depth_timeline, dtype=np.float32)

    # 1. Clean signal: Linear detrending (removes slow seated posture shifts)
    detrended_z = scipy.signal.detrend(z_signal)

    # 2. Check for Breath-Hold (flat displacement across full session)
    peak_to_peak_motion = np.ptp(detrended_z)
    if peak_to_peak_motion < 0.4:
        return 0.0

    # 3. Butterworth Bandpass Filter (0.10 Hz to 0.53 Hz -> 6 to 32 BrPM)
    low_cut = 0.10 / (fs / 2.0)
    high_cut = 0.53 / (fs / 2.0)
    b, a = scipy.signal.butter(2, [low_cut, high_cut], btype='bandpass')
    filtered_z = scipy.signal.filtfilt(b, a, detrended_z)

    # 4. Spectral Estimation via Welch Power Spectral Density
    freqs, psd = scipy.signal.welch(filtered_z, fs=fs, nperseg=min(len(filtered_z), 512), nfft=8192)
    valid_idx = np.where((freqs >= 0.10) & (freqs <= 0.53))[0]

    if len(valid_idx) == 0:
        return 0.0

    peak_idx = valid_idx[np.argmax(psd[valid_idx])]
    peak_power = psd[peak_idx]
    total_power = np.sum(psd[valid_idx])

    # If the spectral peak is indistinguishable from the noise floor, classify as apnea/breath-hold
    if (peak_power / (total_power + 1e-7)) < 0.08:
        return 0.0

    return freqs[peak_idx] * 60.0

if __name__ == "__main__":
    VIDEO_DIR = "data/raw/own"
    depth_files = sorted(glob.glob(os.path.join(VIDEO_DIR, "*_depth.mkv")))

    print("\n" + "=" * 75)
    print("      REALSENSE 16-BIT DEPTH RESPIRATORY RATE EVALUATION")
    print("=" * 75)
    print(f"{'Depth Video File':<38} | {'Predicted':<12} | {'Target':<10} | {'Status'}")
    print("-" * 75)

    errors = []
    for d_path in depth_files:
        fname = os.path.basename(d_path)
        pred_rr = extract_depth_respiration_rate(d_path, fs=29.89)
        target_rr = parse_ground_truth(fname)

        if pred_rr is not None:
            if target_rr is not None:
                err = abs(pred_rr - target_rr)
                errors.append(err)
                status = f"Err: {err:4.1f} BrPM"
                print(f"{fname:<38} | {pred_rr:6.2f} BrPM | {target_rr:5.1f} BrPM | {status}")
            else:
                print(f"{fname:<38} | {pred_rr:6.2f} BrPM | {'N/A':<10} | Free/Rest")
        else:
            print(f"{fname:<38} | Error: Video too short or empty")

    if errors:
        print("-" * 75)
        print(f"Mean Absolute Error (MAE) on Controlled Trials: {np.mean(errors):.2f} BrPM")
    print("=" * 75)