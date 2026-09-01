import os
import glob
import cv2
import numpy as np
import scipy.signal

# ==========================================================
# 1. ADAPTIVE SKIN ROI EXTRACTION
# ==========================================================

def extract_skin_rgb(frame):
    h, w, _ = frame.shape
    # Focus on the upper body/facial area (forehead & cheeks)
    upper_region = frame[int(h * 0.08):int(h * 0.55), int(w * 0.20):int(w * 0.80)]
    
    # YCrCb skin thresholding
    ycrcb = cv2.cvtColor(upper_region, cv2.COLOR_BGR2YCrCb)
    skin_mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
    
    # Extract mean RGB across segmented skin pixels
    coords = cv2.findNonZero(skin_mask)
    if coords is not None and len(coords) > 200:
        x, y, crop_w, crop_h = cv2.boundingRect(coords)
        skin_roi = upper_region[y:y + crop_h, x:x + crop_w]
    else:
        skin_roi = upper_region

    # OpenCV BGR -> RGB spatial mean
    skin_rgb = cv2.cvtColor(skin_roi, cv2.COLOR_BGR2RGB)
    return np.mean(skin_rgb, axis=(0, 1))

# ==========================================================
# 2. PLANE-ORTHOGONAL-TO-SKIN (POS) ALGORITHM
# ==========================================================

def run_pos_algorithm(rgb_signal, fs=29.89):
    # rgb_signal shape: [3, T] -> R, G, B
    T = rgb_signal.shape[1]
    win_len = int(fs * 1.6)  # 1.6s window (~48 frames)
    h_bvp = np.zeros(T)

    for t in range(T - win_len):
        # 1. Temporal normalization within sliding window
        cn = rgb_signal[:, t:t + win_len]
        mean_c = np.mean(cn, axis=1, keepdims=True) + 1e-6
        cn_norm = cn / mean_c

        # 2. Orthogonal chrominance projection planes
        # S1 = G - B
        # S2 = G + B - 2R
        s1 = cn_norm[1, :] - cn_norm[2, :]
        s2 = cn_norm[1, :] + cn_norm[2, :] - 2.0 * cn_norm[0, :]

        # 3. Adaptive chrominance alpha tuning
        std_s1 = np.std(s1)
        std_s2 = np.std(s2)
        if std_s2 == 0:
            continue

        alpha = std_s1 / std_s2
        p = s1 + alpha * s2
        
        # Overlap-add pulse reconstruction
        h_bvp[t:t + win_len] += (p - np.mean(p))

    return h_bvp

# ==========================================================
# 3. SPECTRAL HEART RATE ESTIMATION
# ==========================================================

def calculate_hr_psd(bvp_signal, fs=29.89):
    detrended = scipy.signal.detrend(bvp_signal)
    
    # Butterworth bandpass filter (50 - 130 BPM -> 0.83 - 2.16 Hz)
    nyq = 0.5 * fs
    b, a = scipy.signal.butter(2, [0.83 / nyq, 2.16 / nyq], btype='bandpass')
    filtered = scipy.signal.filtfilt(b, a, detrended)

    # High-resolution Welch PSD
    freqs, psd = scipy.signal.welch(filtered, fs=fs, nperseg=min(len(filtered), 256), nfft=4096)
    valid_idx = np.where((freqs >= 0.83) & (freqs <= 2.16))[0]

    if len(valid_idx) == 0:
        return None

    peak_freq = freqs[valid_idx[np.argmax(psd[valid_idx])]]
    return peak_freq * 60.0

# ==========================================================
# 4. BATCH VIDEO PROCESSING PIPELINE
# ==========================================================

def evaluate_video_pos(video_path, fs=29.89):
    cap = cv2.VideoCapture(video_path)
    rgb_means = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        mean_rgb = extract_skin_rgb(frame)
        rgb_means.append(mean_rgb)
    cap.release()

    if len(rgb_means) < int(fs * 6):
        return None

    rgb_arr = np.array(rgb_means).T  # [3, T]
    bvp = run_pos_algorithm(rgb_arr, fs=fs)
    return calculate_hr_psd(bvp, fs=fs)

if __name__ == "__main__":
    VIDEO_DIR = "data/raw/own"
    video_paths = sorted(glob.glob(os.path.join(VIDEO_DIR, "*_rgb.mp4")))

    print("\n" + "=" * 70)
    print("      CLASSICAL POS ALGORITHM HEART RATE EVALUATION")
    print("=" * 70)

    for vid_path in video_paths:
        fname = os.path.basename(vid_path)
        if "dark_ir_only" in fname:
            print(f"[{fname:<38}] -> Skipped (No RGB illumination)")
            continue

        hr = evaluate_video_pos(vid_path, fs=29.89)
        if hr is not None:
            print(f"[{fname:<38}] -> POS Estimated HR: {hr:6.2f} BPM")
        else:
            print(f"[{fname:<38}] -> Error: Insufficient frames")

    print("=" * 70)