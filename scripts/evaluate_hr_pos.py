import os
import glob
import cv2
import numpy as np
import scipy.signal

# ==============================================================================
# 1. ADAPTIVE FACIAL SKIN SEGMENTATION (CORRECTS CROPPING MISTAKES)
# ==============================================================================

def extract_facial_skin_rgb(frame):
    """
    Extracts mean RGB values exclusively from skin pixels in the upper face,
    avoiding background, hair, and clothing contamination.
    """
    h, w, _ = frame.shape
    # Focus strictly on the upper half of the head region
    upper_face = frame[int(h * 0.08):int(h * 0.50), int(w * 0.25):int(w * 0.75)]

    # Segment skin via YCrCb color space
    ycrcb = cv2.cvtColor(upper_face, cv2.COLOR_BGR2YCrCb)
    # Standard physiological skin thresholds for varied lighting
    skin_mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))

    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    skin_mask = cv2.erode(skin_mask, kernel, iterations=1)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)

    skin_pixels = upper_face[skin_mask > 0]

    if len(skin_pixels) > 100:
        # Convert BGR to RGB (Fixes OpenCV channel swap mistake)
        mean_bgr = np.mean(skin_pixels, axis=0)
        return np.array([mean_bgr[2], mean_bgr[1], mean_bgr[0]], dtype=np.float32)
    else:
        # Fallback to center forehead region if mask drops out
        center_crop = cv2.cvtColor(upper_face, cv2.COLOR_BGR2RGB)
        return np.mean(center_crop, axis=(0, 1))

# ==============================================================================
# 2. PLANE-ORTHOGONAL-TO-SKIN (POS) CORE MATHEMATICS
# ==============================================================================

def compute_pos_pulse_wave(rgb_trace, fs=29.89):
    """
    Implements Wang et al. POS algorithm without trainable parameters.
    rgb_trace: numpy array of shape [3, T] (R, G, B channels across T frames)
    """
    T = rgb_trace.shape[1]
    window_length = int(fs * 1.6)  # 1.6-second optimal temporal window (~48 frames)

    if T <= window_length:
        return None

    h = np.zeros(T, dtype=np.float32)

    for t in range(T - window_length):
        # 1. Window spatial normalization
        chunk = rgb_trace[:, t:t + window_length]
        mean_intensity = np.mean(chunk, axis=1, keepdims=True)
        mean_intensity[mean_intensity == 0] = 1e-6
        normalized_chunk = chunk / mean_intensity

        # 2. Chrominance projection on orthogonal planes
        # S1: Green - Blue
        # S2: Green + Blue - 2*Red
        s1 = normalized_chunk[1, :] - normalized_chunk[2, :]
        s2 = normalized_chunk[1, :] + normalized_chunk[2, :] - (2.0 * normalized_chunk[0, :])

        # 3. Dynamic Alpha tuning based on standard deviation
        std_s1 = np.std(s1)
        std_s2 = np.std(s2)

        if std_s2 < 1e-6:
            alpha = 0.0
        else:
            alpha = std_s1 / std_s2

        # 4. Plane projection
        p = s1 + (alpha * s2)

        # 5. Overlap-add synthesis
        h[t:t + window_length] += (p - np.mean(p))

    return h

# ==============================================================================
# 3. SPECTRAL HEART RATE ESTIMATION
# ==============================================================================

def extract_heart_rate(pulse_wave, fs=29.89):
    """
    Applies zero-phase Butterworth filtering and Welch PSD estimation.
    """
    # Detrend to remove non-linear baseline drift
    detrended = scipy.signal.detrend(pulse_wave)

    # Butterworth bandpass filter for human cardiac range: 45 to 140 BPM (0.75 to 2.33 Hz)
    nyquist = 0.5 * fs
    low_cut = 0.75 / nyquist
    high_cut = 2.33 / nyquist
    b, a = scipy.signal.butter(2, [low_cut, high_cut], btype='bandpass')
    filtered = scipy.signal.filtfilt(b, a, detrended)

    # High-resolution Welch Power Spectral Density
    freqs, psd = scipy.signal.welch(filtered, fs=fs, nperseg=min(len(filtered), 256), nfft=8192)

    # Filter indices within physiological band
    valid_idx = np.where((freqs >= 0.75) & (freqs <= 2.33))[0]
    if len(valid_idx) == 0:
        return None

    dominant_freq = freqs[valid_idx[np.argmax(psd[valid_idx])]]
    heart_rate_bpm = dominant_freq * 60.0
    return heart_rate_bpm

# ==============================================================================
# 4. DATASET EXECUTION PIPELINE
# ==============================================================================

def process_realsense_video(video_path, fs=29.89):
    cap = cv2.VideoCapture(video_path)
    rgb_series = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb_mean = extract_facial_skin_rgb(frame)
        rgb_series.append(rgb_mean)

    cap.release()

    if len(rgb_series) < int(fs * 5):  # Must have at least 5 seconds of footage
        return None

    # Array shape: [3, T]
    rgb_arr = np.array(rgb_series, dtype=np.float32).T
    pulse = compute_pos_pulse_wave(rgb_arr, fs=fs)

    if pulse is None:
        return None

    return extract_heart_rate(pulse, fs=fs)

if __name__ == "__main__":
    DATA_DIR = "data/raw/own"
    video_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_rgb.mp4")))

    print("\n" + "=" * 70)
    print("      CALIBRATED POS ALGORITHM EVALUATION (OWN DATASET)")
    print("=" * 70)
    print(f"{'Video File Name':<42} | {'Estimated HR':<15}")
    print("-" * 70)

    for v_path in video_files:
        fname = os.path.basename(v_path)

        # Infrared/dark captures lack color dynamics needed for POS
        if "dark_ir_only" in fname:
            print(f"{fname:<42} | {'Skipped (No RGB)':<15}")
            continue

        estimated_hr = process_realsense_video(v_path, fs=29.89)

        if estimated_hr is not None:
            print(f"{fname:<42} | {estimated_hr:6.2f} BPM")
        else:
            print(f"{fname:<42} | {'Error/Too Short':<15}")

    print("=" * 70)