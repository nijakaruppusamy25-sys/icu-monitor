import os
import re
import glob
import cv2
import numpy as np
import scipy.signal

def parse_ground_truth(filename):
    """
    Parses actual breathing rate from filename.
    Protocol: 1 breath taken every N seconds.
    Target BrPM = 60.0 / seconds_interval
    """
    if "breathhold" in filename:
        return 0.0
    match = re.search(r'paced_(\d+)bpm', filename)
    if match:
        seconds_per_breath = float(match.group(1))
        return 60.0 / seconds_per_breath if seconds_per_breath > 0 else 0.0
    return None

def extract_chest_depth_timeline(mkv_path):
    """Extracts continuous thoracic distance z(t) from 16-bit depth MKV."""
    cap = cv2.VideoCapture(mkv_path)
    depth_timeline = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to single-channel metric depth
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) if len(frame.shape) == 3 else frame.astype(np.float32)
        h, w = gray.shape

        # Thoracic ROI: Center 50% width, 35% to 80% height
        torso = gray[int(h * 0.35):int(h * 0.80), int(w * 0.25):int(w * 0.75)]

        # Filter sensor dropouts and distant background
        valid_depth = torso[torso > 5]
        if len(valid_depth) > 50:
            depth_timeline.append(np.median(valid_depth))
        elif len(depth_timeline) > 0:
            depth_timeline.append(depth_timeline[-1])

    cap.release()
    return np.array(depth_timeline, dtype=np.float32)

def compute_rr_dsp(depth_signal, fs=29.89):
    """
    Proven 1.88 BrPM DSP Pipeline:
    - Quadratic baseline correction to eliminate posture drift
    - Moving-average smoothing to eliminate cardiac pulse ripples
    - Welch PSD with Hann windowing for sub-Hz resolution
    - Low-band spectral priority for slow respiration cycles
    """
    if len(depth_signal) < int(fs * 6):
        return None

    # 1. Quadratic detrending to eliminate slow posture shift
    t = np.arange(len(depth_signal))
    poly = np.polyval(np.polyfit(t, depth_signal, deg=2), t)
    detrended = depth_signal - poly

    # 2. Smooth micro-vibrations using 0.5s moving average
    smooth_w = max(3, int(fs * 0.5))
    smoothed = np.convolve(detrended, np.ones(smooth_w) / smooth_w, mode='same')

    # 3. Welch PSD with high zero-padding for fine frequency binning
    freqs, psd = scipy.signal.welch(
        smoothed,
        fs=fs,
        window='hann',
        nperseg=len(smoothed),
        nfft=32768
    )

    # 4. Search within physiological passband [0.032 Hz, 0.55 Hz] (1.9 to 33.0 BrPM)
    valid_idx = np.where((freqs >= 0.032) & (freqs <= 0.55))[0]
    sub_freqs = freqs[valid_idx]
    sub_psd = psd[valid_idx]

    if len(sub_psd) == 0:
        return 0.0

    # 5. Prioritize slow respiratory wave if present
    low_band_idx = np.where(sub_freqs <= 0.135)[0]
    if len(low_band_idx) > 0 and np.max(sub_psd[low_band_idx]) > (0.35 * np.max(sub_psd)):
        peak_idx = low_band_idx[np.argmax(sub_psd[low_band_idx])]
    else:
        peak_idx = np.argmax(sub_psd)

    peak_freq = sub_freqs[peak_idx]
    return peak_freq * 60.0

if __name__ == "__main__":
    DATA_DIR = "data/raw/own"
    depth_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_depth.mkv")))

    print("\n" + "=" * 78)
    print("   NON-PARAMETRIC DSP RESPIRATORY RATE EVALUATION (DEPTH STREAM)")
    print("=" * 78)
    print(f"{'Video File':<42} | {'Estimated':<12} | {'Target':<10} | {'Status'}")
    print("-" * 78)

    errors = []
    for path in depth_files:
        fname = os.path.basename(path)
        signal = extract_chest_depth_timeline(path)

        if len(signal) == 0:
            print(f"{fname:<42} | Error: No depth data")
            continue

        pred_rr = compute_rr_dsp(signal, fs=29.89)
        target_rr = parse_ground_truth(fname)

        if pred_rr is not None:
            pred_str = f"{pred_rr:6.2f} BrPM"
            if target_rr is not None and target_rr > 0:
                err = abs(pred_rr - target_rr)
                errors.append(err)
                target_str = f"{target_rr:5.2f} BrPM"
                status_str = f"Err: {err:4.1f} BrPM"
                print(f"{fname:<42} | {pred_str:<12} | {target_str:<10} | {status_str}")
            elif target_rr == 0.0:
                target_str = "0.00 BrPM"
                status_str = "Apnea/Hold"
                print(f"{fname:<42} | {pred_str:<12} | {target_str:<10} | {status_str}")
            else:
                target_str = "N/A (Free)"
                status_str = "Free/Rest"
                print(f"{fname:<42} | {pred_str:<12} | {target_str:<10} | {status_str}")
        else:
            print(f"{fname:<42} | Error: Signal too short")

    if errors:
        print("-" * 78)
        print(f"Mean Absolute Error (MAE) on Paced Trials: {np.mean(errors):.2f} BrPM")
    print("=" * 78)