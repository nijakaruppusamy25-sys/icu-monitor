import argparse
import collections
import time
import cv2
import numpy as np
import scipy.signal

try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except ImportError:
    HAS_REALSENSE = False

def compute_rr_realtime(signal_buffer, fs=29.89):
    """
    Real-Time DSP Estimator on Rolling Buffer:
    - Linear detrending
    - Moving average smoothing (0.35s)
    - Welch PSD over physiological human respiratory passband [0.12 Hz, 0.55 Hz] (7.2 to 33.0 BrPM)
    """
    if len(signal_buffer) < int(fs * 6):
        return None, "BUFFERING", (100, 100, 100)

    sig = np.array(signal_buffer, dtype=np.float32)

    # 1. Linear detrend across the rolling window
    detrended = scipy.signal.detrend(sig, type='linear')

    # 2. Moving average smoothing
    smooth_w = max(3, int(fs * 0.35))
    smoothed = np.convolve(detrended, np.ones(smooth_w) / smooth_w, mode='same')

    # 3. Welch PSD across clinical respiratory band [0.12 Hz, 0.55 Hz] (7.2 to 33.0 BrPM)
    freqs, psd = scipy.signal.welch(
        smoothed,
        fs=fs,
        window='hann',
        nperseg=len(smoothed),
        nfft=16384
    )

    valid_idx = np.where((freqs >= 0.12) & (freqs <= 0.55))[0]
    sub_freqs = freqs[valid_idx]
    sub_psd = psd[valid_idx]

    if len(sub_psd) < 3:
        return None, "NO SIGNAL", (100, 100, 100)

    # 4. Extract dominant respiration peak
    peak_idx = np.argmax(sub_psd)
    peak_freq = sub_freqs[peak_idx]
    rr = peak_freq * 60.0

    # Clinical triaging
    if rr < 9.0:
        status, color = "WARNING: BRADYPNEA", (0, 215, 255)
    elif rr > 22.0:
        status, color = "WARNING: TACHYPNEA", (0, 140, 255)
    else:
        status, color = "NORMAL RESPIRATION", (0, 255, 0)

    return rr, status, color

def draw_hud(frame, roi_rect, depth_buffer, rr_val, status_str, status_color, fps, buffer_fill):
    h, w = frame.shape[:2]
    display = frame.copy()
    if len(display.shape) == 2:
        display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

    # 1. Thoracic ROI Box
    rx1, ry1, rx2, ry2 = roi_rect
    cv2.rectangle(display, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)
    cv2.putText(display, "CHEST ROI", (rx1 + 5, ry1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # 2. Header Telemetry Card
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, display, 0.25, 0, display)

    rr_display = f"{rr_val:5.1f} BrPM" if rr_val is not None else "--.- BrPM"
    cv2.putText(display, "RESPIRATION RATE", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
    cv2.putText(display, rr_display, (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)

    cv2.rectangle(display, (260, 20), (550, 70), status_color, -1)
    cv2.putText(display, status_str, (270, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.putText(display, f"FPS: {fps:.1f}", (w - 110, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(display, f"BUF: {buffer_fill:.0%}", (w - 110, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # 3. Dynamic Waveform Oscilloscope
    if len(depth_buffer) > 10:
        cv2.rectangle(overlay, (0, h - 90), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.75, display, 0.25, 0, display)

        pts = np.array(depth_buffer, dtype=np.float32)
        t_pts = np.arange(len(pts))
        pts_norm = pts - np.polyval(np.polyfit(t_pts, pts, 1), t_pts)

        span = np.ptp(pts_norm)
        if span < 1e-4:
            span = 1.0

        y_scaled = (h - 45) - (pts_norm / span * 30.0)
        x_coords = np.linspace(15, w - 15, len(pts))

        curve = np.column_stack((x_coords, y_scaled)).astype(np.int32)
        cv2.polylines(display, [curve], isClosed=False, color=(0, 255, 0), thickness=2)
        cv2.putText(display, "LIVE CHEST DISPLACEMENT OSCILLOSCOPE", (15, h - 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 255, 120), 1)

    return display

def run_stream(video_path=None):
    BUFFER_SEC = 15.0
    FPS_NOMINAL = 29.89
    buffer_len = int(BUFFER_SEC * FPS_NOMINAL)
    depth_buffer = collections.deque(maxlen=buffer_len)

    if video_path:
        print(f"[MODE] Replaying recorded stream: {video_path}")
        cap = cv2.VideoCapture(video_path)
        is_live = False
    else:
        if not HAS_REALSENSE:
            raise RuntimeError("pyrealsense2 is not installed. Run with --video <file.mkv>")
        print("[MODE] Initializing RealSense D435 Hardware...")
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        pipeline.start(config)
        is_live = True

    fps_timer = time.time()
    frame_count = 0
    fps = 30.0
    latest_rr = None
    latest_status = "BUFFERING"
    latest_color = (100, 100, 100)
    eval_counter = 0

    try:
        while True:
            t0 = time.time()
            if is_live:
                frames = pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                if not depth_frame:
                    continue
                depth_img = np.asanyarray(depth_frame.get_data(), dtype=np.float32)
            else:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                depth_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) if len(frame.shape) == 3 else frame.astype(np.float32)

            h, w = depth_img.shape

            # Centered over the sternum and ribcage
            roi_rect = (int(w * 0.44), int(h * 0.54), int(w * 0.65), int(h * 0.82))
            torso = depth_img[roi_rect[1]:roi_rect[3], roi_rect[0]:roi_rect[2]]

            # Discard dropped depth pixels
            valid = torso[torso > 5]
            if len(valid) > 30:
                depth_buffer.append(float(np.mean(valid)))
            elif len(depth_buffer) > 0:
                depth_buffer.append(depth_buffer[-1])

            # Recalculate RR every 15 frames (~0.5s)
            eval_counter += 1
            if eval_counter % 15 == 0:
                latest_rr, latest_status, latest_color = compute_rr_realtime(depth_buffer, fs=fps)

            frame_count += 1
            if time.time() - fps_timer >= 1.0:
                fps = frame_count / (time.time() - fps_timer)
                frame_count = 0
                fps_timer = time.time()

            buf_fill = len(depth_buffer) / buffer_len

            display_base = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            display_base = cv2.applyColorMap(display_base, cv2.COLORMAP_JET)

            hud_frame = draw_hud(display_base, roi_rect, depth_buffer, latest_rr,
                                 latest_status, latest_color, fps, buf_fill)

            cv2.imshow("VisionICU - Real-Time Respiratory Telemetry", hud_frame)

            if not is_live:
                elapsed = time.time() - t0
                sleep_dur = max(1, int((1.0 / 30.0 - elapsed) * 1000))
                key = cv2.waitKey(sleep_dur) & 0xFF
            else:
                key = cv2.waitKey(1) & 0xFF

            if key in [ord('q'), 27]:
                break
    finally:
        if is_live:
            pipeline.stop()
        else:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live ICU Respiratory Monitor")
    parser.add_argument("--video", type=str, default=None, help="Path to depth MKV file for playback simulation")
    args = parser.parse_args()
    run_stream(args.video)
