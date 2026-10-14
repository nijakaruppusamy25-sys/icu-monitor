import argparse
import collections
import time
import cv2
import numpy as np
import scipy.signal

# ---------------------------------------------------------------------------
# 1. DEEP LEARNING MODEL: YOLOV8-POSE
# ---------------------------------------------------------------------------
try:
    from ultralytics import YOLO
    pose_model = YOLO("yolov8n-pose.pt")
    HAS_YOLO = True
except Exception as e:
    print(f"[WARN] YOLOv8-pose not available: {e}")
    HAS_YOLO = False
    pose_model = None

try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except ImportError:
    HAS_REALSENSE = False

def detect_landmarks_yolo(img):
    if not HAS_YOLO or pose_model is None: return None
    results = pose_model(img, verbose=False, conf=0.4)
    if not results or len(results[0].keypoints) == 0: return None
    return results[0].keypoints.data[0].cpu().numpy() 

def get_face_roi_from_kpts(kpts, img_shape, prev_roi=None):
    h, w = img_shape[:2]
    if kpts is not None and kpts[0, 2] > 0.4:
        nx, ny = kpts[0, 0], kpts[0, 1]
        if kpts[3, 2] > 0.3 and kpts[4, 2] > 0.3: span = abs(kpts[3, 0] - kpts[4, 0])
        elif kpts[1, 2] > 0.3 and kpts[2, 2] > 0.3: span = abs(kpts[1, 0] - kpts[2, 0]) * 2.2
        else: span = w * 0.15
        
        fx1, fx2 = int(max(0, nx - span * 0.55)), int(min(w, nx + span * 0.55))
        fy1, fy2 = int(max(0, ny - span * 0.75)), int(min(h, ny + span * 0.35))
        curr = (fx1, fy1, fx2, fy2)
    else:
        curr = prev_roi if prev_roi else (int(w * 0.35), int(h * 0.25), int(w * 0.52), int(h * 0.50))

    if prev_roi is not None:
        a = 0.3
        return tuple(int(a * curr[i] + (1 - a) * prev_roi[i]) for i in range(4))
    return curr

def get_chest_roi_from_kpts(kpts, depth_img, img_shape, prev_roi=None):
    h, w = img_shape[:2]
    if kpts is not None and kpts[5, 2] > 0.3 and kpts[6, 2] > 0.3:
        sh_l_x, sh_l_y = kpts[5, 0], kpts[5, 1]
        sh_r_x, sh_r_y = kpts[6, 0], kpts[6, 1]
        mid_x, mid_y = (sh_l_x + sh_r_x) / 2.0, (sh_l_y + sh_r_y) / 2.0
        span = abs(sh_l_x - sh_r_x)
        
        cx1, cx2 = int(max(0, mid_x - span * 0.28)), int(min(w, mid_x + span * 0.28))
        cy1, cy2 = int(max(0, mid_y + span * 0.10)), int(min(h, mid_y + span * 0.85))
        curr = (cx1, cy1, cx2, cy2)
    else:
        valid = depth_img[(depth_img > 10) & (depth_img < 1500)]
        if len(valid) > 200:
            d_min = np.percentile(valid, 8)
            mask = ((depth_img >= d_min - 20) & (depth_img <= d_min + 220)).astype(np.uint8)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                bx, by, bw, bh = cv2.boundingRect(max(cnts, key=cv2.contourArea))
                curr = (int(bx + bw * 0.30), int(by + bh * 0.45), int(bx + bw * 0.70), int(by + bh * 0.80))
            else:
                curr = prev_roi if prev_roi else (int(w * 0.56), int(h * 0.52), int(w * 0.78), int(h * 0.82))
        else:
            curr = prev_roi if prev_roi else (int(w * 0.56), int(h * 0.52), int(w * 0.78), int(h * 0.82))

    if prev_roi is not None:
        a = 0.3
        return tuple(int(a * curr[i] + (1 - a) * prev_roi[i]) for i in range(4))
    return curr

# ---------------------------------------------------------------------------
# 2. BIOMEDICAL DSP ENGINES
# ---------------------------------------------------------------------------
def compute_hr_pos(rgb_buffer, fs=29.89):
    if len(rgb_buffer) < int(fs * 6) or fs < 6.5:
        return None, "BUFFERING / LOW FPS", (100, 100, 100), None

    C = np.array(rgb_buffer, dtype=np.float32)
    N = len(C)
    w_size = max(10, int(fs * 1.6))
    H = np.zeros(N, dtype=np.float32)

    for m in range(N - w_size + 1):
        Cn = C[m:m + w_size]
        means = np.mean(Cn, axis=0) + 1e-6
        Cn_norm = Cn / means
        S1 = Cn_norm[:, 1] - Cn_norm[:, 2]
        S2 = Cn_norm[:, 1] + Cn_norm[:, 2] - 2 * Cn_norm[:, 0]
        alpha = (np.std(S1) + 1e-6) / (np.std(S2) + 1e-6)
        P = S1 + alpha * S2
        H[m:m + w_size] += (P - np.mean(P))

    nyq = 0.5 * fs
    low, high = 0.70 / nyq, min(0.99, 3.5 / nyq) # 42 to 210 BPM limits
    if low >= high: return None, "LOW FPS", (0, 0, 255), H

    b, a = scipy.signal.butter(2, [low, high], btype='bandpass')
    bvp = scipy.signal.filtfilt(b, a, H)

    freqs, psd = scipy.signal.welch(bvp, fs=fs, window='hann', nperseg=len(bvp), nfft=16384)
    v_idx = np.where((freqs >= 0.70) & (freqs <= 3.5))[0]
    
    if len(v_idx) < 3: return None, "NO PULSE", (100, 100, 100), bvp

    sub_freqs, sub_psd = freqs[v_idx], psd[v_idx]
    hr = sub_freqs[np.argmax(sub_psd)] * 60.0

    if hr < 50.0: status, color = "BRADYCARDIA", (0, 215, 255)
    elif hr > 115.0: status, color = "TACHYCARDIA", (0, 140, 255)
    else: status, color = "NORMAL SINUS", (0, 255, 0)

    return hr, status, color, bvp

def compute_rr_dsp(depth_buffer, fs=29.89):
    if len(depth_buffer) < int(fs * 6) or fs < 3.0:
        return None, "BUFFERING / LOW FPS", (100, 100, 100), None

    sig = np.array(depth_buffer, dtype=np.float32)
    detrended = scipy.signal.detrend(sig, type='linear')

    nyq = 0.5 * fs
    low, high = 0.06 / nyq, 0.70 / nyq # 3.6 to 42.0 BrPM limits
    if low < high:
        b, a = scipy.signal.butter(2, [low, high], btype='bandpass')
        filtered = scipy.signal.filtfilt(b, a, detrended)
    else:
        filtered = detrended

    smooth_w = max(3, int(fs * 0.35))
    smoothed = np.convolve(filtered, np.ones(smooth_w) / smooth_w, mode='same')

    freqs, psd = scipy.signal.welch(smoothed, fs=fs, window='hann', nperseg=len(smoothed), nfft=16384)
    v_idx = np.where((freqs >= 0.06) & (freqs <= 0.70))[0]
    
    if len(v_idx) < 3: return None, "NO SIGNAL", (100, 100, 100), smoothed

    sub_freqs, sub_psd = freqs[v_idx], psd[v_idx]
    rr = sub_freqs[np.argmax(sub_psd)] * 60.0

    if rr < 9.0: status, color = "BRADYPNEA", (0, 215, 255)
    elif rr > 22.0: status, color = "TACHYPNEA", (0, 140, 255)
    else: status, color = "NORMAL RESP", (0, 255, 0)

    return rr, status, color, smoothed

# ---------------------------------------------------------------------------
# 3. CONSOLE HUD COMPOSITOR
# ---------------------------------------------------------------------------
def draw_dashboard(rgb_frame, depth_frame, face_roi, chest_roi, hr_val, hr_status, hr_color, bvp_wave, rr_val, rr_status, rr_color, rr_wave, fps, buf_fill):
    h, w = rgb_frame.shape[:2]
    fx1, fy1, fx2, fy2 = face_roi
    cv2.rectangle(rgb_frame, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
    cv2.putText(rgb_frame, "FACIAL rPPG", (fx1 + 5, max(20, fy1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    cx1, cy1, cx2, cy2 = chest_roi
    cv2.rectangle(depth_frame, (cx1, cy1), (cx2, cy2), (0, 255, 255), 2)
    cv2.putText(depth_frame, "STERNAL THORAX", (cx1 + 5, max(20, cy1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    combined_views = np.hstack([rgb_frame, depth_frame])
    total_w = w * 2
    canvas = np.zeros((100 + h + 150, total_w, 3), dtype=np.uint8)
    canvas[100:100 + h, 0:total_w] = combined_views

    cv2.rectangle(canvas, (0, 0), (total_w, 100), (22, 22, 22), -1)

    hr_str = f"{hr_val:5.1f} BPM" if hr_val is not None else "--.- BPM"
    cv2.putText(canvas, "HEART RATE (rPPG)", (25, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
    cv2.putText(canvas, hr_str, (25, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
    cv2.rectangle(canvas, (240, 32), (430, 78), hr_color, -1)
    cv2.putText(canvas, hr_status, (250, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    cv2.putText(canvas, "VISION-ICU MONITOR", (total_w // 2 - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 220, 255), 1)
    cv2.putText(canvas, f"STREAM FPS: {fps:4.1f}", (total_w // 2 - 100, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)
    cv2.putText(canvas, f"WINDOW BUF: {buf_fill:4.0%}", (total_w // 2 - 100, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

    rr_str = f"{rr_val:5.1f} BrPM" if rr_val is not None else "--.- BrPM"
    cv2.putText(canvas, "RESPIRATORY RATE", (w + 140, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
    cv2.putText(canvas, rr_str, (w + 140, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
    cv2.rectangle(canvas, (w + 380, 32), (w + 580, 78), rr_color, -1)
    cv2.putText(canvas, rr_status, (w + 395, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    base_y = 100 + h
    cv2.rectangle(canvas, (0, base_y), (total_w, base_y + 150), (14, 14, 14), -1)

    cv2.putText(canvas, "PHOTOPLETHYSMOGRAM (CARDIAC BVP WAVE)", (20, base_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    if bvp_wave is not None and len(bvp_wave) > 10:
        pts = np.array(bvp_wave[-int(fps * 6):], dtype=np.float32)
        span = np.ptp(pts) if np.ptp(pts) > 1e-4 else 1.0
        y_scaled = (base_y + 45) - ((pts - np.mean(pts)) / span * 22.0)
        x_coords = np.linspace(20, total_w - 20, len(pts))
        curve = np.column_stack((x_coords, y_scaled)).astype(np.int32)
        cv2.polylines(canvas, [curve], False, (0, 255, 255), 2)

    cv2.putText(canvas, "PNEUMOGRAM (THORACIC DISPLACEMENT WAVE)", (20, base_y + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    if rr_wave is not None and len(rr_wave) > 10:
        pts = np.array(rr_wave[-int(fps * 10):], dtype=np.float32)
        span = np.ptp(pts) if np.ptp(pts) > 1e-4 else 1.0
        y_scaled = (base_y + 120) - ((pts - np.mean(pts)) / span * 22.0)
        x_coords = np.linspace(20, total_w - 20, len(pts))
        curve = np.column_stack((x_coords, y_scaled)).astype(np.int32)
        cv2.polylines(canvas, [curve], False, (0, 255, 0), 2)

    return canvas

# ---------------------------------------------------------------------------
# 4. STREAM EXECUTION LOOP
# ---------------------------------------------------------------------------
def run_icu_monitor(rgb_path=None, depth_path=None):
    BUFFER_SEC = 15.0
    FPS_NOMINAL = 29.89
    buffer_len = int(BUFFER_SEC * FPS_NOMINAL)

    rgb_buffer = collections.deque(maxlen=buffer_len)
    depth_buffer = collections.deque(maxlen=buffer_len)

    if rgb_path and depth_path:
        print(f"[REPLAY] Loading RGB: {rgb_path}")
        print(f"[REPLAY] Loading Depth: {depth_path}")
        cap_rgb = cv2.VideoCapture(rgb_path)
        cap_depth = cv2.VideoCapture(depth_path)
        is_live = False
    else:
        if not HAS_REALSENSE:
            raise RuntimeError("pyrealsense2 not installed.")
        print("[HARDWARE] Launching RealSense D435 Multi-Modal Pipeline...")
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        profile = pipeline.start(config)
        
        # Exposure lock to prevent rPPG oscillation artifacts
        color_sensor = profile.get_device().query_sensors()[1]
        color_sensor.set_option(rs.option.enable_auto_exposure, 0)
        color_sensor.set_option(rs.option.exposure, 156)
        color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
        color_sensor.set_option(rs.option.white_balance, 4600)

        align = rs.align(rs.stream.color)
        is_live = True

    fps_timer = time.time()
    frame_count = 0
    fps = 30.0

    face_roi = None
    chest_roi = None
    smoothed_hr = None
    hr_val, hr_status, hr_color, bvp_wave = None, "BUFFERING", (100, 100, 100), None
    rr_val, rr_status, rr_color, rr_wave = None, "BUFFERING", (100, 100, 100), None
    eval_counter = 0

    try:
        while True:
            t0 = time.time()
            if is_live:
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                c_frame = aligned_frames.get_color_frame()
                d_frame = aligned_frames.get_depth_frame()
                if not c_frame or not d_frame: continue
                rgb_img = np.asanyarray(c_frame.get_data())
                depth_img = np.asanyarray(d_frame.get_data(), dtype=np.float32)
            else:
                ret_c, rgb_img = cap_rgb.read()
                ret_d, d_frame_raw = cap_depth.read()
                if not ret_c or not ret_d:
                    cap_rgb.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    cap_depth.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                depth_img = cv2.cvtColor(d_frame_raw, cv2.COLOR_BGR2GRAY).astype(np.float32) if len(d_frame_raw.shape) == 3 else d_frame_raw.astype(np.float32)

            depth_display = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_display = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)

            if frame_count % 5 == 0:
                kpts_rgb = detect_landmarks_yolo(rgb_img)
                kpts_depth = detect_landmarks_yolo(depth_display)
                face_roi = get_face_roi_from_kpts(kpts_rgb, rgb_img.shape, face_roi)
                chest_roi = get_chest_roi_from_kpts(kpts_depth, depth_img, depth_img.shape, chest_roi)

            face_patch = rgb_img[face_roi[1]:face_roi[3], face_roi[0]:face_roi[2]]
            if face_patch.size > 0:
                rgb_buffer.append([float(np.mean(face_patch[:, :, 2])), float(np.mean(face_patch[:, :, 1])), float(np.mean(face_patch[:, :, 0]))])

            torso_patch = depth_img[chest_roi[1]:chest_roi[3], chest_roi[0]:chest_roi[2]]
            valid_d = torso_patch[torso_patch > 5]
            if len(valid_d) > 30: depth_buffer.append(float(np.mean(valid_d)))
            elif len(depth_buffer) > 0: depth_buffer.append(depth_buffer[-1])

            eval_counter += 1
            if eval_counter % 15 == 0:
                raw_hr, hr_status, hr_color, bvp_wave = compute_hr_pos(rgb_buffer, fs=fps)
                rr_val, rr_status, rr_color, rr_wave = compute_rr_dsp(depth_buffer, fs=fps)

                # EMA Smoothing for Heart Rate
                if raw_hr is not None:
                    if smoothed_hr is None: smoothed_hr = raw_hr
                    else: smoothed_hr = (0.1 * raw_hr) + (0.9 * smoothed_hr)
                    hr_val = smoothed_hr
                else:
                    hr_val = None

            frame_count += 1
            if time.time() - fps_timer >= 1.0:
                fps = frame_count / (time.time() - fps_timer)
                frame_count = 0
                fps_timer = time.time()

            buf_fill = max(len(rgb_buffer), len(depth_buffer)) / buffer_len
            hud = draw_dashboard(rgb_img, depth_display, face_roi, chest_roi, hr_val, hr_status, hr_color, bvp_wave, rr_val, rr_status, rr_color, rr_wave, fps, buf_fill)

            cv2.imshow("VisionICU - Real-Time Multi-Modal Telemetry Monitor", hud)

            if not is_live:
                elapsed = time.time() - t0
                sleep_dur = max(1, int((1.0 / 30.0 - elapsed) * 1000))
                key = cv2.waitKey(sleep_dur) & 0xFF
            else:
                key = cv2.waitKey(1) & 0xFF

            if key in [ord('q'), 27]: break
    finally:
        if is_live: pipeline.stop()
        else:
            cap_rgb.release()
            cap_depth.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Modal ICU Vitals Monitor")
    parser.add_argument("--rgb", type=str, default=None, help="Path to synchronized RGB video (.mp4)")
    parser.add_argument("--depth", type=str, default=None, help="Path to synchronized Depth video (.mkv)")
    args = parser.parse_args()
    run_icu_monitor(args.rgb, args.depth)