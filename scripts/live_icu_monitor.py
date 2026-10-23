import argparse
import collections
import glob
import os
import sys
import time
import cv2
import numpy as np
import scipy.signal
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# 0. HARDWARE ACCELERATION ENGINE (MPS / CUDA / CPU)
# ---------------------------------------------------------------------------
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    DEVICE_STR = "Apple Silicon GPU (MPS)"
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DEVICE_STR = f"NVIDIA GPU ({torch.cuda.get_device_name(0)})"
else:
    DEVICE = torch.device("cpu")
    DEVICE_STR = "CPU"

print(f"[SYSTEM] Acceleration Engine: {DEVICE_STR}")

# Module-level rate-limiting constants expected by vitals_dashboard_server
HR_MAX_STEP_PER_SEC = 12.0
RR_MAX_STEP_PER_SEC = 4.0
EMA_ALPHA = 0.15

# ---------------------------------------------------------------------------
# 1. 1D-RESNET RESPIRATORY MODEL ARCHITECTURE
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

rr_ai_model = RespiratoryNet1D().to(DEVICE)
possible_rr_paths = [
    "models/best_realsense_rr_model.pth",
    "models/finetuned/rr_tcn_finetuned.pth",
    "best_realsense_rr_model.pth"
]
rr_model_loaded = False
for p in possible_rr_paths:
    if os.path.exists(p):
        try:
            state = torch.load(p, map_location=DEVICE)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            rr_ai_model.load_state_dict(state, strict=False)
            rr_ai_model.eval()
            print(f"[MODEL] Loaded Fine-Tuned Respiratory Checkpoint: {p}")
            rr_model_loaded = True
            break
        except Exception as e:
            print(f"[WARN] Failed loading {p}: {e}")

if not rr_model_loaded:
    print("[INFO] Running with real-time Metric Depth DSP engine.")

# ---------------------------------------------------------------------------
# 2. ANATOMICAL ROI LOCALIZATION
# ---------------------------------------------------------------------------
try:
    from ultralytics import YOLO
    pose_model = YOLO("yolov8n-pose.pt")
    HAS_YOLO = True
except Exception:
    HAS_YOLO = False
    pose_model = None

def detect_landmarks_yolo(img):
    if not HAS_YOLO or pose_model is None:
        return None
    results = pose_model(img, verbose=False, conf=0.4)
    if not results or len(results[0].keypoints) == 0:
        return None
    return results[0].keypoints.data[0].cpu().numpy()

def get_face_roi(kpts, img_shape, prev=None):
    h, w = img_shape[:2]
    if kpts is not None and kpts[0, 2] > 0.4:
        nx, ny = kpts[0, 0], kpts[0, 1]
        span = abs(kpts[3, 0] - kpts[4, 0]) if (kpts[3, 2] > 0.3 and kpts[4, 2] > 0.3) else w * 0.15
        curr = (
            int(max(0, nx - span * 0.55)),
            int(max(0, ny - span * 0.75)),
            int(min(w, nx + span * 0.55)),
            int(min(h, ny + span * 0.35))
        )
    else:
        curr = prev if prev else (int(w * 0.35), int(h * 0.25), int(w * 0.52), int(h * 0.50))

    if prev is not None:
        return tuple(int(0.3 * curr[i] + 0.7 * prev[i]) for i in range(4))
    return curr

def get_chest_roi_depth(depth_img, img_shape, prev=None):
    """
    Anatomical Sternal Thorax Extraction.
    Uses depth silhouette segmentation to lock firmly onto the sternum/chest,
    well below the chin and collarbone.
    """
    h, w = img_shape[:2]
    valid_mask = ((depth_img > 250) & (depth_img < 1600)).astype(np.uint8)
    cnts, _ = cv2.findContours(valid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if cnts:
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) > 2000:
            bx, by, bw, bh = cv2.boundingRect(c)

            # Center horizontally on torso
            cx1 = int(bx + bw * 0.26)
            cx2 = int(bx + bw * 0.74)

            # Center vertically on chest (54% to 85% of upper body height, below chin)
            cy1 = int(by + bh * 0.54)
            cy2 = int(by + bh * 0.85)

            curr = (max(0, cx1), max(0, cy1), min(w, cx2), min(h, cy2))
        else:
            curr = prev if prev else (int(w * 0.46), int(h * 0.56), int(w * 0.72), int(h * 0.86))
    else:
        curr = prev if prev else (int(w * 0.46), int(h * 0.56), int(w * 0.72), int(h * 0.86))

    if prev is not None:
        a = 0.25
        return tuple(int(a * curr[i] + (1.0 - a) * prev[i]) for i in range(4))
    return curr

# ---------------------------------------------------------------------------
# 3. COMPATIBILITY WRAPPERS FOR VITALS_DASHBOARD_SERVER
# ---------------------------------------------------------------------------
get_face_roi_from_kpts = get_face_roi

def get_chest_roi_from_kpts(kpts, depth_img, img_shape, prev=None):
    return get_chest_roi_depth(depth_img, img_shape, prev)

def update_smoothed_vital(current_smoothed, raw_val, rejected_count, max_jump, max_step, alpha=0.15):
    """Safe EMA Rate-Limiter with jump filter and deadlock escape protection."""
    if raw_val is None:
        return current_smoothed, rejected_count
    if current_smoothed is None:
        return raw_val, 0

    diff = raw_val - current_smoothed
    if abs(diff) > max_jump:
        rejected_count += 1
        if rejected_count >= 3:
            return raw_val, 0
        else:
            return current_smoothed + np.sign(diff) * max_step * alpha, rejected_count
    else:
        new_smoothed = (alpha * raw_val) + ((1.0 - alpha) * current_smoothed)
        return new_smoothed, 0

# ---------------------------------------------------------------------------
# 4. BIOMEDICAL DSP ENGINES (HEART RATE & RESPIRATION RATE)
# ---------------------------------------------------------------------------
def compute_hr_pos(rgb_buffer, fs=29.89):
    if len(rgb_buffer) < int(fs * 6):
        return None, "BUFFERING", (100, 100, 100), None

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

    safe_padlen = min(15, len(H) - 2)
    nyq = 0.5 * fs
    b, a = scipy.signal.butter(2, [0.70 / nyq, min(0.99, 3.50 / nyq)], btype='bandpass')
    bvp = scipy.signal.filtfilt(b, a, H, padlen=safe_padlen)

    freqs, psd = scipy.signal.welch(bvp, fs=fs, window='hann', nperseg=len(bvp), nfft=16384)
    v_idx = np.where((freqs >= 0.70) & (freqs <= 3.50))[0]
    if len(v_idx) < 3:
        return None, "NO PULSE", (100, 100, 100), bvp

    sub_freqs, sub_psd = freqs[v_idx], psd[v_idx]
    hr = sub_freqs[np.argmax(sub_psd)] * 60.0

    if hr < 50.0:
        status, color = "BRADYCARDIA", (0, 215, 255)
    elif hr > 115.0:
        status, color = "TACHYCARDIA", (0, 140, 255)
    else:
        status, color = "NORMAL SINUS", (0, 255, 0)

    return hr, status, color, bvp

def compute_rr(depth_buffer, fs=29.89):
    if len(depth_buffer) < int(fs * 6):
        return None, "BUFFERING", (100, 100, 100), None

    sig = np.array(depth_buffer, dtype=np.float32)
    detrended = scipy.signal.detrend(sig, type='linear')
    smooth_w = max(3, int(fs * 0.35))
    smoothed = np.convolve(detrended, np.ones(smooth_w) / smooth_w, mode='same')

    if np.ptp(smoothed) < 0.12:
        return 0.0, "CRITICAL: APNEA", (0, 0, 255), smoothed

    if rr_model_loaded and len(depth_buffer) >= 120:
        d_win = detrended[-120:]
        std_val = np.std(d_win)
        std_val = std_val if std_val > 1e-4 else 1.0
        norm_window = (d_win - np.mean(d_win)) / std_val
        inp_resamp = scipy.signal.resample(norm_window, 300)
        t_tensor = torch.from_numpy(inp_resamp).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
        with torch.no_grad():
            rr = float(rr_ai_model(t_tensor).item())
    else:
        freqs, psd = scipy.signal.welch(smoothed, fs=fs, window='hann', nperseg=len(smoothed), nfft=16384)
        v_idx = np.where((freqs >= 0.10) & (freqs <= 0.70))[0]
        if len(v_idx) >= 3:
            rr = freqs[v_idx][np.argmax(psd[v_idx])] * 60.0
        else:
            rr = 12.0

    if rr < 4.0:
        return 0.0, "CRITICAL: APNEA", (0, 0, 255), smoothed
    elif rr < 10.0:
        status, color = "WARNING: BRADYPNEA", (0, 215, 255)
    elif rr > 24.0:
        status, color = "WARNING: TACHYPNEA", (0, 140, 255)
    else:
        status, color = "NORMAL RESP", (0, 255, 0)

    return rr, status, color, smoothed

# ---------------------------------------------------------------------------
# 5. HUD COMPOSITOR & STANDALONE RUNNER
# ---------------------------------------------------------------------------
def draw_dashboard(rgb_frame, depth_frame, face_roi, chest_roi,
                   hr_val, hr_status, hr_color, bvp_wave,
                   rr_val, rr_status, rr_color, rr_wave,
                   fps, buf_fill):
    h, w = rgb_frame.shape[:2]
    fx1, fy1, fx2, fy2 = face_roi
    cv2.rectangle(rgb_frame, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
    cv2.putText(rgb_frame, "FACIAL rPPG (YOLO)", (fx1 + 5, max(20, fy1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    cx1, cy1, cx2, cy2 = chest_roi
    cv2.rectangle(depth_frame, (cx1, cy1), (cx2, cy2), (0, 255, 255), 2)
    cv2.putText(depth_frame, "STERNAL THORAX (DEPTH)", (cx1 + 5, max(20, cy1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    combined_views = np.hstack([rgb_frame, depth_frame])
    total_w = w * 2
    canvas = np.zeros((100 + h + 150, total_w, 3), dtype=np.uint8)
    canvas[100:100 + h, 0:total_w] = combined_views

    cv2.rectangle(canvas, (0, 0), (total_w, 100), (22, 22, 22), -1)
    hr_str = f"{hr_val:5.1f} BPM" if hr_val is not None else "--.- BPM"
    cv2.putText(canvas, "HEART RATE (rPPG)", (25, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
    cv2.putText(canvas, hr_str, (25, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
    cv2.rectangle(canvas, (250, 32), (450, 78), hr_color, -1)
    cv2.putText(canvas, hr_status, (260, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    cv2.putText(canvas, "VISION-ICU REPLAY MONITOR", (total_w // 2 - 130, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 220, 255), 1)
    cv2.putText(canvas, f"STREAM FPS: {fps:4.1f}", (total_w // 2 - 130, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)
    cv2.putText(canvas, f"BUFFER: {buf_fill:4.0%}", (total_w // 2 - 130, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

    rr_str = f"{rr_val:5.1f} BrPM" if rr_val is not None else "--.- BrPM"
    cv2.putText(canvas, "RESPIRATORY RATE (DEPTH)", (w + 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
    cv2.putText(canvas, rr_str, (w + 120, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
    cv2.rectangle(canvas, (w + 370, 32), (w + 580, 78), rr_color, -1)
    cv2.putText(canvas, rr_status, (w + 385, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    base_y = 100 + h
    cv2.rectangle(canvas, (0, base_y), (total_w, base_y + 150), (14, 14, 14), -1)
    cv2.putText(canvas, "PHOTOPLETHYSMOGRAM (CARDIAC BVP PULSE WAVE)", (20, base_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    if bvp_wave is not None and len(bvp_wave) > 10:
        pts = np.array(bvp_wave[-int(fps * 6):], dtype=np.float32)
        span = np.ptp(pts) if np.ptp(pts) > 1e-4 else 1.0
        y_scaled = (base_y + 45) - ((pts - np.mean(pts)) / span * 22.0)
        x_coords = np.linspace(20, total_w - 20, len(pts))
        curve = np.column_stack((x_coords, y_scaled)).astype(np.int32)
        cv2.polylines(canvas, [curve], False, (0, 255, 255), 2)

    cv2.putText(canvas, "PNEUMOGRAM (THORACIC METRIC DISPLACEMENT WAVE)", (20, base_y + 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    if rr_wave is not None and len(rr_wave) > 10:
        pts = np.array(rr_wave[-int(fps * 10):], dtype=np.float32)
        span = np.ptp(pts) if np.ptp(pts) > 1e-4 else 1.0
        y_scaled = (base_y + 120) - ((pts - np.mean(pts)) / span * 22.0)
        x_coords = np.linspace(20, total_w - 20, len(pts))
        curve = np.column_stack((x_coords, y_scaled)).astype(np.int32)
        cv2.polylines(canvas, [curve], False, (0, 255, 0), 2)

    return canvas

def run_recorded_monitor(rgb_path=None, depth_path=None):
    if not rgb_path or not depth_path:
        rgb_candidates = sorted(glob.glob("data/raw/own/*_rgb.mp4"))
        depth_candidates = sorted(glob.glob("data/raw/own/*_depth.mkv"))
        if not rgb_candidates or not depth_candidates:
            print("[ERROR] No recordings found in data/raw/own/")
            sys.exit(1)
        rgb_path = rgb_candidates[0]
        depth_path = depth_candidates[0]

    print(f"\n[REPLAY] RGB:   {rgb_path}")
    print(f"[REPLAY] Depth: {depth_path}\n")

    cap_rgb = cv2.VideoCapture(rgb_path)
    cap_depth = cv2.VideoCapture(depth_path)
    fps_video = cap_rgb.get(cv2.CAP_PROP_FPS) or 29.89
    buffer_len = int(fps_video * 10.0)
    rgb_buffer = collections.deque(maxlen=buffer_len)
    depth_buffer = collections.deque(maxlen=buffer_len)

    smoothed_hr, smoothed_rr = None, None
    rejected_hr_count, rejected_rr_count = 0, 0
    face_roi, chest_roi = None, None
    hr_val, hr_status, hr_color, bvp_wave = None, "BUFFERING", (100, 100, 100), None
    rr_val, rr_status, rr_color, rr_wave = None, "BUFFERING", (100, 100, 100), None

    frame_count = 0
    eval_counter = 0
    fps = fps_video
    fps_timer = time.time()

    try:
        while True:
            t0 = time.time()
            ret_c, rgb_img = cap_rgb.read()
            ret_d, d_frame_raw = cap_depth.read()

            if not ret_c or not ret_d:
                cap_rgb.set(cv2.CAP_PROP_POS_FRAMES, 0)
                cap_depth.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            depth_img = (
                cv2.cvtColor(d_frame_raw, cv2.COLOR_BGR2GRAY).astype(np.float32)
                if len(d_frame_raw.shape) == 3
                else d_frame_raw.astype(np.float32)
            )
            depth_display = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_display = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)

            if frame_count % 5 == 0:
                kpts_rgb = detect_landmarks_yolo(rgb_img)
                face_roi = get_face_roi(kpts_rgb, rgb_img.shape, face_roi)
                chest_roi = get_chest_roi_depth(depth_img, depth_img.shape, chest_roi)

            face_patch = rgb_img[face_roi[1]:face_roi[3], face_roi[0]:face_roi[2]]
            if face_patch.size > 0:
                rgb_buffer.append([float(np.mean(face_patch[:, :, 2])), float(np.mean(face_patch[:, :, 1])), float(np.mean(face_patch[:, :, 0]))])

            torso_patch = depth_img[chest_roi[1]:chest_roi[3], chest_roi[0]:chest_roi[2]]
            valid_d = torso_patch[torso_patch > 5]
            if len(valid_d) > 30:
                depth_buffer.append(float(np.mean(valid_d)))
            elif len(depth_buffer) > 0:
                depth_buffer.append(depth_buffer[-1])

            eval_counter += 1
            if eval_counter % int(fps_video) == 0:
                safe_fs = max(15.0, min(35.0, fps))
                raw_hr, hr_status, hr_color, bvp_wave = compute_hr_pos(rgb_buffer, fs=safe_fs)
                raw_rr, rr_status, rr_color, rr_wave = compute_rr(depth_buffer, fs=safe_fs)

                smoothed_hr, rejected_hr_count = update_smoothed_vital(
                    smoothed_hr, raw_hr, rejected_hr_count, 15.0, HR_MAX_STEP_PER_SEC
                )
                smoothed_rr, rejected_rr_count = update_smoothed_vital(
                    smoothed_rr, raw_rr, rejected_rr_count, 6.0, RR_MAX_STEP_PER_SEC
                )

                hr_val = round(smoothed_hr, 1) if smoothed_hr is not None else None
                rr_val = round(smoothed_rr, 1) if smoothed_rr is not None else None

            frame_count += 1
            if time.time() - fps_timer >= 1.0:
                fps = max(15.0, min(35.0, frame_count / (time.time() - fps_timer)))
                frame_count = 0
                fps_timer = time.time()

            buf_fill = len(depth_buffer) / buffer_len
            hud = draw_dashboard(rgb_img, depth_display, face_roi, chest_roi,
                                 hr_val, hr_status, hr_color, bvp_wave,
                                 rr_val, rr_status, rr_color, rr_wave,
                                 fps, buf_fill)
            cv2.imshow("VisionICU - Real-Time Replay Monitor", hud)

            elapsed = time.time() - t0
            sleep_dur = max(1, int(((1.0 / fps_video) - elapsed) * 1000))
            if cv2.waitKey(sleep_dur) & 0xFF in [ord('q'), 27]:
                break
    finally:
        cap_rgb.release()
        cap_depth.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VisionICU Video Replay Monitor")
    parser.add_argument("--rgb", type=str, default=None)
    parser.add_argument("--depth", type=str, default=None)
    args = parser.parse_args()
    run_recorded_monitor(args.rgb, args.depth)
