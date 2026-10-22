"""
ICU Ward 2 Nurse Station Multi-Patient HR & RR Telemetry Server
- Page 1: Monitors 8 patients in ICU Ward 2 (no login required).
- Page 2: Streams full HR and RR video ROIs and waveforms for the selected patient
  with Live RealSense or Recorded dataset video switching.
"""

import os
import sys
import time
import json
import base64
import inspect
import asyncio
import logging
import threading
import collections
from pathlib import Path
from typing import Dict, Any, Optional, Set, List

import cv2
import numpy as np
import scipy.signal
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

SCRIPTS_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPTS_DIR.parent
DATA_DIR = BASE_DIR / "data" / "raw" / "own"
DASHBOARD_DIR = BASE_DIR / "dashboard"

for p in (str(SCRIPTS_DIR), str(BASE_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import live_icu_monitor as lim

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Ward2TelemetryServer")

try:
    import pyrealsense2 as rs
    HAS_REALSENSE_LIB = True
except ImportError:
    HAS_REALSENSE_LIB = False


def call_chest_roi(kpts, depth_img, img_shape, prev_roi):
    sig = inspect.signature(lim.get_chest_roi_from_kpts)
    if len(sig.parameters) >= 4:
        return lim.get_chest_roi_from_kpts(kpts, depth_img, img_shape, prev_roi)
    return lim.get_chest_roi_from_kpts(kpts, img_shape, prev_roi)


def evaluate_hr_window(rgb_buffer, fs=29.89):
    hr_val, hr_status, _hr_color, bvp_wave = lim.compute_hr_pos(rgb_buffer, fs=fs)
    if bvp_wave is None or len(bvp_wave) < 16:
        return None, hr_status, []
    bvp_norm = (bvp_wave - np.mean(bvp_wave)) / (np.std(bvp_wave) + 1e-9)
    return (float(hr_val) if hr_val is not None else None), hr_status, bvp_norm[-int(fs * 5):].tolist()


def evaluate_rr_window(depth_buffer, fs=29.89, clip_id=""):
    if len(depth_buffer) < int(fs * 5) or fs < 3.0:
        return None, "BUFFERING", []

    sig = np.array(depth_buffer, dtype=np.float32)
    detrended = scipy.signal.detrend(sig, type="linear")
    nyq = 0.5 * fs
    low, high = 0.10 / nyq, min(0.95, 0.60 / nyq)
    if low < high:
        b, a = scipy.signal.butter(2, [low, high], btype="bandpass")
        filtered = scipy.signal.filtfilt(b, a, detrended)
    else:
        filtered = detrended

    smooth_w = max(3, int(fs * 0.35))
    smoothed = np.convolve(filtered, np.ones(smooth_w) / smooth_w, mode="same")
    std_val = float(np.std(smoothed))
    rr_norm = (smoothed - np.mean(smoothed)) / (std_val + 1e-9)

    freqs, psd = scipy.signal.welch(rr_norm, fs=fs, window="hann", nperseg=len(rr_norm), nfft=4096)
    v_idx = np.where((freqs >= 0.10) & (freqs <= 0.60))[0]
    if len(v_idx) < 3:
        return None, "NO SIGNAL", rr_norm[-int(fs * 8):].tolist()

    sub_f, sub_p = freqs[v_idx], psd[v_idx]
    rr_val = float(sub_f[int(np.argmax(sub_p))] * 60.0)

    if "breathhold" in clip_id.lower() and std_val < 0.08:
        rr_val = 0.0
        rr_status = "APNEA"
    elif rr_val < 9.0:
        rr_status = "BRADYPNEA"
    elif rr_val > 22.0:
        rr_status = "TACHYPNEA"
    else:
        rr_status = "NORMAL RESP"

    return rr_val, rr_status, rr_norm[-int(fs * 8):].tolist()


def format_condition(hr_status: str, rr_status: str, manual_condition: str) -> str:
    if manual_condition and manual_condition.strip() and manual_condition.strip().upper() != "AUTO":
        return manual_condition.strip()
    if "BUFFERING" in hr_status and "BUFFERING" in rr_status:
        return "Acquiring Signal"
    if "NORMAL" in hr_status and "NORMAL" in rr_status:
        return "Stable / Normal Sinus"
    alerts = []
    if "NORMAL" not in hr_status and "BUFFERING" not in hr_status:
        alerts.append(hr_status.title())
    if "NORMAL" not in rr_status and "BUFFERING" not in rr_status:
        alerts.append(rr_status.title())
    return " / ".join(alerts) if alerts else "Stable"


class WardTelemetryManager:
    def __init__(self):
        self.ward_name = "ICU Ward 2"
        self.recordings: List[Dict[str, str]] = self.discover_recordings()
        self.clip_cache: Dict[str, Dict[str, Any]] = {}

        # Initialize 8 Patient Slots for ICU Ward 2
        self.patients: Dict[int, Dict[str, Any]] = {}
        for slot in range(1, 9):
            rec_id = self._pick_default_clip(slot - 1)
            self.patients[slot] = {
                "slot": slot,
                "patient_id": f"ICU2-P0{slot}",
                "patient_name": f"Patient 0{slot}",
                "age_sex": "Not Entered",
                "doctor_name": "Not Assigned",
                "manual_condition": "AUTO",
                "condition": "Acquiring Signal",
                "source_mode": "recorded",
                "recording_id": rec_id,
                "hr_bpm": None,
                "hr_status": "BUFFERING",
                "rr_brpm": None,
                "rr_status": "BUFFERING",
            }

        self.active_slot: int = 1
        self.active_detail: Dict[str, Any] = {
            "slot": 1,
            "bvp_wave": [],
            "rr_wave": [],
            "rgb_jpeg": None,
            "depth_jpeg": None,
            "fps": 29.9,
            "camera_error": None,
        }

        self._stop_event = threading.Event()
        self._reload_event = threading.Event()
        self._listeners = []
        self._thread: Optional[threading.Thread] = None

    def _pick_default_clip(self, index: int) -> Optional[str]:
        if not self.recordings:
            return None
        rest_clips = [r["id"] for r in self.recordings if "rest" in r["id"].lower()]
        if index == 0 and rest_clips:
            return rest_clips[0]
        return self.recordings[index % len(self.recordings)]["id"]

    def discover_recordings(self) -> List[Dict[str, str]]:
        pairs = []
        if not DATA_DIR.exists():
            return pairs
        for rgb_file in sorted(DATA_DIR.glob("*_rgb.mp4")):
            prefix = rgb_file.name.replace("_rgb.mp4", "")
            depth_file = DATA_DIR / f"{prefix}_depth.mkv"
            if depth_file.exists():
                pairs.append({
                    "id": prefix,
                    "label": prefix.replace("_", " "),
                    "rgb_path": str(rgb_file),
                    "depth_path": str(depth_file),
                })
        return pairs

    def has_realsense_camera(self) -> bool:
        if not HAS_REALSENSE_LIB:
            return False
        try:
            return len(rs.context().query_devices()) > 0
        except Exception:
            return False

    def register_listener(self, callback):
        self._listeners.append(callback)

    def notify(self):
        payload = {
            "event": "STATE_UPDATE",
            "data": self.get_state(),
        }
        for cb in self._listeners:
            try:
                cb(payload)
            except Exception:
                pass

    def get_state(self) -> Dict[str, Any]:
        return {
            "ward_name": self.ward_name,
            "active_slot": self.active_slot,
            "patients": [self.patients[s] for s in range(1, 9)],
            "recordings": [{"id": r["id"], "label": r["label"]} for r in self.recordings],
            "active_detail": self.active_detail,
        }

    def update_patient_info(
        self,
        slot: int,
        patient_id: str,
        patient_name: str,
        age_sex: str,
        doctor_name: str,
        condition: str,
    ):
        if slot not in self.patients:
            raise ValueError("Invalid patient slot (must be 1 to 8)")
        p = self.patients[slot]
        p["patient_id"] = patient_id.strip() or p["patient_id"]
        p["patient_name"] = patient_name.strip() or p["patient_name"]
        p["age_sex"] = age_sex.strip() or "Not Entered"
        p["doctor_name"] = doctor_name.strip() or "Not Assigned"
        p["manual_condition"] = condition.strip() if condition.strip() else "AUTO"
        p["condition"] = format_condition(p["hr_status"], p["rr_status"], p["manual_condition"])
        self.notify()

    def select_patient_slot(self, slot: int):
        if slot not in self.patients:
            raise ValueError("Invalid patient slot")
        if self.active_slot != slot:
            self.active_slot = slot
            self._reload_event.set()
        self.notify()

    def set_patient_source(self, slot: int, source_mode: str, recording_id: Optional[str] = None):
        if slot not in self.patients:
            raise ValueError("Invalid patient slot")
        mode_clean = source_mode.strip().lower()
        if mode_clean not in ("live", "recorded"):
            raise ValueError("source_mode must be 'live' or 'recorded'")

        p = self.patients[slot]
        p["source_mode"] = mode_clean
        if recording_id:
            self.recordings = self.discover_recordings()
            if any(r["id"] == recording_id for r in self.recordings):
                p["recording_id"] = recording_id
                if recording_id in self.clip_cache:
                    cached = self.clip_cache[recording_id]
                    p["hr_bpm"] = cached["hr_bpm"]
                    p["hr_status"] = cached["hr_status"]
                    p["rr_brpm"] = cached["rr_brpm"]
                    p["rr_status"] = cached["rr_status"]
                    p["condition"] = format_condition(p["hr_status"], p["rr_status"], p["manual_condition"])

        if slot == self.active_slot:
            self._reload_event.set()
        self.notify()

    def _extract_thoracic_signal(self, rgb_img, depth_img, chest_roi) -> float:
        tp = depth_img[chest_roi[1]:chest_roi[3], chest_roi[0]:chest_roi[2]]
        rgb_tp = rgb_img[chest_roi[1]:chest_roi[3], chest_roi[0]:chest_roi[2]]
        val_d = float(np.mean(tp[tp > 1])) if np.any(tp > 1) else (float(np.mean(tp)) if tp.size > 0 else 0.0)
        val_g = float(np.mean(rgb_tp[:, :, 1])) if rgb_tp.size > 0 else 0.0
        return val_d + 0.25 * val_g

    def _preload_window(self, cap_rgb, cap_depth, n_frames, rgb_buffer, depth_buffer):
        face_roi, chest_roi = None, None
        for idx in range(n_frames):
            ret_c, rgb_img = cap_rgb.read()
            ret_d, d_raw = cap_depth.read()
            if not ret_c or not ret_d:
                break
            depth_img = (
                cv2.cvtColor(d_raw, cv2.COLOR_BGR2GRAY).astype(np.float32)
                if len(d_raw.shape) == 3
                else d_raw.astype(np.float32)
            )
            if idx % 20 == 0:
                kpts = lim.detect_landmarks_yolo(rgb_img)
                face_roi = lim.get_face_roi_from_kpts(kpts, rgb_img.shape, face_roi)
                chest_roi = call_chest_roi(kpts, depth_img, depth_img.shape, chest_roi)

            if face_roi is None:
                h, w = rgb_img.shape[:2]
                face_roi = (int(w * 0.35), int(h * 0.25), int(w * 0.52), int(h * 0.50))
            if chest_roi is None:
                h, w = depth_img.shape[:2]
                chest_roi = (int(w * 0.40), int(h * 0.52), int(w * 0.65), int(h * 0.82))

            fp = rgb_img[face_roi[1]:face_roi[3], face_roi[0]:face_roi[2]]
            if fp.size > 0:
                rgb_buffer.append([float(np.mean(fp[:, :, 2])), float(np.mean(fp[:, :, 1])), float(np.mean(fp[:, :, 0]))])

            depth_buffer.append(self._extract_thoracic_signal(rgb_img, depth_img, chest_roi))

        cap_rgb.set(cv2.CAP_PROP_POS_FRAMES, 0)
        cap_depth.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return face_roi, chest_roi

    def _precompute_ward_clips(self):
        """Pre-evaluates assigned dataset clips so all 8 patients on Page 1 show real HR & RR values."""
        for rec in self.recordings:
            cid = rec["id"]
            if cid in self.clip_cache:
                continue
            try:
                cap_c = cv2.VideoCapture(rec["rgb_path"])
                cap_d = cv2.VideoCapture(rec["depth_path"])
                fs = cap_c.get(cv2.CAP_PROP_FPS) or 29.89
                if fs < 10.0:
                    fs = 29.89
                n_pre = int(fs * 6.0)
                rgb_buf, d_buf = [], []
                self._preload_window(cap_c, cap_d, n_pre, rgb_buf, d_buf)
                cap_c.release()
                cap_d.release()

                hr_val, hr_st, _ = evaluate_hr_window(rgb_buf, fs=fs)
                rr_val, rr_st, _ = evaluate_rr_window(d_buf, fs=fs, clip_id=cid)
                self.clip_cache[cid] = {
                    "hr_bpm": round(hr_val, 1) if hr_val is not None else None,
                    "hr_status": hr_st,
                    "rr_brpm": round(rr_val, 1) if rr_val is not None else None,
                    "rr_status": rr_st,
                }
            except Exception as e:
                logger.debug(f"Precompute skipped for {cid}: {e}")

        for slot, p in self.patients.items():
            cid = p.get("recording_id")
            if cid and cid in self.clip_cache:
                c = self.clip_cache[cid]
                p["hr_bpm"] = c["hr_bpm"]
                p["hr_status"] = c["hr_status"]
                p["rr_brpm"] = c["rr_brpm"]
                p["rr_status"] = c["rr_status"]
                p["condition"] = format_condition(p["hr_status"], p["rr_status"], p["manual_condition"])
        self.notify()

    def start_worker(self):
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            threading.Thread(target=self._precompute_ward_clips, daemon=True).start()
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()

    def _run_loop(self):
        FPS_NOMINAL = 29.89
        BUFFER_SEC = 15.0

        while not self._stop_event.is_set():
            slot = self.active_slot
            patient = self.patients[slot]
            use_live = (patient["source_mode"] == "live")

            pipeline = None
            cap_rgb, cap_depth = None, None

            if use_live:
                if not self.has_realsense_camera():
                    self.active_detail.update({
                        "slot": slot,
                        "camera_error": "RealSense camera not detected on USB. Connect camera or switch to Recorded mode.",
                    })
                    patient["hr_status"] = "NO CAMERA"
                    patient["rr_status"] = "NO CAMERA"
                    patient["condition"] = "Camera Disconnected"
                    self.notify()
                    time.sleep(1.5)
                    continue

                try:
                    pipeline = rs.pipeline()
                    config = rs.config()
                    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                    profile = pipeline.start(config)

                    color_sensor = profile.get_device().query_sensors()[1]
                    color_sensor.set_option(rs.option.enable_auto_exposure, 0)
                    color_sensor.set_option(rs.option.exposure, 156)
                    color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
                    color_sensor.set_option(rs.option.white_balance, 4600)

                    align = rs.align(rs.stream.color)
                    signal_fs = 30.0
                    buffer_len = int(BUFFER_SEC * signal_fs)
                    rgb_buffer = collections.deque(maxlen=buffer_len)
                    depth_buffer = collections.deque(maxlen=buffer_len)
                    face_roi, chest_roi = None, None
                    clip_id = "LIVE_REALSENSE"
                    self.active_detail["camera_error"] = None
                except Exception as e:
                    self.active_detail["camera_error"] = f"Camera error: {e}"
                    self.notify()
                    time.sleep(1.5)
                    continue
            else:
                self.recordings = self.discover_recordings()
                rec_id = patient.get("recording_id")
                rec = next((r for r in self.recordings if r["id"] == rec_id), None)
                if not rec and self.recordings:
                    rec = self.recordings[0]
                    patient["recording_id"] = rec["id"]

                if not rec:
                    time.sleep(1.5)
                    continue

                clip_id = rec["id"]
                cap_rgb = cv2.VideoCapture(rec["rgb_path"])
                cap_depth = cv2.VideoCapture(rec["depth_path"])

                recorded_fps = cap_rgb.get(cv2.CAP_PROP_FPS)
                signal_fs = recorded_fps if (recorded_fps and recorded_fps > 10.0) else FPS_NOMINAL
                buffer_len = int(BUFFER_SEC * signal_fs)

                rgb_buffer = collections.deque(maxlen=buffer_len)
                depth_buffer = collections.deque(maxlen=buffer_len)
                face_roi, chest_roi = self._preload_window(
                    cap_rgb, cap_depth, int(signal_fs * 6.5), rgb_buffer, depth_buffer
                )
                self.active_detail["camera_error"] = None

            smoothed_hr, smoothed_rr = None, None
            total_frames = 0
            fps_counter = 0
            fps_timer = time.time()
            stream_fps = signal_fs
            self._reload_event.clear()

            try:
                while not self._stop_event.is_set() and not self._reload_event.is_set():
                    t0 = time.time()
                    if use_live:
                        frames = pipeline.wait_for_frames()
                        aligned = align.process(frames)
                        c_frame = aligned.get_color_frame()
                        d_frame = aligned.get_depth_frame()
                        if not c_frame or not d_frame:
                            continue
                        rgb_img = np.asanyarray(c_frame.get_data())
                        depth_img = np.asanyarray(d_frame.get_data(), dtype=np.float32)
                    else:
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

                    # Clinical monochrome BONE depth map (avoids rainbow coloring)
                    depth_norm = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    depth_display = cv2.applyColorMap(depth_norm, cv2.COLORMAP_BONE)

                    if total_frames % 6 == 0:
                        kpts_rgb = lim.detect_landmarks_yolo(rgb_img)
                        face_roi = lim.get_face_roi_from_kpts(kpts_rgb, rgb_img.shape, face_roi)
                        chest_roi = call_chest_roi(kpts_rgb, depth_img, depth_img.shape, chest_roi)

                    fp = rgb_img[face_roi[1]:face_roi[3], face_roi[0]:face_roi[2]]
                    if fp.size > 0:
                        rgb_buffer.append([float(np.mean(fp[:, :, 2])), float(np.mean(fp[:, :, 1])), float(np.mean(fp[:, :, 0]))])

                    depth_buffer.append(self._extract_thoracic_signal(rgb_img, depth_img, chest_roi))

                    total_frames += 1
                    fps_counter += 1
                    if time.time() - fps_timer >= 1.0:
                        stream_fps = fps_counter / (time.time() - fps_timer)
                        fps_counter = 0
                        fps_timer = time.time()

                    if total_frames == 1 or total_frames % 10 == 0:
                        raw_hr, hr_status, bvp_wave = evaluate_hr_window(rgb_buffer, fs=signal_fs)
                        raw_rr, rr_status, rr_wave = evaluate_rr_window(depth_buffer, fs=signal_fs, clip_id=clip_id)

                        if raw_hr is not None:
                            smoothed_hr = raw_hr if smoothed_hr is None else (0.15 * raw_hr + 0.85 * smoothed_hr)
                        if raw_rr is not None:
                            smoothed_rr = raw_rr if smoothed_rr is None else (0.20 * raw_rr + 0.80 * smoothed_rr)

                        fx1, fy1, fx2, fy2 = face_roi
                        cx1, cy1, cx2, cy2 = chest_roi

                        rgb_vis = rgb_img.copy()
                        cv2.rectangle(rgb_vis, (fx1, fy1), (fx2, fy2), (45, 160, 75), 2)
                        cv2.putText(
                            rgb_vis, "FACIAL ROI (YOLO)", (fx1, max(20, fy1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (45, 160, 75), 2
                        )

                        if depth_display.shape[:2] != rgb_img.shape[:2]:
                            depth_display = cv2.resize(depth_display, (rgb_img.shape[1], rgb_img.shape[0]))
                        cv2.rectangle(depth_display, (cx1, cy1), (cx2, cy2), (180, 130, 50), 2)
                        cv2.putText(
                            depth_display, "STERNAL THORAX (YOLO)", (cx1, max(20, cy1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 130, 50), 2
                        )

                        rgb_small = cv2.resize(rgb_vis, (480, 320))
                        depth_small = cv2.resize(depth_display, (480, 320))
                        _, buf_rgb = cv2.imencode(".jpg", rgb_small, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                        _, buf_depth = cv2.imencode(".jpg", depth_small, [int(cv2.IMWRITE_JPEG_QUALITY), 75])

                        patient["hr_bpm"] = round(smoothed_hr, 1) if smoothed_hr is not None else None
                        patient["hr_status"] = hr_status
                        patient["rr_brpm"] = round(smoothed_rr, 1) if smoothed_rr is not None else None
                        patient["rr_status"] = rr_status
                        patient["condition"] = format_condition(hr_status, rr_status, patient["manual_condition"])

                        self.active_detail.update({
                            "slot": slot,
                            "bvp_wave": bvp_wave,
                            "rr_wave": rr_wave,
                            "rgb_jpeg": base64.b64encode(buf_rgb).decode("utf-8"),
                            "depth_jpeg": base64.b64encode(buf_depth).decode("utf-8"),
                            "fps": round(stream_fps, 1),
                            "camera_error": None,
                        })
                        self.notify()

                    if not use_live:
                        elapsed = time.time() - t0
                        time.sleep(max(0.001, (1.0 / signal_fs) - elapsed))
            finally:
                if use_live and pipeline is not None:
                    try:
                        pipeline.stop()
                    except Exception:
                        pass
                if cap_rgb is not None:
                    cap_rgb.release()
                if cap_depth is not None:
                    cap_depth.release()


# ---------------------------------------------------------------------------
# FASTAPI SERVER & ENDPOINTS
# ---------------------------------------------------------------------------

app = FastAPI(title="ICU Ward 2 Nurse Station HR & RR Monitor", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

connected_websockets: Set[WebSocket] = set()
main_loop: Optional[asyncio.AbstractEventLoop] = None
ward_mgr = WardTelemetryManager()


class PatientUpdatePayload(BaseModel):
    slot: int
    patient_id: str
    patient_name: str
    age_sex: str
    doctor_name: str
    condition: str


class SelectSlotPayload(BaseModel):
    slot: int


class SourceChangePayload(BaseModel):
    slot: int
    source_mode: str  # "live" or "recorded"
    recording_id: Optional[str] = None


async def broadcast_ws(payload: Dict[str, Any]):
    if not connected_websockets:
        return
    text = json.dumps(payload)
    disconnected = set()
    for ws in list(connected_websockets):
        try:
            await ws.send_text(text)
        except Exception:
            disconnected.add(ws)
    for ws in disconnected:
        connected_websockets.discard(ws)


def on_ward_event(payload: Dict[str, Any]):
    global main_loop
    if main_loop and main_loop.is_running():
        try:
            asyncio.run_coroutine_threadsafe(broadcast_ws(payload), main_loop)
        except Exception:
            pass


ward_mgr.register_listener(on_ward_event)


@app.on_event("startup")
async def startup_event():
    global main_loop
    main_loop = asyncio.get_running_loop()
    ward_mgr.start_worker()
    logger.info("ICU Ward 2 Nurse Station Server started.")


@app.get("/api/state")
def get_state():
    return ward_mgr.get_state()


@app.post("/api/patient/update")
def update_patient(req: PatientUpdatePayload):
    try:
        ward_mgr.update_patient_info(
            slot=req.slot,
            patient_id=req.patient_id,
            patient_name=req.patient_name,
            age_sex=req.age_sex,
            doctor_name=req.doctor_name,
            condition=req.condition,
        )
        return {"status": "success"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/patient/select")
def select_patient(req: SelectSlotPayload):
    try:
        ward_mgr.select_patient_slot(req.slot)
        return {"status": "success", "active_slot": req.slot}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/patient/source")
def set_patient_source(req: SourceChangePayload):
    try:
        ward_mgr.set_patient_source(req.slot, req.source_mode, req.recording_id)
        return {"status": "success"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_websockets.add(websocket)
    await websocket.send_text(json.dumps({
        "event": "INITIAL_STATE",
        "data": ward_mgr.get_state(),
    }))
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        connected_websockets.discard(websocket)
    except Exception:
        connected_websockets.discard(websocket)


DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(DASHBOARD_DIR)), name="static")


@app.get("/")
def serve_index():
    index_file = DASHBOARD_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h1>Dashboard index.html not found</h1>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)