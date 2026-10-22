import time
from pathlib import Path
import numpy as np
import cv2

try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except ImportError:
    HAS_REALSENSE = False

DATA_DIR = Path(__file__).resolve().parent / "data" / "raw" / "own"


def run_live_realsense():
    """Runs your exact RealSense D435 protocol with hardware exposure locks."""
    pipeline = rs.pipeline()
    config = rs.config()

    # Configure streams as specified in the protocol
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 30)

    profile = pipeline.start(config)

    # Verify sensor access and parameter locks (prevents rPPG brightness oscillation)
    color_sensor = profile.get_device().query_sensors()[1]
    color_sensor.set_option(rs.option.enable_auto_exposure, 0)
    color_sensor.set_option(rs.option.exposure, 156)
    color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
    color_sensor.set_option(rs.option.white_balance, 4600)

    align = rs.align(rs.stream.color)

    print("[HARDWARE] Camera connected and streaming with exposure locked! Press 'q' to exit preview.")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_img = np.asanyarray(color_frame.get_data())
            depth_img = np.asanyarray(depth_frame.get_data())
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET
            )

            # Show RGB and aligned Depth side-by-side
            preview = np.hstack((color_img, depth_colormap))
            cv2.imshow("RealSense Stream Test (RGB | Depth)", preview)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


def run_offline_fallback():
    """Fallback when testing without the RealSense camera attached."""
    rgb_files = sorted(DATA_DIR.glob("*_rgb.mp4"))
    if not rgb_files:
        print(f"[ERROR] No RealSense camera connected and no *_rgb.mp4 clips found in {DATA_DIR}")
        return

    # Prefer a resting clip for testing if available
    rgb_path = next((f for f in rgb_files if "rest" in f.name), rgb_files[0])
    prefix = rgb_path.name.replace("_rgb.mp4", "")
    depth_path = DATA_DIR / f"{prefix}_depth.mkv"

    print(f"[OFFLINE MODE] No RealSense camera detected. Streaming dataset clip: {prefix}")
    print("Press 'q' to exit preview.")

    cap_rgb = cv2.VideoCapture(str(rgb_path))
    cap_depth = cv2.VideoCapture(str(depth_path))

    try:
        while True:
            ret_c, color_img = cap_rgb.read()
            ret_d, depth_raw = cap_depth.read()

            if not ret_c or not ret_d:
                cap_rgb.set(cv2.CAP_PROP_POS_FRAMES, 0)
                cap_depth.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            depth_gray = (
                cv2.cvtColor(depth_raw, cv2.COLOR_BGR2GRAY)
                if len(depth_raw.shape) == 3
                else depth_raw
            )
            depth_colormap = cv2.normalize(depth_gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_colormap = cv2.applyColorMap(depth_colormap, cv2.COLORMAP_JET)

            if depth_colormap.shape[:2] != color_img.shape[:2]:
                depth_colormap = cv2.resize(depth_colormap, (color_img.shape[1], color_img.shape[0]))

            preview = np.hstack((color_img, depth_colormap))
            cv2.imshow("RealSense Stream Test (RGB | Depth) [OFFLINE FALLBACK]", preview)

            if cv2.waitKey(33) & 0xFF == ord('q'):
                break
    finally:
        cap_rgb.release()
        cap_depth.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    has_device = False
    if HAS_REALSENSE:
        ctx = rs.context()
        if len(ctx.query_devices()) > 0:
            has_device = True

    if has_device:
        run_live_realsense()
    else:
        run_offline_fallback()