import cv2
import numpy as np
import pyrealsense2 as rs
import time

def setup_d435():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 30)
    align = rs.align(rs.stream.color)
    print("[INFO] Starting RealSense pipeline...")
    profile = pipeline.start(config)
    device = profile.get_device()
    depth_sensor = device.first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    color_sensor = None
    for sensor in device.query_sensors():
        if "RGB Camera" in sensor.get_info(rs.camera_info.name):
            color_sensor = sensor
            break
    if not color_sensor:
        raise RuntimeError("RGB Camera sensor not found on the device.")

    print("[INFO] Locking RGB sensor settings for rPPG stability...")
    color_sensor.set_option(rs.option.enable_auto_exposure, 0)
    color_sensor.set_option(rs.option.exposure, 156)
    color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
    color_sensor.set_option(rs.option.white_balance, 4600)
    time.sleep(1.0)
    return pipeline, align, depth_scale

def run_diagnostic_stream():
    pipeline, align, _ = setup_d435()
    colorizer = rs.colorizer()
    fps_timer = time.time()
    frame_count = 0
    fps = 0.0
    print("[INFO] Diagnostic stream active. Press 'q' or 'ESC' to exit.")
    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            frame_count += 1
            if time.time() - fps_timer >= 1.0:
                fps = frame_count / (time.time() - fps_timer)
                frame_count = 0
                fps_timer = time.time()

            center_x, center_y = 320, 240
            center_depth = depth_frame.get_distance(center_x, center_y)
            cv2.putText(color_image, f"FPS: {fps:.1f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(color_image, f"Center Depth: {center_depth:.2f} m", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.drawMarker(color_image, (center_x, center_y), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.drawMarker(colorized_depth, (center_x, center_y), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)

            stacked = np.hstack((color_image, colorized_depth))
            cv2.imshow("VisionICU - D435 Diagnostic Alignment Test", stacked)
            if (cv2.waitKey(1) & 0xFF) in [ord('q'), 27]:
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_diagnostic_stream()
