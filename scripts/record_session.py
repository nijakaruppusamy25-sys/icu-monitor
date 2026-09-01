import argparse
import pyrealsense2 as rs
import numpy as np
import cv2
import json
import time
import csv
import os

# --- Command Line Argument Parsing
parser = argparse.ArgumentParser(description="Record session")
parser.add_argument("subject_id", type=str, help="e.g., sub01")
parser.add_argument("condition", type=str, help="e.g., upright_rest_bright")
parser.add_argument("--rr-bpm", type=int, default=None, help="Metronome rate for JSON label")
parser.add_argument("--duration", type=int, default=30, help="Recording duration in seconds")
args = parser.parse_args()

subject_id = args.subject_id
condition = args.condition
duration_s = args.duration

# 1. Setup Naming & Storage
timestamp_str = time.strftime("%Y%m%d_%H%M%S")
save_dir = "data/raw/own/"
os.makedirs(save_dir, exist_ok=True)
base_name = f"{save_dir}{subject_id}_{condition}_{timestamp_str}"
rgb_path = f"{base_name}_rgb.mp4"
depth_path = f"{base_name}_depth.mkv"
json_path = f"{base_name}_meta.json"

# 2. Initialize RealSense Cameras
print("Connecting to Intel RealSense...")
pipeline = rs.pipeline()
config = rs.config()

# Native hardware resolutions and frame rates
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 30)
profile = pipeline.start(config)

# Apply 'High Accuracy' preset to the Depth Sensor
depth_sensor = profile.get_device().query_sensors()[0]
depth_sensor.set_option(rs.option.visual_preset, 3)

# Apply exact protocol locks (Exposure and WB) to RGB Sensor
color_sensor = profile.get_device().query_sensors()[1]
color_sensor.set_option(rs.option.enable_auto_exposure, 0)
color_sensor.set_option(rs.option.exposure, 156)
color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
color_sensor.set_option(rs.option.white_balance, 4600)

align = rs.align(rs.stream.color)

# 3. Video Writers
colorwriter = cv2.VideoWriter(rgb_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (640, 480), True)
depthwriter = cv2.VideoWriter(depth_path, cv2.VideoWriter_fourcc(*'XVID'), 10, (640, 480), True)

# Timestamp Sync Log
sync_log_d435 = open(f"{base_name}_synclog_d435.csv", mode='w', newline='')
d435_writer = csv.writer(sync_log_d435)
d435_writer.writerow(["frame", "timestamp_s"])

print(f"Recording to: {save_dir}. Recording started...")
start_time = time.time()
frame_count = 0

try:
    # Automatic Buffer: records for duration + 5 seconds
    while time.time() - start_time < (duration_s + 5):
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )

        colorwriter.write(color_image)
        depthwriter.write(depth_colormap)
        d435_writer.writerow([frame_count, time.time()])
        frame_count += 1
finally:
    pipeline.stop()
    colorwriter.release()
    depthwriter.release()
    sync_log_d435.close()

# 4. Output Auto-JSON Metadata
metadata = {
    "subject_id": subject_id,
    "condition": condition,
    "exposure": 156,
    "white_balance": 4600,
    "lighting": "bright",
    "duration_s": duration_s,
    "streams": ["rgb", "depth"],
    "date": time.strftime("%Y-%m-%d")
}

if args.rr_bpm is not None:
    metadata["rr_bpm"] = args.rr_bpm

with open(json_path, 'w') as f:
    json.dump(metadata, f, indent=4)

print(f"Recording saved successfully to {base_name}")
