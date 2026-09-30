import pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline()
config = rs.config()

# Configure streams as specified in the protocol
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 30)

profile = pipeline.start(config)

# Verify sensor access and parameter locks
color_sensor = profile.get_device().query_sensors()[1]
color_sensor.set_option(rs.option.enable_auto_exposure, 0)
color_sensor.set_option(rs.option.exposure, 156)
color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
color_sensor.set_option(rs.option.white_balance, 4600)

align = rs.align(rs.stream.color)

print("Camera connected and streaming! Press 'q' to exit preview.")

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

        # Show RGB and Depth side-by-side
        preview = np.hstack((color_img, depth_colormap))
        cv2.imshow("RealSense Stream Test (RGB | Depth)", preview)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
