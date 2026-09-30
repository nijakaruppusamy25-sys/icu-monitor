import cv2
import numpy as np
import requests
import time

def test_waveshare_binary_polling():
    url = "http://192.168.4.1/thermal/frame"
    print(f"[INFO] Connecting to Waveshare binary endpoint: {url}")
    print("[INFO] Press 'q' or 'ESC' to exit.")
    session = requests.Session()
    fps_timer = time.time()
    frame_count = 0
    fps = 0.0
    try:
        while True:
            try:
                response = session.get(url, timeout=2.0)
                if response.status_code == 200:
                    raw_bytes = response.content
                    if len(raw_bytes) == 9920:
                        raw_array = np.frombuffer(raw_bytes, dtype=np.int16)
                        thermal_matrix = (raw_array.reshape((62, 80)).astype(np.float32)) / 100.0
                        temp_min, temp_max = np.min(thermal_matrix), np.max(thermal_matrix)
                        if temp_max > temp_min:
                            normalized = ((thermal_matrix - temp_min) / (temp_max - temp_min) * 255).astype(np.uint8)
                        else:
                            normalized = np.zeros_like(thermal_matrix, dtype=np.uint8)
                        colormap_img = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
                        display_img = cv2.resize(colormap_img, (640, 480), interpolation=cv2.INTER_CUBIC)
                        frame_count += 1
                        if time.time() - fps_timer >= 1.0:
                            fps = frame_count / (time.time() - fps_timer)
                            frame_count = 0
                            fps_timer = time.time()
                        cv2.putText(display_img, f"FPS: {fps:.1f}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(display_img, f"Max Temp: {temp_max:.1f} C", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.imshow("VisionICU - Waveshare Binary Stream", display_img)
            except requests.exceptions.RequestException:
                time.sleep(1)
            if (cv2.waitKey(1) & 0xFF) in [ord('q'), 27]:
                break
    finally:
        cv2.destroyAllWindows()
        session.close()

if __name__ == "__main__":
    test_waveshare_binary_polling()
