import requests

def scan_esp32_endpoints():
    base_ip = "http://192.168.4.1"
    endpoints = [
        "/", "/thermal", "/thermal/", "/thermal/raw",
        "/raw", "/stream", "/index.html", "/debug",
        "/thermal/debug", "/status", "/capture"
    ]
    print(f"[INFO] Scanning ESP32-S3 HTTP Server at {base_ip}...")
    active_endpoints = []
    for path in endpoints:
        url = f"{base_ip}{path}"
        try:
            response = requests.get(url, timeout=2.0)
            if response.status_code == 200:
                print(f"  [SUCCESS] 200 OK -> {url}")
                active_endpoints.append(url)
            elif response.status_code == 404:
                print(f"  [404] Missing    -> {url}")
            else:
                print(f"  [{response.status_code}] Other -> {url}")
        except requests.exceptions.RequestException:
            print(f"  [ERR] Failed to connect to {url}")
    print("\n[SUMMARY]")
    if active_endpoints:
        print(f"Found {len(active_endpoints)} working URLs.")
    else:
        print("CRITICAL FAILURE: 0 working URLs found. The SPIFFS partition failed to mount.")

if __name__ == "__main__":
    scan_esp32_endpoints()
