"""Dò camera trong mạng nội bộ và thử các địa chỉ RTSP hay gặp, in độ phân giải và tốc độ khung hình.

    python tools/probe_camera.py --subnet 192.168.1.0/24 --password 123      # quét cả mạng tìm máy mở cổng 554
    python tools/probe_camera.py --ip 192.168.1.50 --password 123             # thử thẳng một địa chỉ
    python tools/probe_camera.py --url "rtsp://admin:123@192.168.1.50:554/onvif1"

Chỉ quét mạng của chính nhóm (router, hotspot), không quét mạng của trường.
"""
import argparse
import concurrent.futures
import ipaddress
import os
import socket
import time
from urllib.parse import quote

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")   # ẩn cảnh báo timeout của FFmpeg khi thử địa chỉ sai
import cv2  # noqa: E402  (phải đặt biến môi trường trước khi nạp OpenCV)

PATHS = [
    "/onvif1", "/onvif2",                              # Yoosee
    "/stream1", "/stream2",                            # Tapo
    "/cam/realmonitor?channel=1&subtype=0",            # Imou, Dahua
    "/h264/ch1/main/av_stream", "/Streaming/Channels/101",  # EZVIZ, Hikvision
    "/live/ch00_0", "/11", "/live", "/",               # một số camera Trung Quốc khác
]


def port_open(ip, port, timeout=0.5):
    try:
        with socket.create_connection((str(ip), port), timeout=timeout):
            return True
    except OSError:
        return False


def try_url(url, timeout_ms=5000, sample=40):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG,
                           [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms, cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms])
    if not cap.isOpened():
        return None
    ok, frame = cap.read()
    if not ok:
        cap.release()
        return None
    t0, n = time.perf_counter(), 0
    while n < sample:
        ok, _ = cap.read()
        if not ok:
            break
        n += 1
    measured = n / (time.perf_counter() - t0) if n else 0.0
    info = {"w": frame.shape[1], "h": frame.shape[0], "fps_bao": cap.get(cv2.CAP_PROP_FPS), "fps_do": round(measured, 1)}
    cap.release()
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subnet", help="ví dụ 192.168.1.0/24")
    ap.add_argument("--ip")
    ap.add_argument("--url")
    ap.add_argument("--port", type=int, default=554)
    ap.add_argument("--user", default="admin")
    ap.add_argument("--password", default="123")
    args = ap.parse_args()

    if args.url:
        print(args.url, "->", try_url(args.url) or "không mở được", flush=True)
        return

    hosts = []
    if args.ip:
        if not port_open(args.ip, args.port, timeout=2):
            print(f"{args.ip} không mở cổng {args.port}: sai địa chỉ IP, camera chưa vào mạng, hoặc camera không có RTSP.")
            return
        hosts = [args.ip]
    elif args.subnet:
        net = list(ipaddress.ip_network(args.subnet, strict=False).hosts())
        print(f"Quét {len(net)} địa chỉ, cổng {args.port}…")
        with concurrent.futures.ThreadPoolExecutor(64) as ex:
            hosts = [str(ip) for ip, ok in zip(net, ex.map(lambda ip: port_open(ip, args.port), net)) if ok]
        print("Máy mở cổng RTSP:", hosts or "không có", flush=True)
    else:
        ap.error("cần --subnet, --ip hoặc --url")

    cred = f"{quote(args.user)}:{quote(args.password)}@" if args.user else ""
    for host in hosts:
        for path in PATHS:
            for prefix in (cred, ""):
                url = f"rtsp://{prefix}{host}:{args.port}{path}"
                info = try_url(url)
                shown = f"rtsp://{args.user + ':***@' if prefix else ''}{host}:{args.port}{path}"
                print(f"  {'CÓ HÌNH ' if info else 'không   '} {shown}  {info or ''}", flush=True)
                if info:
                    break


if __name__ == "__main__":
    main()
