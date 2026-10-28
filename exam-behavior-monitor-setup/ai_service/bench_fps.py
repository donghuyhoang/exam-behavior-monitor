"""Đo tốc độ RF-DETR trên máy đang chạy."""
import argparse
import time

import cv2
import torch

from ai_service.detector import RFDETRDetector
from settings import ROOT, load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--no-optimize", action="store_true")
    args = ap.parse_args()

    print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available(),
          "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "không có GPU")
    cap = cv2.VideoCapture(args.video)
    frames = []
    while len(frames) < args.frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    if not frames:
        raise SystemExit(f"Không đọc được khung nào từ {args.video}")
    print(f"{len(frames)} khung, kích thước {frames[0].shape[1]}x{frames[0].shape[0]}")

    m = load_config()["model"]
    det = RFDETRDetector(str(ROOT / m["weights"]), m.get("size", "medium"), m.get("threshold", 0.3),
                         optimize=not args.no_optimize)
    for f in frames[:10]:
        det(f)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for f in frames:
        det(f)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"FPS: {len(frames) / dt:.1f} ({1000 * dt / len(frames):.1f} ms/khung, gồm chuyển màu và hậu xử lý)")
    if torch.cuda.is_available():
        print(f"Bộ nhớ GPU cao nhất: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
