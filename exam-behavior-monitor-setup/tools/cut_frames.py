"""Cắt khung từ video để gán nhãn, đặt tên theo giây và chia sẵn train / valid / test theo mốc thời gian.

Tên file dạng <tên video>_s000495.jpg (giây thứ 495), nên sau khi qua Roboflow vẫn biết khung nằm ở đâu trong video
và tools/check_split.py kiểm tra được việc chia tập.

    # Buổi quay 2: 1 khung mỗi 2 giây, trước 08:15 vào train, từ 08:15 vào valid
    python tools/cut_frames.py --video videos/buoi_quay2.mp4 --every 2 --valid-from 08:15 --out frames_buoi2

    # Video nghiệm thu 1: 1 khung mỗi 10 giây, tất cả vào test (không bao giờ train)
    python tools/cut_frames.py --video nghiemthu1.mp4 --every 10 --all test --out frames_nghiemthu1

Kết quả: <out>/train/, <out>/valid/, <out>/test/ (thư mục rỗng thì không tạo). Mỗi thư mục upload lên Roboflow
thành một batch riêng rồi chọn "Add all images to Train/Valid/Test" tương ứng.
"""
import argparse
import os
import sys

import cv2


def mmss(text):
    """'08:15' -> 495.0 ; '1:02:03' -> 3723.0 ; '75' -> 75.0"""
    sec = 0.0
    for part in str(text).strip().split(":"):
        sec = sec * 60 + float(part)
    return sec


def split_of(t, valid_from=None, test_from=None, all_to=None):
    if all_to:
        return all_to
    if test_from is not None and t >= test_from:
        return "test"
    if valid_from is not None and t >= valid_from:
        return "valid"
    return "train"


def frame_name(video_path, t):
    stem = os.path.splitext(os.path.basename(video_path))[0].replace(".", "_").replace(" ", "_")
    return f"{stem}_s{int(round(t)):06d}.jpg"


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Cắt khung từ video và chia train/valid/test theo thời gian")
    ap.add_argument("--video", required=True)
    ap.add_argument("--every", type=float, default=2.0, help="số giây giữa hai khung")
    ap.add_argument("--start", default="0", help="bỏ qua đoạn đầu, ví dụ 00:05")
    ap.add_argument("--end", help="dừng ở mốc này, ví dụ 10:15")
    ap.add_argument("--valid-from", help="từ mốc này trở đi vào valid, ví dụ 08:15")
    ap.add_argument("--test-from", help="từ mốc này trở đi vào test")
    ap.add_argument("--all", choices=["train", "valid", "test"], help="đưa tất cả vào một tập")
    ap.add_argument("--out", required=True)
    ap.add_argument("--quality", type=int, default=95, help="chất lượng JPEG")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Không mở được {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = n_frames / fps if n_frames else None
    print(f"{args.video}: {w}x{h}, {fps:.1f} khung/giây, dài {duration:.0f} giây" if duration
          else f"{args.video}: {w}x{h}, {fps:.1f} khung/giây")
    if w and w < 1280:
        print(f"  [CHÚ Ý] Video chỉ rộng {w} pixel, nhiều khả năng là luồng phụ. Luồng chính cho ảnh nét hơn.")

    start = mmss(args.start)
    end = mmss(args.end) if args.end else (duration or float("inf"))
    valid_from = mmss(args.valid_from) if args.valid_from else None
    test_from = mmss(args.test_from) if args.test_from else None

    counts = {}
    t = start
    while t <= end:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, frame = cap.read()
        if not ok:
            break
        split = split_of(t, valid_from, test_from, args.all)
        folder = os.path.join(args.out, split)
        os.makedirs(folder, exist_ok=True)
        cv2.imwrite(os.path.join(folder, frame_name(args.video, t)), frame, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
        counts[split] = counts.get(split, 0) + 1
        t += args.every
    cap.release()
    print("Đã cắt:", ", ".join(f"{k} {v} khung" for k, v in sorted(counts.items())) or "0 khung", f"-> {args.out}")


if __name__ == "__main__":
    main()
