"""Kiểm tra cách chia dataset theo video (bổ sung cho benchmark/check_leakage.py).

check_leakage.py tìm ảnh giống nhau giữa các tập. Với ảnh camera cố định thì mọi khung đều giống nhau vì cùng phòng,
nên điều cần kiểm tra là theo video:
  1. Video nào có khung trong test thì không được có khung trong train hay valid -> LỖI.
  2. Video chia giữa train và valid thì phải cắt theo một mốc thời gian, không xen kẽ -> CẢNH BÁO nếu xen kẽ.
  3. Ảnh camera có kích thước 512x512 là đã bị Resize khi xuất từ Roboflow -> CẢNH BÁO.
In thêm số ảnh và số đối tượng từng lớp trong mỗi tập.

    python tools/check_split.py dataset_v5/
    python tools/check_split.py dataset_v5/ --sample-fps 0.5   # khung do Roboflow tự cắt từ video (tên *_mp4-0016)

Tên khung hiểu được:
  buoi_quay2_s000495   (tools/cut_frames.py: giây thứ 495)
  test1_mp4-0016       (Roboflow tự cắt video: khung thứ 16; cần --sample-fps để đổi ra giây)
Ảnh không theo hai dạng trên (ảnh tải từ mạng...) không tính vào kiểm tra theo video.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

SPLITS = ("train", "valid", "test")
PAT_SECONDS = re.compile(r"^(?P<video>.+)_s(?P<sec>\d{6})$")
PAT_ROBOFLOW_VIDEO = re.compile(r"^(?P<video>.+_(?:mp4|avi|mov|mkv))-(?P<idx>\d+)$", re.IGNORECASE)


def original_stem(file_name):
    """'buoi_quay2_s000495_jpg.rf.ab12.jpg' -> 'buoi_quay2_s000495'"""
    name = os.path.basename(file_name)
    for marker in ("_jpg.rf.", "_jpeg.rf.", "_png.rf.", ".rf."):
        if marker in name:
            return name.split(marker)[0]
    return os.path.splitext(name)[0]


def parse_frame(file_name, sample_fps=None):
    """Trả về (video, giây hoặc None) nếu là khung cắt từ video, ngược lại (None, None)."""
    stem = original_stem(file_name)
    m = PAT_SECONDS.match(stem)
    if m:
        return m["video"], float(m["sec"])
    m = PAT_ROBOFLOW_VIDEO.match(stem)
    if m:
        return m["video"], (int(m["idx"]) / sample_fps if sample_fps else None)
    return None, None


def load_split(root, split):
    path = os.path.join(root, split, "_annotations.coco.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def check(root, sample_fps=None):
    """Trả về (danh sách lỗi, danh sách cảnh báo, các dòng thống kê)."""
    errors, warnings, lines = [], [], []
    frames = defaultdict(lambda: defaultdict(list))   # video -> split -> [giây]
    found = False
    for split in SPLITS:
        coco = load_split(root, split)
        if coco is None:
            continue
        found = True
        cat_name = {c["id"]: c["name"] for c in coco["categories"]}
        per_class = Counter(cat_name.get(a["category_id"], a["category_id"]) for a in coco["annotations"])
        n_cam, resized = 0, 0
        for im in coco["images"]:
            video, sec = parse_frame(im["file_name"], sample_fps)
            if video is None:
                continue
            n_cam += 1
            frames[video][split].append(sec)
            if im.get("width") == 512 and im.get("height") == 512:
                resized += 1
        classes = ", ".join(f"{k} {v}" for k, v in sorted(per_class.items())) or "không có nhãn"
        lines.append(f"{split}: {len(coco['images'])} ảnh ({n_cam} khung camera) | {classes}")
        if resized:
            warnings.append(f"{split}: {resized} khung camera có kích thước 512x512, có vẻ đã bị Resize khi xuất. "
                            f"Tạo lại version trên Roboflow, bỏ bước Resize.")
    if not found:
        errors.append(f"Không thấy train/valid/test/_annotations.coco.json trong {root}")
        return errors, warnings, lines

    lines.append("")
    for video in sorted(frames):
        by_split = frames[video]
        parts = []
        for split in SPLITS:
            secs = [s for s in by_split.get(split, []) if s is not None]
            if split in by_split:
                rng = f" ({min(secs):.0f}s-{max(secs):.0f}s)" if secs else ""
                parts.append(f"{split} {len(by_split[split])}{rng}")
        lines.append(f"{video}: " + ", ".join(parts))
        if "test" in by_split and ("train" in by_split or "valid" in by_split):
            errors.append(f"{video}: có khung ở test và cả ở train/valid. Video dùng để chấm thì không được train.")
        tr = [s for s in by_split.get("train", []) if s is not None]
        va = [s for s in by_split.get("valid", []) if s is not None]
        if tr and va and not (max(tr) < min(va) or max(va) < min(tr)):
            warnings.append(f"{video}: khung train và valid xen kẽ nhau theo thời gian "
                            f"(train {min(tr):.0f}-{max(tr):.0f}s, valid {min(va):.0f}-{max(va):.0f}s). "
                            f"Nên cắt theo một mốc, ví dụ --valid-from 08:15.")
        if ("train" in by_split and "valid" in by_split) and not (tr and va):
            warnings.append(f"{video}: có ở cả train và valid nhưng không đọc được thời gian của khung; "
                            f"thêm --sample-fps nếu khung do Roboflow tự cắt.")
    return errors, warnings, lines


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Kiểm tra chia dataset theo video")
    ap.add_argument("dataset", help="thư mục chứa train/ valid/ test/ (bản xuất COCO của Roboflow)")
    ap.add_argument("--sample-fps", type=float, help="số khung/giây Roboflow đã cắt từ video (để đổi số thứ tự ra giây)")
    args = ap.parse_args()
    errors, warnings, lines = check(args.dataset, args.sample_fps)
    print("\n".join(lines))
    for w in warnings:
        print("[CẢNH BÁO]", w)
    for e in errors:
        print("[LỖI]", e)
    print("\nĐạt: không có lỗi chia tập theo video." if not errors else f"\nKhông đạt: {len(errors)} lỗi.")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
