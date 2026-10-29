"""Chỉnh ngưỡng luật vi phạm trên video thật mà không phải chạy lại RF-DETR cho mỗi lần thử.

Bước 1 (Colab hoặc laptop): chạy RF-DETR một lần trên mỗi video buổi 1, lưu kết quả phát hiện từng khung.
    python -m ai_service.main --source buoi1_000.mp4 --dry-run --out runs/buoi1_000 \
        --threshold 0.2 --dump-dets runs/buoi1_000/dets.jsonl

Bước 2: A gửi file đáp án scenes.csv (lấy từ kịch bản quay, thời gian tính trong từng file video):
    video,bat_dau,ket_thuc,loai
    buoi1_000.mp4,00:30,00:45,phone
    buoi1_000.mp4,00:45,02:10,normal
    buoi1_000.mp4,02:10,02:25,calculator
  loai: phone (đang dùng điện thoại), normal (làm bài bình thường), hoặc tên cảnh khác (calculator, head_turn…).

Bước 3: thử nhiều bộ ngưỡng, mỗi bộ chỉ mất một phần giây:
    python tools/tune_rules.py --dets runs/buoi1_*/dets.jsonl --scenes scenes_buoi1.csv

Cách tính:
  - Bắt được: cảnh phone có ít nhất một cảnh báo trong khoảng [bắt đầu, kết thúc + 3 giây].
  - Báo sai: cảnh báo rơi vào cảnh không phải phone.
  - Ngoài kịch bản: cảnh báo rơi vào đoạn không có trong scenes.csv (không tính là sai, nhưng nên xem lại).
"""
import argparse
import csv
import glob
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai_service.rules import Det, ViolationRules  # noqa: E402

TOLERANCE_S = 3.0


def mmss(text):
    parts = [float(p) for p in str(text).strip().split(":")]
    sec = 0.0
    for p in parts:
        sec = sec * 60 + p
    return sec


def load_dets(path):
    with open(path, encoding="utf-8") as f:
        meta = json.loads(f.readline())
        frames = []
        for line in f:
            row = json.loads(line)
            frames.append((row["t"], [Det(d[0], d[1], d[2], tuple(d[3:7])) for d in row["d"]]))
    return meta, frames


def load_scenes(path):
    scenes = {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            scenes.setdefault(row["video"].strip(), []).append(
                (mmss(row["bat_dau"]), mmss(row["ket_thuc"]), row["loai"].strip().lower()))
    return scenes


def run_rules(frames, params, alert_classes):
    rules = ViolationRules(alert_classes, params["min_confidence"], params["window_s"], params["min_ratio"],
                           params["cooldown_s"])
    return [t for t, dets in frames for _ in rules.update(t, dets)]


def score(alert_times, scenes, target="phone"):
    hits = misses = false_alarms = outside = 0
    for start, end, kind in scenes:
        if kind == target:
            if any(start <= t <= end + TOLERANCE_S for t in alert_times):
                hits += 1
            else:
                misses += 1
    for t in alert_times:
        inside = [kind for start, end, kind in scenes if start <= t <= end + (TOLERANCE_S if kind == target else 0)]
        if target in inside:
            continue
        if inside:
            false_alarms += 1
        else:
            outside += 1
    normal_min = sum(end - start for start, end, kind in scenes if kind == "normal") / 60
    return hits, misses, false_alarms, outside, normal_min


def main():
    # Windows ghi ra file hoặc ống dẫn bằng bảng mã cp1252, không in được tiếng Việt
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Thử nhiều bộ ngưỡng luật vi phạm trên kết quả phát hiện đã lưu")
    ap.add_argument("--dets", nargs="+", required=True, help="file dets.jsonl (dùng được dấu *)")
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--alert-class", nargs="+", default=["phone"],
                    help="lớp được báo; ví dụ --alert-class phone calculator để thử phương án báo cả máy tính")
    ap.add_argument("--target", default="phone", help="loại cảnh trong scenes.csv được coi là vi phạm cần bắt")
    ap.add_argument("--cooldown", type=float, default=10.0,
                    help="khi chỉnh nên để ngắn hơn lúc chạy thật (30 giây) để các cảnh liền nhau không bị nuốt")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    paths = sorted({p for pattern in args.dets for p in glob.glob(pattern)})
    if not paths:
        sys.exit("Không thấy file dets.jsonl nào")
    scenes = load_scenes(args.scenes)
    videos = []
    for p in paths:
        meta, frames = load_dets(p)
        if meta["video"] not in scenes:
            print(f"[bỏ qua] {p}: video {meta['video']} không có trong {args.scenes}")
            continue
        videos.append((meta, frames, scenes[meta["video"]]))
        print(f"{meta['video']}: {len(frames)} khung, {len(scenes[meta['video']])} cảnh, ngưỡng phát hiện {meta['threshold']}")
    if not videos:
        sys.exit("Không có video nào khớp giữa dets và scenes.csv")

    grid = {"min_confidence": [0.3, 0.4, 0.5, 0.6], "window_s": [1.0, 1.5, 2.0, 3.0],
            "min_ratio": [0.4, 0.5, 0.6, 0.7], "cooldown_s": [args.cooldown]}
    results = []
    for combo in itertools.product(*grid.values()):
        params = dict(zip(grid, combo))
        total = [0, 0, 0, 0, 0.0]
        for meta, frames, sc in videos:
            for k, v in enumerate(score(run_rules(frames, params, args.alert_class), sc, args.target)):
                total[k] += v
        results.append((params, *total))

    # Ưu tiên bắt được nhiều nhất, rồi ít báo sai nhất, rồi cửa sổ dài hơn (ổn định hơn)
    results.sort(key=lambda r: (-r[1], r[3], -r[0]["window_s"], -r[0]["min_confidence"]))
    n_phone = results[0][1] + results[0][2]
    normal_min = results[0][5]
    print(f"\nBáo các lớp {args.alert_class}; {n_phone} cảnh '{args.target}', {normal_min:.1f} phút làm bài bình thường\n")
    print(f"{'min_conf':>8} {'window':>6} {'ratio':>5} | {'bắt được':>9} {'báo sai':>7} {'ngoài KB':>8}")
    for params, hits, misses, fa, outside, _ in results[:args.top]:
        print(f"{params['min_confidence']:>8} {params['window_s']:>6} {params['min_ratio']:>5} | "
              f"{hits:>4}/{n_phone:<4} {fa:>7} {outside:>8}")
    best = results[0][0]
    print("\nGhi vào config.yaml (mục rules), giữ cooldown_s: 30 khi chạy thật:")
    print(f"  alert_classes: [{', '.join(args.alert_class)}]")
    print(f"  min_confidence: {best['min_confidence']}\n  window_s: {best['window_s']}\n  min_ratio: {best['min_ratio']}")


if __name__ == "__main__":
    main()
