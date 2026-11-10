"""Chấm phát hiện nhìn bài (quay đầu) trên video có đáp án, rồi thử nhiều ngưỡng mà không chạy lại MediaPipe.

Cần trước: dets.jsonl của video (chạy dịch vụ AI với --dump-dets, xem tools/review_alerts.py) để lấy khung thí sinh,
và mô hình MediaPipe trong weights/ (face_landmarker.task hoặc pose_landmarker_lite.task, link ở đầu
behavior/head_pose.py và behavior/pose_turn.py).

Bước 1, đo góc quay của từng thí sinh (chậm, chạy một lần cho mỗi cách; MediaPipe chạy CPU):
    python tools/eval_head_turn.py measure --video videos/buoi_quay2.mp4 --dets runs/buoi2/dets.jsonl \
        --method pose --every 5 --out runs/buoi2/turn_pose.jsonl
    python tools/eval_head_turn.py measure --video videos/buoi_quay2.mp4 --dets runs/buoi2/dets.jsonl \
        --method face --every 5 --out runs/buoi2/turn_face.jsonl

Bước 2, thử ngưỡng (vài giây):
    python tools/eval_head_turn.py score --ratios runs/buoi2/turn_pose.jsonl --scenes docs/khung/scenes_buoi2.csv

In tỉ lệ đo được góc (MediaPipe có thấy mặt/người không) và bảng yaw_ratio × min_duration_s → bắt được bao nhiêu
cảnh head_turn, báo sai bao nhiêu lần trong cảnh normal. Ghi bộ tốt nhất vào config.yaml mục behavior
(method, yaw_ratio, min_duration_s) rồi mới bật enabled: true.
"""
import argparse
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.review_alerts import classify, fmt  # noqa: E402
from tools.tune_rules import load_scenes  # noqa: E402


def simulate(samples, yaw_ratio, min_duration_s, cooldown_s):
    """Cùng luật với HeadTurnDetector.analyze / PoseTurnDetector.update: lệch quá yaw_ratio liên tục
    min_duration_s giây thì báo, rồi nghỉ cooldown_s giây. samples: [(t, [(track_id, ratio hoặc None), ...]), ...]"""
    state, events = {}, []
    for t, items in samples:
        for track_id, r in items:
            st = state.setdefault(track_id, {"since": None, "next_ok": float("-inf")})
            if r is None or abs(r) < yaw_ratio:
                st["since"] = None
                continue
            if st["since"] is None:
                st["since"] = t
            if t - st["since"] >= min_duration_s and t >= st["next_ok"]:
                events.append({"t": t, "track_id": track_id, "ratio": r})
                st["next_ok"] = t + cooldown_s
    return events


def load_ratios(path):
    with open(path, encoding="utf-8") as f:
        meta = json.loads(f.readline())
        samples = [(row["t"], [(x[0], x[1]) for x in row["r"]]) for row in map(json.loads, f)]
    return meta, samples


def measure(args):
    import cv2

    if args.method == "pose":
        from behavior.pose_turn import PoseTurnDetector
        det = PoseTurnDetector()
        crop = det.person_crop
    else:
        from behavior.head_pose import HeadTurnDetector
        det = HeadTurnDetector()
        crop = det.head_crop
    with open(args.dets, encoding="utf-8") as f:
        meta = json.loads(f.readline())
        rows = [json.loads(line) for line in f]
    cap = cv2.VideoCapture(args.video)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    seen = found = 0
    with open(args.out, "w", encoding="utf-8") as out:
        out.write(json.dumps({"video": meta["video"], "method": args.method, "every": args.every}) + "\n")
        for i, row in enumerate(rows):          # dets.jsonl của file video có đúng một dòng cho mỗi khung
            ok, frame = cap.read()
            if not ok:
                break
            if i % args.every:
                continue
            items = []
            for d in row["d"]:
                if d[1] != "student":
                    continue
                r = det.ratio(crop(frame, d[3:7]))
                seen += 1
                found += r is not None
                items.append([d[0], None if r is None else round(r, 3)])
            out.write(json.dumps({"t": row["t"], "r": items}) + "\n")
            if i % (args.every * 200) == 0:
                print(f"{fmt(row['t'])}  đo được {found}/{seen}", flush=True)
    cap.release()
    print(f"Xong: đo được góc quay {found}/{seen} lần ({100 * found / max(seen, 1):.0f}%). Lưu ở {args.out}")


def score(args):
    meta, samples = load_ratios(args.ratios)
    scenes = load_scenes(*args.scenes).get(meta["video"])
    if not scenes:
        sys.exit(f"Video {meta['video']} không có trong {args.scenes}")
    n_all = sum(len(items) for _, items in samples)
    n_ok = sum(r is not None for _, items in samples for _, r in items)
    targets = sum(k == args.target for _, _, k in scenes)
    print(f"{meta['video']} ({meta['method']}): đo được góc {n_ok}/{n_all} lần ({100 * n_ok / max(n_all, 1):.0f}%), "
          f"{targets} cảnh {args.target}")
    if not n_ok:
        sys.exit("MediaPipe không đo được góc nào: thử --method pose, hoặc ghi giới hạn này vào báo cáo")
    results = []
    for yaw, dur in itertools.product(args.yaw, args.duration):
        events = simulate(samples, yaw, dur, args.cooldown)
        events, per_scene = classify(events, scenes, args.target)
        hits = sum(p.get("hit", False) for p in per_scene)
        fa = sum(e["verdict"] == "bao_sai" for e in events)
        results.append((hits, fa, yaw, dur))
    results.sort(key=lambda x: (-x[0], x[1], -x[2], -x[3]))   # hoà thì chọn ngưỡng chặt hơn: ít báo sai khi chạy thật
    print(f"\n{'yaw_ratio':>9} {'min_dur':>7} | {'bắt được':>9} {'báo sai':>7}")
    for hits, fa, yaw, dur in results[:args.top]:
        print(f"{yaw:>9} {dur:>7} | {hits:>4}/{targets:<4} {fa:>7}")
    hits, fa, yaw, dur = results[0]
    print(f"\nGhi vào config.yaml (mục behavior): method: {meta['method']}, yaw_ratio: {yaw}, min_duration_s: {dur}")
    print("Đạt nghiệm thu khi bắt được ít nhất 3/5 và báo sai không quá 2 trong 10 phút bình thường.")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Chấm phát hiện nhìn bài trên video có đáp án")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure", help="đo góc quay của từng thí sinh bằng MediaPipe, lưu ra .jsonl")
    m.add_argument("--video", required=True)
    m.add_argument("--dets", required=True)
    m.add_argument("--method", choices=["pose", "face"], default="pose")
    m.add_argument("--every", type=int, default=5, help="đo 1 khung trong N khung, giống behavior.every_n_frames")
    m.add_argument("--out", required=True)
    s = sub.add_parser("score", help="thử nhiều ngưỡng trên góc đã đo")
    s.add_argument("--ratios", required=True)
    s.add_argument("--scenes", nargs="+", required=True)
    s.add_argument("--target", default="head_turn")
    s.add_argument("--yaw", type=float, nargs="+", default=[0.2, 0.25, 0.3, 0.35, 0.4, 0.5])
    s.add_argument("--duration", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    s.add_argument("--cooldown", type=float, default=30.0)
    s.add_argument("--top", type=int, default=10)
    args = ap.parse_args()
    measure(args) if args.cmd == "measure" else score(args)


if __name__ == "__main__":
    main()
