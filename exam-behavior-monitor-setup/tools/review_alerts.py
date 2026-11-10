"""Chấm lại buổi nghiệm thu trên video đã quay: cảnh nào bắt được, bắt sau bao lâu, cảnh báo sai là vật gì.

Đếm tay trên giao diện dễ lệch giờ (giờ camera, giờ laptop, giờ kịch bản). Ở đây mọi thứ tính theo giờ TRONG video,
cùng mô hình và cùng luật như lúc chạy thật, nên số liệu khớp và chạy lại được khi đổi ngưỡng.

Bước 1: chạy dịch vụ AI một lần trên video, lưu kết quả phát hiện từng khung (laptop GPU hoặc Colab):
    python -m ai_service.main --source videos/nghiemthu1.mp4 --dry-run --out runs/nt1 \
        --threshold 0.2 --dump-dets runs/nt1/dets.jsonl

Bước 2: áp luật trong config.yaml và so với đáp án (vài giây):
    python tools/review_alerts.py --dets runs/nt1/dets.jsonl --scenes docs/khung/scenes_nghiemthu1.csv \
        --video videos/nghiemthu1.mp4 --out runs/nt1/review
    # thử bộ ngưỡng khác mà không sửa config.yaml:
    python tools/review_alerts.py --dets runs/nt1/dets.jsonl --scenes docs/khung/scenes_nghiemthu1.csv \
        --min-confidence 0.6 --window-s 3 --min-ratio 0.6

Kết quả:
  - Bảng từng cảnh: cảnh phone thì bắt được không, cảnh báo đầu tiên sau mấy giây; cảnh normal thì mấy cảnh báo sai.
  - <out>/alerts.csv và <out>/scenes_result.csv: mở bằng Excel, chép số vào bảng kết quả nghiệm thu.
  - <out>/bao_sai/, <out>/bat_duoc/, <out>/ngoai_kich_ban/: khung hình lúc cảnh báo, khoanh đỏ vật bị báo (cần --video).
  - <out>/tong_hop_bao_sai.jpg: các vật bị báo sai ghép thành một tấm, nhìn là biết ví, hộp bút hay bàn tay.

Độ trễ = giờ cảnh báo đầu tiên − bat_dau của cảnh trong scenes.csv. Luật cần thấy điện thoại đủ window_s giây mới báo,
nên độ trễ không thể nhỏ hơn window_s. Muốn đo đúng thì sửa bat_dau của các lần đo độ trễ thành đúng giây điện thoại
xuất hiện trên video. Độ trễ người dùng thấy = số này + latency_ms trên thanh trạng thái (thời gian xử lý một khung).
"""
import argparse
import csv
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai_service.rules import ViolationRules  # noqa: E402
from tools.tune_rules import TOLERANCE_S, load_dets, load_scenes  # noqa: E402

VERDICTS = {"bat_duoc": "bắt được", "bao_sai": "báo sai", "ngoai_kich_ban": "ngoài kịch bản"}
LABELS = {"bat_duoc": "BAT DUOC", "bao_sai": "BAO SAI", "ngoai_kich_ban": "NGOAI KICH BAN"}   # OpenCV chỉ vẽ được chữ không dấu


def fmt(t):
    m, s = divmod(int(round(t)), 60)
    return f"{m:02d}:{s:02d}"


def run_alerts(frames, alert_classes, min_confidence, window_s, min_ratio, cooldown_s):
    """Chạy luật vi phạm trên kết quả phát hiện đã lưu, trả về danh sách cảnh báo theo thời gian."""
    rules = ViolationRules(alert_classes, min_confidence, window_s, min_ratio, cooldown_s)
    alerts = []
    for t, dets in frames:
        for a in rules.update(t, dets):
            alerts.append({"t": t, "cls": a.cls, "conf": a.conf, "box": a.box,
                           "student": a.student_track_id, "student_box": a.student_box})
    return alerts


def classify(alerts, scenes, target="phone", tol=TOLERANCE_S):
    """Gắn mỗi cảnh báo vào một cảnh, cùng cách tính với tools/tune_rules.py:
    - bat_duoc: rơi vào cảnh target, cho phép trễ tol giây sau khi cảnh kết thúc;
    - bao_sai: rơi vào cảnh khác (normal, calculator...);
    - ngoai_kich_ban: không thuộc cảnh nào.
    Trả về (alerts đã gắn verdict, danh sách cảnh kèm cảnh báo, hit và độ trễ)."""
    per_scene = [{"start": s, "end": e, "kind": k, "alerts": []} for s, e, k in sorted(scenes)]
    for a in alerts:
        tgt = [p for p in per_scene if p["kind"] == target and p["start"] <= a["t"] <= p["end"] + tol]
        other = [p for p in per_scene if p["kind"] != target and p["start"] <= a["t"] <= p["end"]]
        if tgt:
            a["verdict"], a["scene"] = "bat_duoc", max(tgt, key=lambda p: p["start"])
        elif other:
            a["verdict"], a["scene"] = "bao_sai", other[0]
        else:
            a["verdict"], a["scene"] = "ngoai_kich_ban", None
        if a["scene"] is not None:
            a["scene"]["alerts"].append(a)
    for p in per_scene:
        if p["kind"] == target:
            p["hit"] = bool(p["alerts"])
            p["latency"] = round(p["alerts"][0]["t"] - p["start"], 1) if p["alerts"] else None
    return alerts, per_scene


def summary(alerts, per_scene, target="phone"):
    tg = [p for p in per_scene if p["kind"] == target]
    lat = [p["latency"] for p in tg if p["latency"] is not None]
    normal_min = sum(p["end"] - p["start"] for p in per_scene if p["kind"] == "normal") / 60
    return {"hits": sum(p["hit"] for p in tg), "scenes": len(tg),
            "latency_median": statistics.median(lat) if lat else None,
            "latency_max": max(lat) if lat else None,
            "false_alarms": sum(a["verdict"] == "bao_sai" for a in alerts),
            "outside": sum(a["verdict"] == "ngoai_kich_ban" for a in alerts),
            "normal_min": normal_min}


def print_report(alerts, per_scene, params, target="phone"):
    print("Luật: " + " · ".join(f"{k} {v}" for k, v in params.items()))
    print(f"\n{'Cảnh':<13} {'Loại':<10} Kết quả")
    for p in per_scene:
        span = f"{fmt(p['start'])}–{fmt(p['end'])}"
        if p["kind"] == target:
            if p["hit"]:
                a = p["alerts"][0]
                res = f"bắt được sau {p['latency']:.1f} s ({a['cls']} {a['conf']:.2f})"
            else:
                res = "BỎ SÓT"
        else:
            n = len(p["alerts"])
            res = "không có cảnh báo" if not n else f"{n} cảnh báo sai: " + ", ".join(
                f"{fmt(a['t'])} {a['cls']} {a['conf']:.2f}" for a in p["alerts"][:6]) + (" …" if n > 6 else "")
        print(f"{span:<13} {p['kind']:<10} {res}")
    s = summary(alerts, per_scene, target)
    lat = "" if s["latency_median"] is None else f" · độ trễ trung vị {s['latency_median']:.1f} s, lâu nhất {s['latency_max']:.1f} s"
    print(f"\nTổng: bắt được {s['hits']}/{s['scenes']} cảnh {target}{lat} · báo sai {s['false_alarms']} "
          f"(trong {s['normal_min']:.1f} phút normal) · ngoài kịch bản {s['outside']}")
    by_cls = {}
    for a in alerts:
        if a["verdict"] == "bao_sai":
            by_cls[a["cls"]] = by_cls.get(a["cls"], 0) + 1
    if by_cls:
        print("Báo sai theo lớp: " + ", ".join(f"{c} {n}" for c, n in sorted(by_cls.items(), key=lambda x: -x[1])))
    return s


def write_csv(out, alerts, per_scene):
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "alerts.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gio_video", "giay", "lop", "tin_cay", "thi_sinh", "x1", "y1", "x2", "y2", "ket_luan", "canh"])
        for a in alerts:
            sc = a["scene"]
            w.writerow([fmt(a["t"]), round(a["t"], 1), a["cls"], a["conf"], a["student"], *[int(v) for v in a["box"]],
                        VERDICTS[a["verdict"]], f"{fmt(sc['start'])}-{fmt(sc['end'])} {sc['kind']}" if sc else ""])
    with open(out / "scenes_result.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bat_dau", "ket_thuc", "loai", "so_canh_bao", "bat_duoc", "do_tre_s"])
        for p in per_scene:
            w.writerow([fmt(p["start"]), fmt(p["end"]), p["kind"], len(p["alerts"]),
                        "" if "hit" not in p else ("có" if p["hit"] else "không"), p.get("latency") or ""])


def crop_around(frame, box, scale=3.0, min_px=96):
    """Cắt vùng quanh vật bị báo, rộng gấp scale lần khung để còn thấy tay và mặt bàn."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(max(x2 - x1, y2 - y1) * scale, min_px) / 2
    H, W = frame.shape[:2]
    return frame[int(max(cy - half, 0)):int(min(cy + half, H)), int(max(cx - half, 0)):int(min(cx + half, W))]


def save_images(video, out, alerts, tile=160, cols=6):
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"Không mở được {video}, bỏ qua phần lưu ảnh")
        return
    tiles = []
    for a in alerts:
        cap.set(cv2.CAP_PROP_POS_MSEC, a["t"] * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        x1, y1, x2, y2 = map(int, a["box"])
        img = frame.copy()
        if a["student_box"] is not None:
            sx1, sy1, sx2, sy2 = map(int, a["student_box"])
            cv2.rectangle(img, (sx1, sy1), (sx2, sy2), (0, 215, 255), 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        label = f"{fmt(a['t'])} {a['cls']} {a['conf']:.2f} {LABELS[a['verdict']]}"
        cv2.putText(img, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
        folder = out / a["verdict"]
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / f"{fmt(a['t']).replace(':', '')}_{a['cls']}_{a['conf']:.2f}.jpg"), img)
        if a["verdict"] == "bao_sai":
            c = crop_around(frame, a["box"])
            if c.size:
                c = cv2.resize(c, (tile, tile))
                cv2.rectangle(c, (0, tile - 22), (tile, tile), (0, 0, 0), -1)
                cv2.putText(c, f"{fmt(a['t'])} {a['cls'][:5]} {a['conf']:.2f}", (4, tile - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
                tiles.append(c)
    cap.release()
    if tiles:
        rows = [tiles[i:i + cols] for i in range(0, len(tiles), cols)]
        blank = np.zeros_like(tiles[0])
        sheet = np.vstack([np.hstack(r + [blank] * (cols - len(r))) for r in rows])
        cv2.imwrite(str(out / "tong_hop_bao_sai.jpg"), sheet)
        print(f"Ảnh: {out}/bao_sai, {out}/bat_duoc; tổng hợp báo sai: {out / 'tong_hop_bao_sai.jpg'}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    from settings import ROOT, load_config

    ap = argparse.ArgumentParser(description="Chấm lại buổi nghiệm thu trên video theo file đáp án scenes.csv")
    ap.add_argument("--dets", required=True, help="dets.jsonl do ai_service.main --dump-dets ghi ra")
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--video", help="video gốc; có thì lưu ảnh từng cảnh báo")
    ap.add_argument("--out", help="thư mục kết quả (mặc định: cạnh file dets, thư mục review)")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--target", default="phone")
    ap.add_argument("--alert-class", nargs="+", help="ghi đè rules.alert_classes")
    ap.add_argument("--min-confidence", type=float)
    ap.add_argument("--window-s", type=float)
    ap.add_argument("--min-ratio", type=float)
    ap.add_argument("--cooldown-s", type=float)
    args = ap.parse_args()

    r = load_config(args.config)["rules"]
    params = {"alert_classes": args.alert_class or r["alert_classes"],
              "min_confidence": args.min_confidence if args.min_confidence is not None else r["min_confidence"],
              "window_s": args.window_s if args.window_s is not None else r["window_s"],
              "min_ratio": args.min_ratio if args.min_ratio is not None else r["min_ratio"],
              "cooldown_s": args.cooldown_s if args.cooldown_s is not None else r["cooldown_s"]}
    meta, frames = load_dets(args.dets)
    scenes = load_scenes(*args.scenes).get(meta["video"])
    if not scenes:
        sys.exit(f"Video {meta['video']} không có trong {args.scenes}: sửa cột video trong scenes.csv cho đúng tên file")
    print(f"{meta['video']}: {len(frames)} khung, {len(scenes)} cảnh, ngưỡng phát hiện lúc chạy {meta['threshold']}")

    alerts = run_alerts(frames, **params)
    alerts, per_scene = classify(alerts, scenes, args.target)
    print_report(alerts, per_scene, params, args.target)
    out = Path(args.out) if args.out else Path(args.dets).parent / "review"
    write_csv(out, alerts, per_scene)
    print(f"Bảng chi tiết: {out / 'alerts.csv'}, {out / 'scenes_result.csv'}")
    if args.video:
        save_images(args.video, out, alerts)


if __name__ == "__main__":
    main()
