"""Dịch vụ AI: đọc camera, phát hiện bằng RF-DETR, theo dõi từng đối tượng, áp luật vi phạm, gửi sự kiện.

Trên laptop nhóm trưởng (camera thật, gửi MQTT, phát video ở cổng 8001):
    python -m ai_service.main
    python -m ai_service.main --show            # mở thêm cửa sổ xem trực tiếp, bấm q để thoát

Trên Colab hoặc máy không có camera (đọc file, ghi runs/events.jsonl, ảnh bằng chứng và video kết quả):
    python -m ai_service.main --source videos/mau_000.mp4 --dry-run --save-video runs/mau_000_out.mp4
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2

from ai_service.detector import RFDETRDetector
from ai_service.publisher import EventPublisher
from ai_service.rules import Det, ViolationRules
from ai_service.stream import FrameSource
from ai_service.tracker import IouTracker
from settings import ROOT, load_config

COLORS = {"student": (110, 170, 60), "phone": (0, 140, 255), "calculator": (210, 160, 0)}  # BGR
RED = (0, 0, 255)


def to_dets(det, tracker, name_of):
    """supervision.Detections của RF-DETR -> danh sách Det có track_id."""
    if len(det) == 0 or det.class_id is None:
        tracker.update([], [])
        return []
    ids = tracker.update(det.xyxy, det.class_id)
    # Ưu tiên tên lớp rfdetr gắn sẵn vào kết quả (cách đã dùng khi chấm benchmark), không có mới tra theo class_id
    names = (det.data or {}).get("class_name")
    if names is None or len(names) != len(det):
        names = [name_of(cid) for cid in det.class_id]
    return [Det(int(tid), str(name), float(conf), tuple(float(v) for v in box))
            for box, name, conf, tid in zip(det.xyxy, names, det.confidence, ids)]


class DetectionSummary:
    """Gom số khung và độ tin cậy cao nhất của từng lớp, in mỗi vài giây để biết mô hình đang thấy gì."""

    def __init__(self, classes=("student", "phone", "calculator")):
        self.classes = classes
        self.stats = {}

    def add(self, dets):
        per_frame = {}
        for d in dets:
            n, best = per_frame.get(d.cls, (0, 0.0))
            per_frame[d.cls] = (n + 1, max(best, d.conf))
        for cls, (n, best) in per_frame.items():
            most, top = self.stats.get(cls, (0, 0.0))
            self.stats[cls] = (max(most, n), max(top, best))

    def line(self, t):
        names = list(self.classes) + [c for c in self.stats if c not in self.classes]
        parts = [f"{c}: tối đa {self.stats[c][0]}/khung, tin cậy cao nhất {self.stats[c][1]:.2f}" if c in self.stats
                 else f"{c}: không thấy" for c in names]
        self.stats = {}
        return f"[{t:7.1f}s] " + " | ".join(parts)


def draw(frame, dets, red_boxes=(), banner=""):
    img = frame.copy()
    for d in dets:
        x1, y1, x2, y2 = map(int, d.box)
        color = COLORS.get(d.cls, (200, 200, 200))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{d.cls} {d.conf:.2f} #{d.track_id}", (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    for box in red_boxes:
        if box is not None:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), RED, 4)
    if banner:
        cv2.putText(img, banner, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, RED if "ALERT" in banner else (255, 255, 255),
                    2, cv2.LINE_AA)
    return img


def xywh(box):
    x1, y1, x2, y2 = box
    return [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]


class EventMaker:
    """Tạo sự kiện đúng mẫu JSON chung và lưu ảnh bằng chứng."""

    def __init__(self, cfg, snapshot_root):
        self.cfg = cfg
        self.snapshot_root = Path(snapshot_root)
        self.seq = 0

    def make(self, type_, conf, box, track_id, duration_s, evidence_img, student_track_id=None, video_t=None):
        now = datetime.now().astimezone()
        self.seq += 1
        event_id = f"{self.cfg['camera']['room']}-{now:%Y%m%d-%H%M%S}-{self.seq:03d}"
        day = now.strftime("%Y-%m-%d")
        path = self.snapshot_root / day / f"{event_id}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), evidence_img)
        event = {
            "event_id": event_id,
            "camera_id": self.cfg["camera"]["id"],
            "ts": now.isoformat(timespec="seconds"),
            "type": type_,
            "confidence": round(float(conf), 3),
            "track_id": int(track_id),
            "student_track_id": student_track_id,
            "bbox": xywh(box),
            "duration_s": duration_s,
            "snapshot": f"snapshots/{day}/{event_id}.jpg",
        }
        if video_t is not None:
            event["video_t"] = video_t   # giây thứ mấy trong file video, để so với kịch bản quay
        return event


def main():
    ap = argparse.ArgumentParser(description="Dịch vụ AI phát hiện vi phạm")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--source", help="ghi đè camera.source: file video, địa chỉ RTSP hoặc 0")
    ap.add_argument("--dry-run", action="store_true", help="không dùng MQTT/MJPEG, ghi sự kiện ra runs/events.jsonl")
    ap.add_argument("--out", default="runs", help="thư mục kết quả khi --dry-run")
    ap.add_argument("--save-video", help="ghi video đã vẽ khung ra file .mp4")
    ap.add_argument("--show", action="store_true", help="mở cửa sổ xem trực tiếp (bấm q để thoát)")
    ap.add_argument("--dump-dets", help="lưu kết quả phát hiện từng khung ra file .jsonl để chỉnh ngưỡng bằng tools/tune_rules.py")
    ap.add_argument("--threshold", type=float, help="ghi đè model.threshold (khi chỉnh ngưỡng nên dùng 0.2)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    source = FrameSource(args.source if args.source is not None else cfg["camera"]["source"])
    m = cfg["model"]
    threshold = args.threshold if args.threshold is not None else m.get("threshold", 0.3)
    detector = RFDETRDetector(str(ROOT / m["weights"]), m.get("size", "medium"), threshold, m.get("optimize", True))
    dump = None
    if args.dump_dets:
        Path(args.dump_dets).parent.mkdir(parents=True, exist_ok=True)
        dump = open(args.dump_dets, "w", encoding="utf-8")
        dump.write(json.dumps({"video": Path(str(source.source)).name, "fps": source.fps, "threshold": threshold}) + "\n")
    fps_guess = source.fps if source.fps and source.fps < 120 else 15
    tracker = IouTracker(max_lost=int(2 * fps_guess))   # mất dấu quá khoảng 2 giây thì coi là đối tượng mới
    r = cfg["rules"]
    rules = ViolationRules(r["alert_classes"], r["min_confidence"], r["window_s"], r["min_ratio"], r["cooldown_s"])
    publisher = EventPublisher(cfg, dry_run=args.dry_run, out_dir=args.out)
    snapshot_root = Path(args.out) / "snapshots" if args.dry_run else ROOT / cfg["output"]["snapshot_dir"]
    events = EventMaker(cfg, snapshot_root)

    mjpeg = None
    if not args.dry_run:
        from ai_service.mjpeg import MjpegServer
        mjpeg = MjpegServer(cfg["output"]["mjpeg_port"], cfg["output"].get("stream_width", 1280))

    behavior = None
    b = cfg.get("behavior", {})
    if b.get("enabled"):
        from behavior.head_pose import HeadTurnDetector
        behavior = HeadTurnDetector(yaw_ratio=b["yaw_ratio"], min_duration_s=b["min_duration_s"], cooldown_s=b["cooldown_s"])

    writer = None
    summary, last_summary = DetectionSummary(), None
    fps_ema, last_status, last_tick = 0.0, 0.0, time.perf_counter()
    alert_until, alert_boxes = 0.0, []
    try:
        while True:
            ok, frame, t, captured_at = source.read()
            if not ok:
                if not source.live:
                    break  # hết file video
                publisher.status({"camera_id": cfg["camera"]["id"], "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
                                  "fps": 0.0, "latency_ms": None, "camera_ok": False})
                continue

            dets = to_dets(detector(frame), tracker, detector.name_of)
            summary.add(dets)
            if dump:
                dump.write(json.dumps({"t": round(t, 3), "d": [[d.track_id, d.cls, round(d.conf, 3)] + [int(v) for v in d.box]
                                                                for d in dets]}) + "\n")
            video_t = None if source.live else round(t, 1)

            for a in rules.update(t, dets, payload=frame):
                evidence = draw(a.payload, [], [a.box, a.student_box], f"ALERT: {a.cls} {a.conf:.2f}")
                publisher.event(events.make(a.cls, a.conf, a.box, a.track_id, a.duration_s, evidence, a.student_track_id,
                                            video_t))
                alert_until, alert_boxes = t + 3.0, [a.box, a.student_box]

            if behavior is not None and source.frame_idx % b.get("every_n_frames", 5) == 0:
                persons = [(d.track_id, d.box) for d in dets if d.cls == "student"]
                for e in behavior.analyze(t, frame, persons):
                    evidence = draw(frame, [], [e["box"]], "ALERT: head turn")
                    publisher.event(events.make("head_turn", e["confidence"], e["box"], e["track_id"],
                                                e["duration_s"], evidence, e["track_id"], video_t))

            now_tick = time.perf_counter()
            fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / max(now_tick - last_tick, 1e-6)) if fps_ema else 1.0 / max(now_tick - last_tick, 1e-6)
            last_tick = now_tick
            alerting = t < alert_until
            vis = draw(frame, dets, alert_boxes if alerting else (), f"ALERT  {fps_ema:.1f} FPS" if alerting else f"OK  {fps_ema:.1f} FPS")

            if mjpeg:
                mjpeg.update(vis)
            if args.save_video:
                if writer is None:
                    Path(args.save_video).parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*"mp4v"),
                                             source.fps if source.fps else 15, (vis.shape[1], vis.shape[0]))
                writer.write(vis)
            if args.show:
                cv2.imshow("PBL4", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if last_summary is None:
                last_summary = t
            elif t - last_summary >= cfg["output"]["status_every_s"]:
                last_summary = t
                print(summary.line(t), flush=True)
            if time.time() - last_status >= cfg["output"]["status_every_s"]:
                last_status = time.time()
                publisher.status({"camera_id": cfg["camera"]["id"],
                                  "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
                                  "fps": round(fps_ema, 1), "latency_ms": int((time.time() - captured_at) * 1000),
                                  "camera_ok": source.camera_ok})
    except KeyboardInterrupt:
        pass
    finally:
        source.close()
        publisher.close()
        if dump:
            dump.close()
        if writer is not None:
            writer.release()
        if mjpeg:
            mjpeg.close()
        if args.show:
            cv2.destroyAllWindows()
        print(f"Đã xử lý {source.frame_idx} khung, FPS trung bình gần nhất {fps_ema:.1f}")


if __name__ == "__main__":
    main()
