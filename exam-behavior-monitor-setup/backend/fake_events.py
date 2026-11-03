"""Phát sự kiện và trạng thái giả qua MQTT để làm backend/giao diện khi chưa có dịch vụ AI.

    python -m backend.fake_events --every 3                    # điện thoại
    python -m backend.fake_events --every 3 --type head_turn   # quay đầu (nhìn bài)
    python -m backend.fake_events --every 3 --type mix         # trộn ngẫu nhiên ba loại
"""
import argparse
import json
import random
import time
from datetime import datetime

import paho.mqtt.client as mqtt
from PIL import Image, ImageDraw

from settings import ROOT, load_config


TYPES = ("phone", "calculator", "head_turn")


def make_snapshot(path, box, image=None, label="phone"):
    img = Image.open(image).convert("RGB") if image else Image.new("RGB", (1280, 720), (70, 76, 84))
    x, y, w, h = box
    d = ImageDraw.Draw(img)
    d.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=4)
    d.text((x, max(y - 16, 0)), f"{label} (fake)", fill=(255, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=85)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=float, default=5.0, help="số giây giữa hai cảnh báo")
    ap.add_argument("--count", type=int, default=0, help="số cảnh báo, 0 = chạy mãi")
    ap.add_argument("--image", help="ảnh nền cho ảnh bằng chứng")
    ap.add_argument("--type", choices=TYPES + ("mix",), default="phone", help="loại vi phạm; mix = trộn ngẫu nhiên")
    args = ap.parse_args()

    cfg = load_config()
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="pbl4-fake")
    client.connect(cfg["mqtt"]["host"], int(cfg["mqtt"]["port"]), keepalive=30)
    client.loop_start()
    snap_root = ROOT / cfg["output"]["snapshot_dir"]
    n, next_event = 0, time.time()
    try:
        while args.count == 0 or n < args.count:
            now = datetime.now().astimezone()
            client.publish(cfg["mqtt"]["topic_status"], json.dumps({
                "camera_id": cfg["camera"]["id"], "ts": now.isoformat(timespec="seconds"),
                "fps": round(random.uniform(12, 18), 1), "latency_ms": random.randint(120, 400), "camera_ok": True}))
            if time.time() >= next_event:
                n += 1
                event_id = f"{cfg['camera']['room']}-{now:%Y%m%d-%H%M%S}-{n:03d}"
                kind = random.choice(TYPES) if args.type == "mix" else args.type
                if kind == "head_turn":   # khung quanh đầu và vai thí sinh, to hơn khung điện thoại
                    box = [random.randint(100, 1000), random.randint(80, 400), random.randint(90, 140), random.randint(110, 170)]
                else:
                    box = [random.randint(100, 1000), random.randint(100, 500), random.randint(30, 70), random.randint(40, 90)]
                rel = f"snapshots/{now:%Y-%m-%d}/{event_id}.jpg"
                make_snapshot(snap_root / f"{now:%Y-%m-%d}" / f"{event_id}.jpg", box, args.image, kind)
                event = {"event_id": event_id, "camera_id": cfg["camera"]["id"], "ts": now.isoformat(timespec="seconds"),
                         "type": kind, "confidence": round(random.uniform(0.5, 0.97), 2),
                         "track_id": random.randint(1, 40), "student_track_id": random.randint(1, 20),
                         "bbox": box, "duration_s": round(random.uniform(2.0, 4.0), 1), "snapshot": rel}
                client.publish(cfg["mqtt"]["topic_events"], json.dumps(event, ensure_ascii=False), qos=1)
                print("[fake] đã gửi", event_id, kind)
                next_event = time.time() + args.every
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
