"""Phát sự kiện và trạng thái giả qua MQTT để làm backend/giao diện khi chưa có dịch vụ AI."""
import argparse
import json
import random
import time
from datetime import datetime

import paho.mqtt.client as mqtt
from PIL import Image, ImageDraw

from settings import ROOT, load_config


def make_snapshot(path, box, image=None):
    img = Image.open(image).convert("RGB") if image else Image.new("RGB", (1280, 720), (70, 76, 84))
    x, y, w, h = box
    d = ImageDraw.Draw(img)
    d.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=4)
    d.text((x, max(y - 16, 0)), "phone (fake)", fill=(255, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=85)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=float, default=5.0, help="số giây giữa hai cảnh báo")
    ap.add_argument("--count", type=int, default=0, help="số cảnh báo, 0 = chạy mãi")
    ap.add_argument("--image", help="ảnh nền cho ảnh bằng chứng")
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
                box = [random.randint(100, 1000), random.randint(100, 500), random.randint(30, 70), random.randint(40, 90)]
                rel = f"snapshots/{now:%Y-%m-%d}/{event_id}.jpg"
                make_snapshot(snap_root / f"{now:%Y-%m-%d}" / f"{event_id}.jpg", box, args.image)
                event = {"event_id": event_id, "camera_id": cfg["camera"]["id"], "ts": now.isoformat(timespec="seconds"),
                         "type": "phone", "confidence": round(random.uniform(0.5, 0.97), 2),
                         "track_id": random.randint(1, 40), "student_track_id": random.randint(1, 20),
                         "bbox": box, "duration_s": round(random.uniform(2.0, 4.0), 1), "snapshot": rel}
                client.publish(cfg["mqtt"]["topic_events"], json.dumps(event, ensure_ascii=False), qos=1)
                print("[fake] đã gửi", event_id)
                next_event = time.time() + args.every
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
