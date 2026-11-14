"""Gửi sự kiện và trạng thái: qua MQTT khi chạy trên laptop, hoặc ghi ra file JSONL khi chạy thử (--dry-run)."""
import base64
import json
from pathlib import Path, PurePosixPath

LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")


def should_send_snapshot(mqtt_cfg):
    """auto: Mosquitto ở máy khác (Raspberry Pi) thì backend cũng ở đó, phải gửi ảnh kèm sự kiện."""
    value = mqtt_cfg.get("send_snapshot", "auto")
    if isinstance(value, str) and value.lower() == "auto":
        return str(mqtt_cfg.get("host", "127.0.0.1")).lower() not in LOCAL_HOSTS
    return bool(value)


class EventPublisher:
    def __init__(self, cfg, dry_run=False, out_dir="runs", snapshot_root=None):
        self.topic_events = cfg["mqtt"]["topic_events"]
        self.topic_status = cfg["mqtt"]["topic_status"]
        self.dry_run = dry_run
        self.snapshot_root = Path(snapshot_root) if snapshot_root else None
        self.send_snapshot = (not dry_run) and self.snapshot_root is not None and should_send_snapshot(cfg["mqtt"])
        if dry_run:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            self.events_file = open(Path(out_dir) / "events.jsonl", "a", encoding="utf-8")
            self.client = None
        else:
            import paho.mqtt.client as mqtt

            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"ai-{cfg['camera']['id']}")
            self.client.connect_async(cfg["mqtt"]["host"], int(cfg["mqtt"]["port"]), keepalive=30)
            self.client.loop_start()  # tự kết nối lại nếu Mosquitto khởi động sau hoặc bị ngắt

    def with_snapshot(self, payload):
        """Thêm ảnh bằng chứng dạng base64 để backend trên Pi tự lưu lại. Không đọc được ảnh thì gửi sự kiện không kèm ảnh."""
        if not self.send_snapshot or not payload.get("snapshot"):
            return payload
        try:
            rel = PurePosixPath(payload["snapshot"]).parts[1:]   # "snapshots/<ngày>/<id>.jpg" -> <snapshot_root>/<ngày>/<id>.jpg
            data = self.snapshot_root.joinpath(*rel).read_bytes()
        except OSError as e:
            print("[event] không đọc được ảnh bằng chứng:", e)
            return payload
        return {**payload, "snapshot_b64": base64.b64encode(data).decode("ascii")}

    def event(self, payload):
        text = json.dumps(payload, ensure_ascii=False)
        if self.dry_run:
            self.events_file.write(text + "\n")
            self.events_file.flush()
        else:
            self.client.publish(self.topic_events, json.dumps(self.with_snapshot(payload), ensure_ascii=False), qos=1)
        print("[event]", text)

    def status(self, payload):
        if not self.dry_run:
            self.client.publish(self.topic_status, json.dumps(payload), qos=0)

    def close(self):
        if self.dry_run:
            self.events_file.close()
        else:
            self.client.loop_stop()
            self.client.disconnect()
