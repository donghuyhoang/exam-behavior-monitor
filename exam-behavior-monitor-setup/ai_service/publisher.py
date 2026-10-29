"""Gửi sự kiện và trạng thái: qua MQTT khi chạy trên laptop, hoặc ghi ra file JSONL khi chạy thử (--dry-run)."""
import json
from pathlib import Path


class EventPublisher:
    def __init__(self, cfg, dry_run=False, out_dir="runs"):
        self.topic_events = cfg["mqtt"]["topic_events"]
        self.topic_status = cfg["mqtt"]["topic_status"]
        self.dry_run = dry_run
        if dry_run:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            self.events_file = open(Path(out_dir) / "events.jsonl", "a", encoding="utf-8")
            self.client = None
        else:
            import paho.mqtt.client as mqtt

            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"ai-{cfg['camera']['id']}")
            self.client.connect_async(cfg["mqtt"]["host"], int(cfg["mqtt"]["port"]), keepalive=30)
            self.client.loop_start()  # tự kết nối lại nếu Mosquitto khởi động sau hoặc bị ngắt

    def event(self, payload):
        text = json.dumps(payload, ensure_ascii=False)
        if self.dry_run:
            self.events_file.write(text + "\n")
            self.events_file.flush()
        else:
            self.client.publish(self.topic_events, text, qos=1)
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
