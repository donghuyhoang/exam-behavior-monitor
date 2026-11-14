"""Trạm báo cho giám thị trên Raspberry Pi: 3 đèn, 2 nút (không dùng còi).

  Đỏ nháy nhanh 3 giây       : vừa có cảnh báo mới
  Đỏ sáng liền / nháy chậm   : còn 1 / từ 2 cảnh báo trở lên chưa xử lý
  Nút Xác nhận / Bỏ qua      : xử lý cảnh báo cũ nhất, ghi confirmed / dismissed vào backend
  Xanh sáng                  : dịch vụ AI trên laptop đang chạy
  Vàng sáng liền             : AI vẫn chạy nhưng mất camera
  Vàng nháy                  : mất AI (laptop tắt, treo, mất mạng) hoặc laptop chưa chạy AI

Chạy trên Pi, trong thư mục exam-behavior-monitor-setup:
    .venv/bin/python -m pi.station               # pi/setup.sh đã cài thành dịch vụ pbl4-station
    .venv/bin/python -m pi.station --self-test   # thử dây: lần lượt sáng từng đèn, in ra khi nhấn nút
Muốn thêm còi sau này: ghi pi_station.buzzer: 4 trong config.yaml (còi 3 chân, VCC nối 3.3V, I/O nối GPIO4).
"""
import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from settings import load_config  # noqa: E402

DEFAULT_PINS = {"red": 17, "yellow": 27, "green": 22, "confirm": 5, "dismiss": 6}
FLASH_S = 3          # cảnh báo mới: đèn đỏ nháy nhanh trong 3 giây
WARN_EVERY_S = 10    # chỉ dùng khi có còi


class Api:
    """Gọi API của backend chạy cùng máy Pi."""

    def __init__(self, base, timeout=3):
        self.base = base.rstrip("/")
        self.timeout = timeout

    def _call(self, method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read() or b"null")

    def get_status(self, event_id):
        """Trạng thái hiện tại của cảnh báo; "gone" nếu backend không còn; None nếu không gọi được backend."""
        try:
            return self._call("GET", f"/api/events/{event_id}")["status"]
        except urllib.error.HTTPError as e:
            return "gone" if e.code == 404 else None
        except (OSError, ValueError, KeyError):
            return None

    def set_status(self, event_id, status):
        try:
            self._call("PATCH", f"/api/events/{event_id}", {"status": status})
            return True
        except (OSError, ValueError) as e:
            print(f"[station] không ghi được {status} cho {event_id}: {e}", flush=True)
            return False


def make_devices(pins, buzzer_pin=None):
    from gpiozero import LED, Button, Buzzer

    return {
        "red": LED(pins["red"]), "yellow": LED(pins["yellow"]), "green": LED(pins["green"]),
        "buzzer": Buzzer(buzzer_pin, active_high=False) if buzzer_pin is not None else None,  # còi S8550 kêu ở mức thấp
        "confirm": Button(pins["confirm"], pull_up=True, bounce_time=0.05),
        "dismiss": Button(pins["dismiss"], pull_up=True, bounce_time=0.05),
    }


class Station:
    def __init__(self, cfg, api, devices=None, clock=time.monotonic):
        sc = cfg.get("pi_station", {})
        self.timeout = float(sc.get("status_timeout_s", 15))
        pins = {**DEFAULT_PINS, **sc.get("pins", {})}
        d = devices or make_devices(pins, sc.get("buzzer"))
        self.red, self.yellow, self.green = d["red"], d["yellow"], d["green"]
        self.buzzer = d.get("buzzer")
        self.api, self.clock = api, clock
        self.lock = threading.Lock()
        self.pending = OrderedDict()   # event_id -> loại cảnh báo, cũ nhất đứng đầu
        self.red_mode = "off"
        self.flash_until = 0.0
        self.last_status = None        # thời điểm nhận trạng thái gần nhất từ dịch vụ AI
        self.camera_ok = True
        self.seen_ai = False
        self.state = None
        self.last_warn = 0.0
        d["confirm"].when_pressed = lambda: self.resolve("confirmed")
        d["dismiss"].when_pressed = lambda: self.resolve("dismissed")

    def beep(self, **kw):
        if self.buzzer is not None:
            self.buzzer.beep(background=True, **kw)

    # ---------- tin nhắn MQTT ----------
    def on_event(self, data):
        event_id = data.get("event_id")
        if not event_id:
            return
        with self.lock:
            if event_id in self.pending:
                return
            self.pending[event_id] = data.get("type")
            self.flash_until = self.clock() + FLASH_S
        print(f"[station] cảnh báo {data.get('type')} {event_id}", flush=True)
        self.beep(on_time=0.15, off_time=0.1, n=3)
        self.update_red()

    def on_status(self, data):
        with self.lock:
            self.last_status = self.clock()
            self.camera_ok = bool(data.get("camera_ok", True))
            self.seen_ai = True

    # ---------- nút bấm ----------
    def resolve(self, status):
        with self.lock:
            event_id = next(iter(self.pending), None)
        if event_id is None:
            return
        if self.api.set_status(event_id, status):
            with self.lock:
                self.pending.pop(event_id, None)
            print(f"[station] {status}: {event_id}", flush=True)
        self.update_red()   # backend không trả lời thì cảnh báo vẫn còn, đèn đỏ vẫn sáng

    def update_red(self):
        with self.lock:
            n = len(self.pending)
            flashing = self.clock() < self.flash_until
        if n == 0:
            mode = "off"
        elif flashing:
            mode = "flash"
        else:
            mode = "on" if n == 1 else "blink"
        if mode == self.red_mode:
            return
        self.red_mode = mode
        if mode == "off":
            self.red.off()
        elif mode == "on":
            self.red.on()
        elif mode == "flash":
            self.red.blink(on_time=0.1, off_time=0.1)
        else:
            self.red.blink(on_time=0.5, off_time=0.5)

    def sync(self):
        """Giám thị đã xử lý trên giao diện web thì trạm cũng bỏ cảnh báo đó."""
        with self.lock:
            ids = list(self.pending)
        for event_id in ids:
            status = self.api.get_status(event_id)
            if status is not None and status != "new":
                with self.lock:
                    self.pending.pop(event_id, None)
                print(f"[station] đã xử lý trên web ({status}): {event_id}", flush=True)
        self.update_red()

    # ---------- đèn trạng thái ----------
    def tick(self):
        now = self.clock()
        self.update_red()   # hết 3 giây nháy nhanh thì chuyển sang sáng liền
        with self.lock:
            last, camera_ok, seen_ai = self.last_status, self.camera_ok, self.seen_ai
        if last is None or now - last > self.timeout:
            state = "no_ai"
        elif not camera_ok:
            state = "no_camera"
        else:
            state = "ok"
        if state != self.state:
            self.state = state
            print(f"[station] trạng thái: {state}", flush=True)
            if state == "ok":
                self.yellow.off()
                self.green.on()
            elif state == "no_camera":
                self.green.off()
                self.yellow.on()
            else:
                self.green.off()
                self.yellow.blink(on_time=0.5, off_time=0.5)
            self.last_warn = now - WARN_EVERY_S
        if self.buzzer is not None and state != "ok" and seen_ai and now - self.last_warn >= WARN_EVERY_S:
            self.last_warn = now
            self.beep(on_time=0.08, off_time=0.12, n=2)
        return state


def run(cfg):
    import paho.mqtt.client as mqtt

    sc = cfg.get("pi_station", {})
    station = Station(cfg, Api(sc.get("api", "http://127.0.0.1:8000")))
    topic_events, topic_status = cfg["mqtt"]["topic_events"], cfg["mqtt"]["topic_status"]

    def on_message(client, userdata, msg):
        try:
            data = json.loads(msg.payload)
        except ValueError:
            return
        if msg.topic == topic_events:
            station.on_event(data)
        elif msg.topic == topic_status:
            station.on_status(data)

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="pbl4-pi-station")
    client.on_connect = lambda c, u, f, rc, p: c.subscribe([(topic_events, 1), (topic_status, 0)])
    client.on_message = on_message
    client.connect_async(cfg["mqtt"]["host"], int(cfg["mqtt"]["port"]), keepalive=30)
    client.loop_start()

    for led in (station.red, station.yellow, station.green):   # bật lên: 3 đèn cùng sáng 1 giây
        led.on()
    station.beep(on_time=0.2, off_time=0.1, n=1)
    time.sleep(1)
    for led in (station.red, station.yellow, station.green):
        led.off()

    print("[station] đang chạy, Ctrl+C để dừng", flush=True)
    last_sync = 0.0
    try:
        while True:
            station.tick()
            if time.monotonic() - last_sync >= 3:
                last_sync = time.monotonic()
                station.sync()
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()
        client.disconnect()


def self_test(cfg):
    sc = cfg.get("pi_station", {})
    pins = {**DEFAULT_PINS, **sc.get("pins", {})}
    d = make_devices(pins, sc.get("buzzer"))
    for name, vi in (("red", "đỏ"), ("yellow", "vàng"), ("green", "xanh")):
        print(f"đèn {vi} (GPIO{pins[name]})", flush=True)
        d[name].on()
        time.sleep(1)
        d[name].off()
    if d["buzzer"] is not None:
        print(f"còi (GPIO{sc['buzzer']}): kêu 3 tiếng", flush=True)
        d["buzzer"].beep(on_time=0.2, off_time=0.2, n=3, background=False)
    d["confirm"].when_pressed = lambda: print("nhấn nút Xác nhận", flush=True)
    d["dismiss"].when_pressed = lambda: print("nhấn nút Bỏ qua", flush=True)
    print("Nhấn thử 2 nút trong 20 giây...", flush=True)
    time.sleep(20)
    print("Xong thử dây.")


def main():
    ap = argparse.ArgumentParser(description="Trạm báo cho giám thị trên Raspberry Pi")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--self-test", action="store_true", help="thử dây đèn, nút rồi thoát")
    args = ap.parse_args()
    cfg = load_config(args.config)
    self_test(cfg) if args.self_test else run(cfg)


if __name__ == "__main__":
    main()
