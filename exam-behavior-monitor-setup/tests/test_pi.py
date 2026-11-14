"""Kiểm thử phần Raspberry Pi (không cần Pi, dùng chân giả của gpiozero):  python tests/test_pi.py
Cần: pip install gpiozero fastapi paho-mqtt pyyaml"""
import base64
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ai_service.publisher import EventPublisher, should_send_snapshot  # noqa: E402

JPG = b"\xff\xd8\xff\xe0" + b"0" * 100


# ---------- laptop gửi ảnh kèm sự kiện ----------
def test_send_snapshot_auto():
    assert not should_send_snapshot({"host": "127.0.0.1"})
    assert not should_send_snapshot({"host": "localhost", "send_snapshot": "auto"})
    assert should_send_snapshot({"host": "192.168.1.50"})
    assert should_send_snapshot({"host": "127.0.0.1", "send_snapshot": True})
    assert not should_send_snapshot({"host": "192.168.1.50", "send_snapshot": False})


def test_publisher_attaches_snapshot():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "data" / "snapshots"
        (root / "2026-11-14").mkdir(parents=True)
        (root / "2026-11-14" / "room1-1.jpg").write_bytes(JPG)
        pub = object.__new__(EventPublisher)   # không mở kết nối MQTT thật
        pub.snapshot_root, pub.send_snapshot = root, True
        ev = {"event_id": "room1-1", "snapshot": "snapshots/2026-11-14/room1-1.jpg"}
        out = pub.with_snapshot(ev)
        assert base64.b64decode(out["snapshot_b64"]) == JPG
        assert "snapshot_b64" not in ev, "không được sửa sự kiện gốc (dòng in ra và file dry-run)"
        assert "snapshot_b64" not in pub.with_snapshot({**ev, "snapshot": "snapshots/2026-11-14/khong-co.jpg"})
        pub.send_snapshot = False
        assert pub.with_snapshot(ev) is ev


# ---------- backend trên Pi lưu ảnh ----------
def test_backend_saves_snapshot():
    from backend.app import save_snapshot

    with tempfile.TemporaryDirectory() as d:
        e = {"snapshot": "snapshots/2026-11-14/room1-20261114-005012-001.jpg", "snapshot_b64": base64.b64encode(JPG).decode()}
        assert save_snapshot(e, d)
        assert (Path(d) / "2026-11-14" / "room1-20261114-005012-001.jpg").read_bytes() == JPG
        assert "snapshot_b64" not in e, "không được lưu chuỗi ảnh vào cơ sở dữ liệu"
        for bad_path in ("../x.jpg", "snapshots/2026-11-14/../../x.jpg", "snapshots/abc/x.jpg", "/etc/x.jpg"):
            assert not save_snapshot({"snapshot": bad_path, "snapshot_b64": base64.b64encode(JPG).decode()}, d), bad_path
        assert not save_snapshot({"snapshot": "snapshots/2026-11-14/a.jpg", "snapshot_b64": base64.b64encode(b"not jpeg").decode()}, d)
        assert not save_snapshot({"snapshot": "snapshots/2026-11-14/a.jpg", "snapshot_b64": "@@@"}, d)
        assert not save_snapshot({"snapshot": "snapshots/2026-11-14/a.jpg"}, d)
        assert sorted(p.name for p in Path(d).rglob("*.jpg")) == ["room1-20261114-005012-001.jpg"]


# ---------- trạm báo trên Pi ----------
class FakeApi:
    def __init__(self):
        self.calls, self.status, self.ok = [], {}, True

    def set_status(self, event_id, status):
        self.calls.append((event_id, status))
        return self.ok

    def get_status(self, event_id):
        return self.status.get(event_id, "new")


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def make_station(buzzer_pin=None):
    from gpiozero import Device
    from gpiozero.pins.mock import MockFactory, MockPWMPin

    if Device.pin_factory is not None:
        Device.pin_factory.reset()
    Device.pin_factory = MockFactory(pin_class=MockPWMPin)
    from pi.station import Station, make_devices

    api, clock = FakeApi(), Clock()
    devices = make_devices({"red": 17, "yellow": 27, "green": 22, "confirm": 5, "dismiss": 6}, buzzer_pin)
    st = Station({"pi_station": {"status_timeout_s": 15}}, api, devices, clock)
    return st, api, clock, devices


def close(devices):
    for dev in devices.values():
        if dev is not None:
            dev.close()


def press(button):
    button.pin.drive_low()
    time.sleep(0.15)
    button.pin.drive_high()
    time.sleep(0.15)


def test_station_alerts_and_buttons():
    st, api, clock, d = make_station()
    assert d["buzzer"] is None, "mặc định không dùng còi"
    st.on_event({"event_id": "e1", "type": "phone"})
    assert st.red_mode == "flash", "cảnh báo mới: đèn đỏ nháy nhanh"
    clock.t += 3
    st.tick()
    assert st.red_mode == "on" and d["red"].is_lit, "hết 3 giây thì sáng liền"
    st.on_event({"event_id": "e1", "type": "phone"})   # MQTT gửi lại (qos 1) thì không tính hai lần
    assert st.red_mode == "on"
    st.on_event({"event_id": "e2", "type": "head_turn"})
    assert st.red_mode == "flash" and list(st.pending) == ["e1", "e2"]
    clock.t += 3
    st.tick()
    assert st.red_mode == "blink", "còn 2 cảnh báo: nháy chậm"
    press(d["confirm"])
    assert api.calls == [("e1", "confirmed")] and st.red_mode == "on"
    press(d["dismiss"])
    assert api.calls[-1] == ("e2", "dismissed") and st.red_mode == "off" and not d["red"].is_lit
    press(d["confirm"])                                  # hết cảnh báo: nhấn không gọi backend
    assert len(api.calls) == 2
    st.on_event({"event_id": "e3", "type": "phone"})
    press(d["confirm"])                                  # xác nhận ngay khi đang nháy nhanh
    assert not st.pending and st.red_mode == "off"
    api.ok = False                                        # backend không trả lời: giữ cảnh báo lại
    st.on_event({"event_id": "e4", "type": "phone"})
    clock.t += 3
    press(d["confirm"])
    assert list(st.pending) == ["e4"] and st.red_mode == "on"
    api.status["e4"] = "confirmed"                        # giám thị xử lý trên web
    st.sync()
    assert not st.pending and st.red_mode == "off"
    close(d)


def test_station_status_lights():
    st, api, clock, d = make_station()
    assert st.tick() == "no_ai" and not d["green"].is_lit
    st.on_status({"fps": 24, "camera_ok": True})
    assert st.tick() == "ok" and d["green"].is_lit and not d["yellow"].is_lit
    st.on_status({"fps": 0, "camera_ok": False})
    assert st.tick() == "no_camera" and d["yellow"].is_lit and not d["green"].is_lit
    st.on_status({"fps": 24, "camera_ok": True})
    clock.t += 14
    assert st.tick() == "ok"
    clock.t += 2                                          # quá 15 giây không có trạng thái
    assert st.tick() == "no_ai" and not d["green"].is_lit
    st.on_status({"camera_ok": True})
    assert st.tick() == "ok" and d["green"].is_lit
    close(d)


def test_station_optional_buzzer():
    st, api, clock, d = make_station(buzzer_pin=4)       # muốn thêm còi sau này
    beeps = []
    st.buzzer.beep = lambda *a, **k: beeps.append(k)
    st.tick()
    assert not beeps, "mới bật Pi, chưa từng có AI thì không kêu"
    st.on_event({"event_id": "e1", "type": "phone"})
    assert len(beeps) == 1
    st.on_status({"camera_ok": True})
    st.tick()
    clock.t += 16
    st.tick()
    assert len(beeps) == 2
    clock.t += 5
    st.tick()
    assert len(beeps) == 2, "kêu nhắc mỗi 10 giây, không kêu liên tục"
    clock.t += 5
    st.tick()
    assert len(beeps) == 3
    close(d)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK ", name)
