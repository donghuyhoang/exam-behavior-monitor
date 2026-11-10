"""Kiểm thử code tuần 9/11 (không cần GPU, camera, VLC hay MediaPipe):  python tests/test_tools_tuan3.py"""
import ctypes
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools.eval_head_turn import simulate  # noqa: E402
from tools.monitor_ram import label_of, summarize  # noqa: E402
from tools.review_alerts import classify, crop_around, run_alerts, summary  # noqa: E402
from tools.tune_rules import load_dets  # noqa: E402


# ---------- review_alerts ----------
def test_classify_hits_false_alarms_and_latency():
    scenes = [(0, 600, "normal"), (616, 626, "phone"), (631, 687, "normal"), (692, 702, "phone")]
    alerts = [{"t": 120.0}, {"t": 619.5}, {"t": 628.0}, {"t": 650.0}, {"t": 900.0}]
    alerts, per_scene = classify(alerts, scenes)
    assert [a["verdict"] for a in alerts] == ["bao_sai", "bat_duoc", "bat_duoc", "bao_sai", "ngoai_kich_ban"]
    phone = [p for p in per_scene if p["kind"] == "phone"]
    assert phone[0]["hit"] and phone[0]["latency"] == 3.5     # 628 nằm trong 3 giây cho phép nên vẫn tính cảnh 1
    assert not phone[1]["hit"] and phone[1]["latency"] is None
    s = summary(alerts, per_scene)
    assert (s["hits"], s["scenes"], s["false_alarms"], s["outside"]) == (1, 2, 2, 1)


def fake_dets(path):
    """60 giây ở 10 khung/giây: điện thoại thật 10–20 s, một hộp bút bị nhận nhầm 40–45 s."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"video": "fake.mp4", "fps": 10, "threshold": 0.2}) + "\n")
        for i in range(600):
            t = i / 10
            d = [[1, "student", 0.9, 100, 100, 300, 400], [5, "student", 0.9, 400, 100, 600, 400]]
            if 10 <= t < 20:
                d.append([2, "phone", 0.7, 180, 250, 220, 300])
            if 40 <= t < 45:
                d.append([3, "calculator", 0.5, 480, 250, 520, 290])
            f.write(json.dumps({"t": t, "d": d}) + "\n")


def test_run_alerts_on_saved_detections():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "dets.jsonl")
        fake_dets(p)
        _, frames = load_dets(p)
    alerts = run_alerts(frames, ["phone", "calculator"], 0.4, 2.0, 0.5, 30)
    assert [(a["cls"], a["student"]) for a in alerts] == [("phone", 1), ("calculator", 5)]
    assert abs(alerts[0]["t"] - 12.0) < 0.15                    # cần đủ window_s = 2 giây mới báo
    alerts, per_scene = classify(alerts, [(0, 9, "normal"), (10, 20, "phone"), (21, 60, "normal")])
    assert per_scene[1]["latency"] == 2.0 and alerts[1]["verdict"] == "bao_sai"
    assert run_alerts(frames, ["phone"], 0.4, 2.0, 0.5, 30)[0]["cls"] == "phone"
    assert len(run_alerts(frames, ["phone"], 0.4, 2.0, 0.5, 30)) == 1     # bỏ calculator thì hết báo sai


def test_review_cli_writes_tables():
    with tempfile.TemporaryDirectory() as tmp:
        dets, scenes, out = (os.path.join(tmp, n) for n in ("dets.jsonl", "scenes.csv", "review"))
        fake_dets(dets)
        with open(scenes, "w", encoding="utf-8") as f:
            f.write("video,bat_dau,ket_thuc,loai\nfake.mp4,00:00,00:09,normal\nfake.mp4,00:10,00:20,phone\n"
                    "fake.mp4,00:21,01:00,normal\n")
        r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "review_alerts.py"), "--dets", dets,
                            "--scenes", scenes, "--out", out, "--alert-class", "phone", "calculator",
                            "--min-confidence", "0.4", "--window-s", "2", "--min-ratio", "0.5", "--cooldown-s", "30"],
                           capture_output=True, text=True, encoding="utf-8")
        assert r.returncode == 0, r.stderr
        assert "bắt được 1/1 cảnh phone" in r.stdout and "báo sai 1" in r.stdout and "calculator 1" in r.stdout
        with open(os.path.join(out, "alerts.csv"), encoding="utf-8-sig") as f:
            lines = f.read().splitlines()
        assert len(lines) == 3 and "bắt được" in lines[1] and "báo sai" in lines[2]


def test_crop_around_stays_inside_frame():
    import numpy as np
    frame = np.zeros((720, 1280, 3), np.uint8)
    assert crop_around(frame, (0, 0, 10, 10)).shape[:2] == (53, 53)          # sát mép ảnh thì bị cắt bớt
    assert crop_around(frame, (600, 300, 640, 340)).shape[:2] == (120, 120)


# ---------- eval_head_turn ----------
def test_simulate_head_turn_needs_duration_and_resets():
    samples = [(t / 2, [(7, 0.5 if 2 <= t / 2 < 6 else 0.1), (8, None)]) for t in range(40)]
    ev = simulate(samples, yaw_ratio=0.35, min_duration_s=3.0, cooldown_s=30)
    assert [(e["t"], e["track_id"]) for e in ev] == [(5.0, 7)]
    assert simulate(samples, yaw_ratio=0.6, min_duration_s=1.0, cooldown_s=30) == []
    assert len(simulate(samples, yaw_ratio=0.35, min_duration_s=1.0, cooldown_s=2)) == 2   # cooldown ngắn thì báo lại


def test_head_turn_score_cli():
    with tempfile.TemporaryDirectory() as tmp:
        ratios, scenes = os.path.join(tmp, "turn.jsonl"), os.path.join(tmp, "scenes.csv")
        with open(ratios, "w", encoding="utf-8") as f:
            f.write(json.dumps({"video": "v.mp4", "method": "pose", "every": 5}) + "\n")
            for i in range(240):                                   # 120 giây, 2 mẫu mỗi giây
                t = i / 2
                r = 0.45 if 30 <= t < 35 or 80 <= t < 85 else (0.3 if 60 <= t < 64 else 0.05)
                f.write(json.dumps({"t": t, "r": [[1, r], [2, None]]}) + "\n")
        with open(scenes, "w", encoding="utf-8") as f:
            f.write("video,bat_dau,ket_thuc,loai\nv.mp4,00:00,00:29,normal\nv.mp4,00:30,00:35,head_turn\n"
                    "v.mp4,00:36,01:19,normal\nv.mp4,01:20,01:25,head_turn\nv.mp4,01:26,02:00,normal\n")
        r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "eval_head_turn.py"), "score",
                            "--ratios", ratios, "--scenes", scenes], capture_output=True, text=True, encoding="utf-8")
        assert r.returncode == 0, r.stderr
        # 0.3 trong 4 giây ở cảnh normal là nhìn lệch nhẹ: bộ được chọn phải bắt đủ 2 cảnh mà không báo nó
        assert "đo được góc 240/480" in r.stdout and "2/2" in r.stdout
        assert "yaw_ratio: 0.4, min_duration_s: 3.0" in r.stdout


# ---------- monitor_ram ----------
def test_monitor_labels_and_summary():
    assert label_of("python.exe", ["python", "-m", "ai_service.main", "--show"]) == "ai"
    assert label_of("python.exe", ["python", "-m", "uvicorn", "backend.app:app"]) == "backend"
    assert label_of("mosquitto.exe", None) == "mosquitto" and label_of("chrome.exe", ["chrome"]) is None
    steady = [(t, "ai", 1500 + (t % 3)) for t in range(0, 1800, 10)]
    leak = [(t, "backend", 200 + t / 3) for t in range(0, 1800, 10)]   # tăng đều ~200 MB mỗi 10 phút
    s = summarize(steady + leak)
    assert s["ai"]["ket_luan"] == "ổn định" and s["backend"]["ket_luan"].startswith("RAM tăng")
    assert s["ai"]["phut"] == 29.8


# ---------- stream: LibVLC giả lập ----------
CAM = {"on": True, "fps": 20}


class _FakePlayer:
    def __init__(self):
        self.stopped = False

    def set_media(self, m):
        pass

    def video_set_callbacks(self, lock, unlock, display, opaque):
        self.cb = (lock, unlock, display)

    def video_set_format(self, chroma, w, h, pitch):
        self.size = w * h * 3

    def video_get_size(self, num=0):
        return (1280, 720)

    def play(self):
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        lock, unlock, display = self.cb
        i = 0
        while not self.stopped and CAM["on"]:      # rút điện: phiên RTSP chết hẳn, VLC không tự nối lại
            planes = [None]
            lock(None, planes)
            ctypes.memset(planes[0], i % 256, self.size)
            unlock(None, None, planes)
            i += 1
            time.sleep(1 / CAM["fps"])

    def stop(self):
        self.stopped = True

    def release(self):
        pass


def install_fake_vlc():
    vlc = types.ModuleType("vlc")
    vlc.CallbackDecorators = types.SimpleNamespace(VideoLockCb=lambda f: f, VideoUnlockCb=lambda f: f,
                                                   VideoDisplayCb=lambda f: f)
    vlc.Instance = lambda *a: types.SimpleNamespace(media_player_new=_FakePlayer, media_new=lambda url: url,
                                                    release=lambda: None)
    sys.modules["vlc"] = vlc


def test_vlc_reader_returns_only_new_frames_and_reports_stall():
    install_fake_vlc()
    from ai_service.stream import _VLCCap
    CAM["on"] = True
    cap = _VLCCap("rtsp://cam/onvif1", width=64, height=36, first_frame_s=2, stall_s=0.5)
    assert cap.isOpened() and cap.native_size() == (1280, 720)
    t0, values = time.time(), set()
    for _ in range(10):
        ok, frame = cap.read()
        assert ok and frame.shape == (36, 64, 3)
        values.add(int(frame[0, 0, 0]))
    assert len(values) >= 9                         # mỗi lần đọc là một khung mới, không lặp khung cũ
    assert time.time() - t0 >= 0.35                 # 10 khung ở 20 khung/giây: phải chờ camera chứ không quay vòng
    CAM["on"] = False
    time.sleep(0.1)
    assert cap.read() == (False, None)              # hết khung mới quá stall_s thì báo mất camera
    cap.release()


def test_frame_source_reports_lost_camera_and_reconnects():
    install_fake_vlc()
    import ai_service.stream as stream
    stream.VLC_FIRST_FRAME_S, stream.VLC_STALL_S = 0.5, 0.5
    CAM["on"] = True
    src = stream.FrameSource("rtsp://cam/onvif1", reconnect_s=0.2)
    try:
        assert src.read()[0] and src.camera_ok
        CAM["on"] = False
        t0 = time.time()
        while src.camera_ok and time.time() - t0 < 3:
            src.read(timeout=0.3)
        assert not src.camera_ok, "rút điện camera mà không báo mất kết nối"
        CAM["on"] = True
        t0 = time.time()
        while not (src.read(timeout=0.3)[0] and src.camera_ok) and time.time() - t0 < 5:
            pass
        assert src.camera_ok, "cắm lại camera mà không tự chạy lại"
    finally:
        CAM["on"] = False
        src.close()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK ", name)
