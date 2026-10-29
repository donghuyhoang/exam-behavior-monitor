"""Kiểm thử công cụ chỉnh ngưỡng bằng video giả: python tests/test_tune_rules.py"""
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools.tune_rules import mmss, score  # noqa: E402


def test_mmss():
    assert mmss("01:30") == 90 and mmss("90") == 90 and mmss("0:00:05") == 5


def test_score_counts_hits_false_alarms_and_outside():
    scenes = [(30, 45, "phone"), (45, 120, "normal"), (120, 135, "phone"), (135, 200, "normal")]
    hits, misses, fa, outside, normal_min = score([33, 47.5, 80, 250], scenes)
    # 33 trúng cảnh phone 1; 47.5 nằm trong 3 giây sau cảnh phone 1 nên vẫn tính là trúng; 80 là báo sai;
    # 250 nằm ngoài kịch bản; cảnh phone thứ 2 bị bỏ sót
    assert (hits, misses, fa, outside) == (1, 1, 1, 1) and abs(normal_min - (75 + 65) / 60) < 1e-9


def test_end_to_end_on_fake_video():
    fps, frames = 10, []
    for i in range(600):                                   # 60 giây
        t = i / fps
        d = [[1, "student", 0.9, 100, 100, 300, 400]]
        if 10 <= t < 20 or 40 <= t < 42.5:                 # 2 cảnh điện thoại: 10 giây và 2,5 giây
            d.append([2, "phone", 0.55, 180, 250, 220, 300])
        if 30 <= t < 30.8:                                  # thoáng qua 0,8 giây ở cảnh bình thường: không được báo
            d.append([3, "phone", 0.7, 180, 250, 220, 300])
        frames.append({"t": t, "d": d})
    with tempfile.TemporaryDirectory() as tmp:
        dets = os.path.join(tmp, "dets.jsonl")
        with open(dets, "w", encoding="utf-8") as f:
            f.write(json.dumps({"video": "fake.mp4", "fps": fps, "threshold": 0.2}) + "\n")
            f.writelines(json.dumps(r) + "\n" for r in frames)
        scenes = os.path.join(tmp, "scenes.csv")
        with open(scenes, "w", encoding="utf-8") as f:
            f.write("video,bat_dau,ket_thuc,loai\nfake.mp4,00:10,00:20,phone\nfake.mp4,00:20,00:40,normal\n"
                    "fake.mp4,00:40,00:42.5,phone\nfake.mp4,00:42.5,01:00,normal\n")
        out = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "tune_rules.py"), "--dets", dets,
                              "--scenes", scenes, "--top", "3"], capture_output=True, text=True, encoding="utf-8")
        assert out.returncode == 0, out.stderr
        first_row = [line for line in out.stdout.splitlines() if "|" in line][1]
        assert "2/2" in first_row and first_row.split("|")[1].split()[1] == "0", out.stdout
        print(out.stdout)

        # Phương án báo cả máy tính: điện thoại bị nhận thành calculator vẫn được tính là bắt được cảnh phone
        with open(dets, "w", encoding="utf-8") as f:
            f.write(json.dumps({"video": "fake.mp4", "fps": fps, "threshold": 0.2}) + "\n")
            for r in frames:
                f.write(json.dumps({"t": r["t"], "d": [d if d[1] != "phone" else [d[0], "calculator"] + d[2:]
                                                       for d in r["d"]]}) + "\n")
        only_phone = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "tune_rules.py"), "--dets", dets,
                                     "--scenes", scenes, "--top", "1"], capture_output=True, text=True, encoding="utf-8")
        both = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "tune_rules.py"), "--dets", dets,
                               "--scenes", scenes, "--top", "1", "--alert-class", "phone", "calculator"],
                              capture_output=True, text=True, encoding="utf-8")
        assert "0/2" in [l for l in only_phone.stdout.splitlines() if "|" in l][1], only_phone.stdout
        assert "2/2" in [l for l in both.stdout.splitlines() if "|" in l][1], both.stdout


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")               # in bảng kết quả tiếng Việt khi ghi ra file trên Windows
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK ", name)
