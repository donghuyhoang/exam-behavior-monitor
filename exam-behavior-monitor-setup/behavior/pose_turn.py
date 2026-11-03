"""Phát hiện quay đầu bằng MediaPipe Pose Landmarker: dùng khi Face Landmarker không thấy mặt vì camera ở xa.

Pose nhìn cả người nên vẫn chạy được khi mặt chỉ còn vài chục pixel. Ước lượng góc quay từ mũi và hai tai:
nhìn thẳng thì mũi nằm giữa hai tai; quay ngang thì mũi lệch hẳn về một bên và một tai bị che (visibility thấp).

Tải mô hình (khoảng 6 MB) về weights/pose_landmarker_lite.task:
    https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task

Thử nhanh trên video, in tỉ lệ lệch của từng thí sinh để chọn ngưỡng:
    python -m behavior.pose_turn --video videos/buoi_quay2.mp4

Bật trong dịch vụ AI: config.yaml -> behavior: enabled: true, method: pose
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
NOSE, LEFT_EAR, RIGHT_EAR, LEFT_SHOULDER, RIGHT_SHOULDER = 0, 7, 8, 11, 12


def turn_ratio(lm, min_visibility=0.3):
    """Độ lệch của mũi so với giữa hai tai, chia cho khoảng cách hai tai (âm/dương tùy hướng quay).

    lm: danh sách điểm mốc có .x, .y, .visibility (toạ độ chuẩn hoá 0..1 trong ảnh cắt).
    Một tai bị che hẳn thì dùng hai vai làm mốc ngang thay cho hai tai.
    Trả về None nếu không đủ điểm tin cậy.
    """
    nose = lm[NOSE]
    if getattr(nose, "visibility", 1.0) < min_visibility:
        return None
    le, re_ = lm[LEFT_EAR], lm[RIGHT_EAR]
    ears_ok = min(getattr(le, "visibility", 1.0), getattr(re_, "visibility", 1.0)) >= min_visibility
    if ears_ok and abs(le.x - re_.x) > 1e-3:
        mid, width = (le.x + re_.x) / 2, abs(le.x - re_.x)
    else:
        ls, rs = lm[LEFT_SHOULDER], lm[RIGHT_SHOULDER]
        if min(getattr(ls, "visibility", 1.0), getattr(rs, "visibility", 1.0)) < min_visibility:
            return None
        # vai rộng hơn đầu khoảng 2,5 lần: quy về cùng thang đo với hai tai
        mid, width = (ls.x + rs.x) / 2, abs(ls.x - rs.x) / 2.5
        if width < 1e-3:
            return None
    return (nose.x - mid) / width


class PoseTurnDetector:
    """Cùng giao diện với behavior.head_pose.HeadTurnDetector: analyze(t, frame, persons) -> danh sách sự kiện."""

    def __init__(self, model_path=ROOT / "weights" / "pose_landmarker_lite.task", yaw_ratio=0.35,
                 min_duration_s=3.0, cooldown_s=30.0, min_person_px=40, upscale_to=256, landmarker=None):
        if landmarker is None:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            options = vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.IMAGE, num_poses=1)
            landmarker = vision.PoseLandmarker.create_from_options(options)
        self.landmarker = landmarker
        self.yaw_ratio = yaw_ratio
        self.min_duration_s = min_duration_s
        self.cooldown_s = cooldown_s
        self.min_person_px = min_person_px
        self.upscale_to = upscale_to
        self.state = {}

    @staticmethod
    def person_crop(frame, box, pad_frac=0.1):
        x1, y1, x2, y2 = map(int, box)
        pad = int(pad_frac * max(x2 - x1, y2 - y1))
        H, W = frame.shape[:2]
        return frame[max(y1 - pad, 0):min(y2 + pad, H), max(x1 - pad, 0):min(x2 + pad, W)]

    def landmarks(self, crop_bgr):
        if crop_bgr.size == 0 or min(crop_bgr.shape[:2]) < self.min_person_px:
            return None
        short = min(crop_bgr.shape[:2])
        if self.upscale_to and short < self.upscale_to:
            k = self.upscale_to / short
            crop_bgr = cv2.resize(crop_bgr, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
        import mediapipe as mp

        rgb = np.ascontiguousarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        result = self.landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        return result.pose_landmarks[0] if result.pose_landmarks else None

    def ratio(self, crop_bgr):
        lm = self.landmarks(crop_bgr)
        return None if lm is None else turn_ratio(lm)

    def update(self, t, track_id, box, r):
        """Áp luật thời gian cho một thí sinh; tách riêng để kiểm thử không cần MediaPipe."""
        st = self.state.setdefault(track_id, {"since": None, "next_ok": float("-inf")})
        if r is None or abs(r) < self.yaw_ratio:
            st["since"] = None
            return None
        if st["since"] is None:
            st["since"] = t
        duration = t - st["since"]
        if duration >= self.min_duration_s and t >= st["next_ok"]:
            st["next_ok"] = t + self.cooldown_s
            return {"track_id": track_id, "box": box, "duration_s": round(duration, 1),
                    "confidence": round(min(1.0, abs(r) / (2 * self.yaw_ratio)), 2), "ratio": round(r, 2)}
        return None

    def analyze(self, t, frame, persons):
        events = []
        for track_id, box in persons:
            e = self.update(t, track_id, box, self.ratio(self.person_crop(frame, box)))
            if e:
                events.append(e)
        return events


def main():
    from ai_service.detector import RFDETRDetector
    from ai_service.main import to_dets
    from ai_service.tracker import IouTracker
    from settings import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--every", type=int, default=10, help="phân tích 1 khung trong N khung")
    args = ap.parse_args()
    m = load_config()["model"]
    det = RFDETRDetector(str(ROOT / m["weights"]), m.get("size", "medium"), m.get("threshold", 0.3),
                         resolution=m.get("resolution"))
    tracker = IouTracker()
    pose = PoseTurnDetector()
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    i, seen, found_n = 0, 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        dets = to_dets(det(frame), tracker, det.name_of)
        if i % args.every:
            continue
        found = []
        for d in dets:
            if d.cls == "student":
                r = pose.ratio(pose.person_crop(frame, d.box))
                seen += 1
                found_n += r is not None
                found.append(f"#{d.track_id}:{'không thấy' if r is None else f'{r:+.2f}'}")
        print(f"{i / fps:6.1f}s  " + "  ".join(found))
    print(f"\nĐo được góc quay {found_n}/{seen} lần ({100 * found_n / max(seen, 1):.0f}%).")


if __name__ == "__main__":
    main()
