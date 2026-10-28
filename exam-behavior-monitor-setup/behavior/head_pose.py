"""Phát hiện quay đầu (nghi nhìn bài) bằng MediaPipe Face Landmarker. Việc P1, phải thử trên video thật trước.

Tải mô hình (khoảng 4 MB) về weights/face_landmarker.task:
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

Thử nhanh trên video, in tỉ lệ lệch của từng thí sinh để chọn ngưỡng:
    python -m behavior.head_pose --video videos/mau_000.mp4
"""
import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parent.parent
NOSE_TIP, EYE_OUTER_R, EYE_OUTER_L = 1, 33, 263


class HeadTurnDetector:
    def __init__(self, model_path=ROOT / "weights" / "face_landmarker.task", yaw_ratio=0.35, min_duration_s=3.0,
                 cooldown_s=30.0, min_face_px=40, head_fraction=0.45):
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE, num_faces=1)
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self.yaw_ratio = yaw_ratio
        self.min_duration_s = min_duration_s
        self.cooldown_s = cooldown_s
        self.min_face_px = min_face_px
        self.head_fraction = head_fraction
        self.state = {}

    def head_crop(self, frame, box):
        x1, y1, x2, y2 = map(int, box)
        h = int((y2 - y1) * self.head_fraction)
        pad = int(0.1 * (x2 - x1))
        H, W = frame.shape[:2]
        return frame[max(y1 - pad, 0):min(y1 + h, H), max(x1 - pad, 0):min(x2 + pad, W)]

    def ratio(self, crop_bgr):
        if crop_bgr.size == 0 or min(crop_bgr.shape[:2]) < self.min_face_px:
            return None
        rgb = np.ascontiguousarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        result = self.landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not result.face_landmarks:
            return None
        lm = result.face_landmarks[0]
        eye_w = abs(lm[EYE_OUTER_L].x - lm[EYE_OUTER_R].x)
        if eye_w < 1e-6:
            return None
        return (lm[NOSE_TIP].x - (lm[EYE_OUTER_L].x + lm[EYE_OUTER_R].x) / 2) / eye_w

    def analyze(self, t, frame, persons):
        events = []
        for track_id, box in persons:
            r = self.ratio(self.head_crop(frame, box))
            st = self.state.setdefault(track_id, {"since": None, "next_ok": float("-inf")})
            if r is None or abs(r) < self.yaw_ratio:
                st["since"] = None
                continue
            if st["since"] is None:
                st["since"] = t
            duration = t - st["since"]
            if duration >= self.min_duration_s and t >= st["next_ok"]:
                events.append({"track_id": track_id, "box": box, "duration_s": round(duration, 1),
                               "confidence": round(min(1.0, abs(r) / (2 * self.yaw_ratio)), 2), "ratio": round(r, 2)})
                st["next_ok"] = t + self.cooldown_s
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
    det = RFDETRDetector(str(ROOT / m["weights"]), m.get("size", "medium"), m.get("threshold", 0.3))
    tracker = IouTracker()
    head = HeadTurnDetector()
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    i = 0
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
                r = head.ratio(head.head_crop(frame, d.box))
                found.append(f"#{d.track_id}:{'không thấy mặt' if r is None else f'{r:+.2f}'}")
        print(f"{i / fps:6.1f}s  " + "  ".join(found))


if __name__ == "__main__":
    main()
