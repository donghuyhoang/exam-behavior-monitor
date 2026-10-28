"""Luật vi phạm: biến kết quả phát hiện theo từng khung hình thành sự kiện dùng điện thoại."""
from collections import deque
from dataclasses import dataclass


@dataclass
class Det:
    track_id: int
    cls: str
    conf: float
    box: tuple


@dataclass
class Alert:
    cls: str
    conf: float
    box: tuple
    track_id: int
    student_track_id: object
    student_box: object
    duration_s: float
    payload: object = None


def center(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def assign_student(obj, students):
    if not students:
        return None
    cx, cy = center(obj.box)
    inside = [s for s in students if s.box[0] <= cx <= s.box[2] and s.box[1] <= cy <= s.box[3]]
    pool = inside or students
    return min(pool, key=lambda s: (center(s.box)[0] - cx) ** 2 + (center(s.box)[1] - cy) ** 2)


class ViolationRules:
    def __init__(self, alert_classes=("phone",), min_confidence=0.5, window_s=2.0, min_ratio=0.6,
                 cooldown_s=30.0, person_class="student"):
        self.alert_classes = set(alert_classes)
        self.min_confidence = min_confidence
        self.window_s = window_s
        self.min_ratio = min_ratio
        self.cooldown_s = cooldown_s
        self.person_class = person_class
        self.frames = deque()
        self.tracks = {}

    def update(self, t, dets, payload=None):
        self.frames.append(t)
        while self.frames and t - self.frames[0] > self.window_s:
            self.frames.popleft()

        students = [d for d in dets if d.cls == self.person_class]
        seen = set()
        for d in dets:
            if d.cls not in self.alert_classes or d.conf < self.min_confidence:
                continue
            s = assign_student(d, students)
            key = f"{d.cls}:s{s.track_id}" if s is not None else f"{d.cls}:t{d.track_id}"
            st = self.tracks.get(key)
            if key not in seen:
                if st is None or t - st["last"] > self.window_s:
                    st = {"hits": deque(), "first": t, "best": None,
                          "next_ok": st["next_ok"] if st else float("-inf")}
                    self.tracks[key] = st
                st["hits"].append(t)
                st["last"] = t
                seen.add(key)
            if st["best"] is None or d.conf > st["best"][0]:
                st["best"] = (d.conf, d, s, payload)

        alerts = []
        for key, st in list(self.tracks.items()):
            while st["hits"] and t - st["hits"][0] > self.window_s:
                st["hits"].popleft()
            if not st["hits"]:
                if t - st["last"] > 2 * self.window_s and t >= st["next_ok"]:
                    del self.tracks[key]
                continue
            ratio = len(st["hits"]) / max(len(self.frames), 1)
            if (t - st["first"] >= self.window_s and ratio >= self.min_ratio
                    and t >= st["next_ok"] and st["best"] is not None):
                conf, d, s, frame = st["best"]
                alerts.append(Alert(cls=d.cls, conf=round(conf, 3), box=d.box, track_id=d.track_id,
                                    student_track_id=s.track_id if s is not None else None,
                                    student_box=s.box if s is not None else None,
                                    duration_s=round(t - st["first"], 1), payload=frame))
                st["next_ok"] = t + self.cooldown_s
                st["best"] = None
        return alerts
