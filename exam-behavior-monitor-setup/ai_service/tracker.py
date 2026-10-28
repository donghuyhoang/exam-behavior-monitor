"""Gán track_id cho từng khung phát hiện qua các khung hình bằng cách ghép theo IoU."""
import numpy as np


def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class IouTracker:
    def __init__(self, iou_threshold=0.3, max_lost=30):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.tracks = {}
        self.next_id = 1

    def update(self, xyxy, class_id):
        xyxy = np.asarray(xyxy, dtype=float).reshape(-1, 4)
        class_id = np.asarray(class_id).reshape(-1)
        ids = np.zeros(len(xyxy), dtype=int)

        pairs = []
        for tid, tr in self.tracks.items():
            for j in range(len(xyxy)):
                if class_id[j] == tr["cls"]:
                    iou = box_iou(tr["box"], xyxy[j])
                    if iou >= self.iou_threshold:
                        pairs.append((iou, tid, j))
        used_tracks, unmatched = set(), set(range(len(xyxy)))
        for _, tid, j in sorted(pairs, reverse=True):
            if tid in used_tracks or j not in unmatched:
                continue
            used_tracks.add(tid)
            unmatched.discard(j)
            ids[j] = tid
            self.tracks[tid].update(box=xyxy[j], lost=0)

        for tid in list(self.tracks):
            if tid not in used_tracks:
                self.tracks[tid]["lost"] += 1
                if self.tracks[tid]["lost"] > self.max_lost:
                    del self.tracks[tid]
        for j in sorted(unmatched):
            ids[j] = self.next_id
            self.tracks[self.next_id] = {"box": xyxy[j], "cls": class_id[j], "lost": 0}
            self.next_id += 1
        return ids
