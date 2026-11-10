"""Tên lớp đúng với cả rfdetr cũ (class_names dạng dict) và mới (dạng list):  python tests/test_class_names.py"""
import json
import os
import runpy
import sys
import tempfile
import types

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ai_service.detector import class_labels, name_from_id  # noqa: E402

NEW = ["calculator", "phone", "student"]                       # rfdetr mới (Colab 10/11)
OLD = {1: "calculator", 2: "phone", 3: "student"}              # rfdetr cũ


class FakeDet:
    """Giống supervision.Detections ở những chỗ code dùng tới. class_id là id COCO của Roboflow (lớp thật từ 1)."""

    def __init__(self, class_id, conf, data=None):
        self.class_id = np.array(class_id)
        self.confidence = np.array(conf, dtype=float)
        self.xyxy = np.array([[10.0 * i, 10.0, 10.0 * i + 30, 50.0] for i in range(len(class_id))])
        self.data = data or {}

    def __len__(self):
        return len(self.class_id)


def test_name_from_id_list_and_dict_agree():
    for cid, name in OLD.items():
        assert name_from_id(NEW, cid) == name_from_id(OLD, cid) == name
    assert name_from_id(NEW, 0) == "0"                         # lớp cha Roboflow, không phải lớp thật
    assert name_from_id(NEW, 4) == "4"


def test_class_labels_prefers_names_given_by_rfdetr():
    det = FakeDet([3, 2], [0.9, 0.8], {"class_name": np.array(["student", "phone"])})
    assert class_labels(det, NEW) == ["student", "phone"]
    assert class_labels(FakeDet([3, 2, 1], [0.9, 0.8, 0.7]), NEW) == ["student", "phone", "calculator"]


def run_export(names):
    """Chạy tools/export_preds_rfdetr.py với một rfdetr giả trả class_names dạng `names`."""
    class FakeModel:
        def __init__(self, **kw):
            self.class_names = names

        def predict(self, img, threshold=0.5):
            return FakeDet([3, 2, 0, 1], [0.95, 0.6, 0.3, 0.05])  # có cả một khung lớp cha (id 0)

    sys.modules["rfdetr"] = types.SimpleNamespace(RFDETRMedium=FakeModel, RFDETRSmall=FakeModel, RFDETRNano=FakeModel)
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(2):
            Image.new("RGB", (64, 64)).save(os.path.join(tmp, f"a{i}.jpg"))
        out = os.path.join(tmp, "pred.json")
        sys.argv = ["export", "--images", tmp, "--weights", "w.pth", "--out", out]
        runpy.run_path(os.path.join(ROOT, "tools", "export_preds_rfdetr.py"), run_name="__main__")
        with open(out, encoding="utf-8") as f:
            return json.load(f)


def test_export_same_labels_for_old_and_new_rfdetr():
    for names in (NEW, OLD):
        preds = run_export(names)
        assert [p["class_name"] for p in preds[:3]] == ["student", "phone", "calculator"], names
        assert len(preds) == 6 and all(p["class_name"] != "0" for p in preds)   # bỏ khung lớp cha


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK ", name)
