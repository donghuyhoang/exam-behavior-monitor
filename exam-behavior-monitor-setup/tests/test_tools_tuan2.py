"""Kiểm thử các công cụ tuần 2/11 (không cần GPU, MediaPipe hay rfdetr):  python tests/test_tools_tuan2.py"""
import json
import os
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from behavior.pose_turn import LEFT_EAR, LEFT_SHOULDER, NOSE, RIGHT_EAR, RIGHT_SHOULDER, PoseTurnDetector, turn_ratio  # noqa: E402
from tools.check_split import check, original_stem, parse_frame  # noqa: E402
from tools.cut_frames import frame_name, mmss, split_of  # noqa: E402
from tools.eval_crops import crop_predictions, expand_box, map_back, nms  # noqa: E402


# ---------- cut_frames ----------
def test_mmss_and_split_by_time():
    assert mmss("08:15") == 495 and mmss("1:02:03") == 3723 and mmss("75") == 75
    assert split_of(494, valid_from=495) == "train" and split_of(495, valid_from=495) == "valid"
    assert split_of(10, all_to="test") == "test"
    assert frame_name("videos/buoi_quay2.mp4", 495.2) == "buoi_quay2_s000495.jpg"


# ---------- check_split ----------
def test_roboflow_names_are_parsed():
    assert original_stem("buoi_quay2_s000495_jpg.rf.ab12cd.jpg") == "buoi_quay2_s000495"
    assert parse_frame("buoi_quay2_s000495_jpg.rf.ab12cd.jpg") == ("buoi_quay2", 495.0)
    assert parse_frame("test1_mp4-0016_jpg.rf.147aef.jpg", sample_fps=0.5) == ("test1_mp4", 32.0)
    assert parse_frame("IMG_2034_jpg.rf.99.jpg") == (None, None)


def _write_split(root, split, files, size=(640, 360)):
    os.makedirs(os.path.join(root, split), exist_ok=True)
    coco = {"categories": [{"id": 0, "name": "x"}, {"id": 2, "name": "phone"}],
            "images": [{"id": i, "file_name": f, "width": size[0], "height": size[1]} for i, f in enumerate(files)],
            "annotations": [{"id": 1, "image_id": 0, "category_id": 2, "bbox": [1, 1, 5, 5]}] if files else []}
    with open(os.path.join(root, split, "_annotations.coco.json"), "w", encoding="utf-8") as f:
        json.dump(coco, f)


def test_good_split_passes():
    with tempfile.TemporaryDirectory() as root:
        _write_split(root, "train", ["buoi_quay2_s000010_jpg.rf.a.jpg", "buoi_quay2_s000400_jpg.rf.b.jpg", "web1_jpg.rf.c.jpg"])
        _write_split(root, "valid", ["buoi_quay2_s000500_jpg.rf.d.jpg"])
        _write_split(root, "test", ["test1_mp4-0016_jpg.rf.e.jpg"])
        errors, warnings, _ = check(root)
        assert errors == [] and warnings == []


def test_same_video_in_train_and_test_is_error():
    with tempfile.TemporaryDirectory() as root:
        _write_split(root, "train", ["nghiemthu1_s000100_jpg.rf.a.jpg"])
        _write_split(root, "test", ["nghiemthu1_s000300_jpg.rf.b.jpg"])
        errors, _, _ = check(root)
        assert len(errors) == 1 and "nghiemthu1" in errors[0]


def test_interleaved_train_valid_and_resized_frames_warn():
    with tempfile.TemporaryDirectory() as root:
        _write_split(root, "train", ["v_s000010_jpg.rf.a.jpg", "v_s000600_jpg.rf.b.jpg"], size=(512, 512))
        _write_split(root, "valid", ["v_s000300_jpg.rf.c.jpg"])
        errors, warnings, _ = check(root)
        assert errors == []
        assert any("xen kẽ" in w for w in warnings) and any("512x512" in w for w in warnings)


# ---------- eval_crops ----------
def test_expand_and_map_back():
    assert expand_box([100, 100, 200, 300], 1.5, 1000, 1000) == (75, 50, 225, 350)
    assert expand_box([0, 0, 100, 100], 2.0, 120, 120) == (0, 0, 120, 120)   # cắt theo biên ảnh
    assert map_back([10, 20, 30, 40], (75, 50)) == [85, 70, 105, 90]


def test_nms_keeps_best_per_class():
    dets = [("phone", 0.9, [0, 0, 10, 10]), ("phone", 0.5, [1, 1, 11, 11]), ("calculator", 0.6, [0, 0, 10, 10])]
    kept = nms(dets)
    assert [d[1] for d in kept] == [0.9, 0.6]


def test_crop_predictions_maps_to_full_image():
    class FakeImage:
        size = (640, 360)

        def crop(self, box):
            return SimpleNamespace(box=box)

    def predict(crop):   # mô hình giả: luôn thấy một điện thoại ở toạ độ (5,5)-(15,25) của ảnh cắt, và một thí sinh
        return [("phone", 0.8, [5, 5, 15, 25]), ("student", 0.9, [0, 0, 50, 50])]

    out = crop_predictions(FakeImage(), [[100, 100, 140, 180]], predict, factor=1.5)
    assert out == [("phone", 0.8, [95, 85, 105, 105])]   # vùng cắt bắt đầu ở (90, 80); thí sinh trong ảnh cắt bị bỏ


# ---------- pose_turn ----------
def _pose(nose_x, ear_l=0.6, ear_r=0.4, ear_vis=0.9):
    lm = [SimpleNamespace(x=0.5, y=0.5, visibility=0.9) for _ in range(33)]
    lm[NOSE] = SimpleNamespace(x=nose_x, y=0.3, visibility=0.9)
    lm[LEFT_EAR] = SimpleNamespace(x=ear_l, y=0.3, visibility=ear_vis)
    lm[RIGHT_EAR] = SimpleNamespace(x=ear_r, y=0.3, visibility=0.9)
    lm[LEFT_SHOULDER] = SimpleNamespace(x=0.75, y=0.6, visibility=0.9)
    lm[RIGHT_SHOULDER] = SimpleNamespace(x=0.25, y=0.6, visibility=0.9)
    return lm


def test_turn_ratio_from_ears_and_shoulders():
    assert abs(turn_ratio(_pose(0.5))) < 1e-9                      # nhìn thẳng
    assert abs(turn_ratio(_pose(0.58)) - 0.4) < 1e-9               # mũi lệch 0.08 / tai cách 0.2
    r = turn_ratio(_pose(0.6, ear_vis=0.1))                       # một tai bị che: dùng vai
    assert abs(r - 0.1 / (0.5 / 2.5)) < 1e-9


def test_pose_rule_needs_duration_and_cooldown():
    det = PoseTurnDetector(landmarker=object(), yaw_ratio=0.35, min_duration_s=3.0, cooldown_s=30.0)
    assert det.update(0.0, 7, [0, 0, 1, 1], 0.5) is None
    assert det.update(2.0, 7, [0, 0, 1, 1], 0.5) is None
    e = det.update(3.0, 7, [0, 0, 1, 1], 0.5)
    assert e and e["track_id"] == 7 and e["duration_s"] == 3.0
    assert det.update(10.0, 7, [0, 0, 1, 1], 0.5) is None           # đang trong 30 giây chờ
    assert det.update(11.0, 7, [0, 0, 1, 1], 0.1) is None           # quay lại nhìn thẳng thì đếm lại từ đầu


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK ", name)
