"""Kiểm thử bộ theo dõi IoU:  python tests/test_tracker.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai_service.tracker import IouTracker  # noqa: E402

STUDENT, PHONE = 1, 2


def test_same_object_keeps_id_while_moving_slowly():
    tr = IouTracker()
    ids = [tr.update([[100 + 2 * i, 100, 300 + 2 * i, 400]], [STUDENT])[0] for i in range(20)]
    assert len(set(ids)) == 1


def test_two_students_get_different_ids_and_keep_them():
    tr = IouTracker()
    a = tr.update([[0, 0, 100, 200], [300, 0, 400, 200]], [STUDENT, STUDENT])
    b = tr.update([[302, 0, 402, 200], [1, 0, 101, 200]], [STUDENT, STUDENT])
    assert a[0] != a[1] and b[1] == a[0] and b[0] == a[1]


def test_different_class_never_matched():
    tr = IouTracker()
    a = tr.update([[0, 0, 100, 100]], [STUDENT])
    b = tr.update([[0, 0, 100, 100]], [PHONE])
    assert a[0] != b[0]


def test_short_gap_keeps_id_long_gap_new_id():
    tr = IouTracker(max_lost=5)
    first = tr.update([[0, 0, 100, 100]], [STUDENT])[0]
    for _ in range(3):
        tr.update([], [])
    assert tr.update([[0, 0, 100, 100]], [STUDENT])[0] == first
    for _ in range(10):
        tr.update([], [])
    assert tr.update([[0, 0, 100, 100]], [STUDENT])[0] != first


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK ", name)
