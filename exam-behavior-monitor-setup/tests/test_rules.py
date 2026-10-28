"""Kiểm thử luật vi phạm bằng dữ liệu giả, không cần GPU:  python tests/test_rules.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai_service.rules import Det, ViolationRules  # noqa: E402

STUDENT = Det(1, "student", 0.9, (100, 100, 300, 400))


def phone(conf=0.8, track_id=10):
    return Det(track_id, "phone", conf, (180, 250, 220, 300))


def run(seq, fps=10, **kw):
    rules = ViolationRules(**kw)
    out = []
    for i, dets in enumerate(seq):
        t = i / fps
        out += [(t, a) for a in rules.update(t, dets, payload=i)]
    return out


def test_phone_held_3s_alerts_once():
    alerts = run([[STUDENT, phone()]] * 30)
    assert len(alerts) == 1
    t, a = alerts[0]
    assert 2.0 <= t <= 2.2 and a.student_track_id == 1 and a.duration_s >= 2.0


def test_short_glimpse_no_alert():
    alerts = run([[STUDENT, phone()]] * 10 + [[STUDENT]] * 30)
    assert alerts == []


def test_low_confidence_ignored():
    assert run([[STUDENT, phone(conf=0.3)]] * 40) == []


def test_flicker_and_track_change_still_counts():
    seq = [[STUDENT, phone(track_id=10 + i)] if i % 4 else [STUDENT] for i in range(40)]
    assert len(run(seq)) == 1


def test_sparse_detections_below_ratio_no_alert():
    seq = [[STUDENT, phone()] if i % 2 == 0 else [STUDENT] for i in range(60)]
    assert run(seq) == []


def test_cooldown_then_alert_again():
    alerts = run([[STUDENT, phone()]] * 400)
    assert len(alerts) == 2 and abs((alerts[1][0] - alerts[0][0]) - 30) < 0.2


def test_best_frame_is_highest_confidence():
    seq = [[STUDENT, phone(conf=0.6 + 0.01 * (i % 7))] for i in range(30)]
    (_, a), = run(seq)
    assert abs(a.conf - 0.66) < 1e-6 and abs(seq[a.payload][1].conf - 0.66) < 1e-6


def test_calculator_ignored_unless_enabled():
    calc = Det(20, "calculator", 0.9, (180, 250, 220, 300))
    assert run([[STUDENT, calc]] * 30) == []
    assert len(run([[STUDENT, calc]] * 30, alert_classes=("phone", "calculator"))) == 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK ", name)
