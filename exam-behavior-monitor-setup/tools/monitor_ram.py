"""Ghi RAM và CPU của các tiến trình hệ thống cho bài nghiệm thu "Chạy dài" (30 phút, RAM không tăng mãi).

Mở thêm một cửa sổ PowerShell sau khi đã bật hệ thống (run_demo.ps1), chạy:
    python tools/monitor_ram.py --minutes 30 --every 10 --out runs/ram_nghiemthu2.csv

Mỗi 10 giây ghi một dòng cho từng tiến trình: dịch vụ AI (ai_service.main), backend (uvicorn), máy chủ MQTT
(mqtt_broker hoặc mosquitto). Tắt backend rồi bật lại giữa chừng vẫn được: tiến trình được tìm lại theo tên.
Hết giờ hoặc bấm Ctrl+C thì in bảng: RAM trung bình 5 phút đầu (bỏ 2 phút khởi động), 5 phút cuối, cao nhất,
và kết luận. Chép bảng này vào dòng "Chạy dài" của file kết quả nghiệm thu.
"""
import argparse
import csv
import os
import statistics
import sys
import time
from datetime import datetime

PROCS = {"ai": "ai_service.main", "backend": "uvicorn", "mqtt": "mqtt_broker", "mosquitto": "mosquitto"}


def label_of(name, cmdline):
    text = " ".join([name or ""] + (cmdline or [])).lower()
    for label, key in PROCS.items():
        if key in text:
            return label
    return None


def find_procs():
    import psutil

    found = {}
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            label = label_of(p.info["name"], p.info["cmdline"])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if label and label not in found:
            found[label] = p
    return found


def summarize(rows, warmup_s=120, window_s=300, max_growth_mb=100, max_growth_pct=10):
    """rows: [(giây từ lúc bắt đầu, nhãn, rss_mb)]. Trả về {nhãn: thống kê và kết luận}."""
    result = {}
    for label in sorted({r[1] for r in rows}):
        pts = [(t, mb) for t, lb, mb in rows if lb == label]
        end = pts[-1][0]
        first = [mb for t, mb in pts if warmup_s <= t < warmup_s + window_s] or [pts[0][1]]
        last = [mb for t, mb in pts if t > end - window_s] or [pts[-1][1]]
        a, b = statistics.mean(first), statistics.mean(last)
        growth = b - a
        pct = 100 * growth / a if a else 0.0
        ok = growth <= max_growth_mb or pct <= max_growth_pct
        result[label] = {"dau_mb": round(a), "cuoi_mb": round(b), "cao_nhat_mb": round(max(mb for _, mb in pts)),
                         "tang_mb": round(growth), "tang_pct": round(pct, 1), "phut": round(end / 60, 1),
                         "ket_luan": "ổn định" if ok else "RAM tăng liên tục, cần xem lại"}
    return result


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Theo dõi RAM/CPU các tiến trình hệ thống")
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--every", type=float, default=10, help="giây giữa hai lần ghi")
    ap.add_argument("--out", default="runs/ram.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rows, t0, cache = [], time.time(), {}
    procs = find_procs()
    if not procs:
        sys.exit("Chưa thấy tiến trình nào của hệ thống: bật run_demo.ps1 trước rồi chạy lại")
    print("Đang theo dõi: " + ", ".join(f"{k} (pid {p.pid})" for k, p in procs.items()))
    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gio", "giay", "tien_trinh", "pid", "ram_mb", "cpu_pct"])
        try:
            while time.time() - t0 < args.minutes * 60:
                # giữ lại đối tượng Process theo pid: cpu_percent chỉ đúng từ lần đo thứ hai của cùng một đối tượng
                procs = {label: cache.setdefault(p.pid, p) for label, p in find_procs().items()}
                elapsed = time.time() - t0
                line = []
                for label, p in procs.items():
                    try:
                        mb = p.memory_info().rss / 2 ** 20
                        cpu = p.cpu_percent(interval=None)
                    except Exception:
                        continue
                    rows.append((elapsed, label, mb))
                    w.writerow([datetime.now().strftime("%H:%M:%S"), round(elapsed), label, p.pid, round(mb, 1), cpu])
                    line.append(f"{label} {mb:.0f} MB")
                f.flush()
                print(f"{elapsed / 60:5.1f} phút  " + " · ".join(line), flush=True)
                time.sleep(args.every)
        except KeyboardInterrupt:
            pass
    if not rows:
        return
    print(f"\n{'Tiến trình':<10} {'đầu':>7} {'cuối':>7} {'cao nhất':>9} {'tăng':>12}  Kết luận")
    for label, s in summarize(rows).items():
        print(f"{label:<10} {s['dau_mb']:>5} MB {s['cuoi_mb']:>5} MB {s['cao_nhat_mb']:>6} MB "
              f"{s['tang_mb']:>+5} MB {s['tang_pct']:>+5}%  {s['ket_luan']} ({s['phut']} phút)")
    print(f"Chi tiết: {args.out}")


if __name__ == "__main__":
    main()
