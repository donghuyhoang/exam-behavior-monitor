"""Thử nhanh (P1, không cần train): tìm điện thoại trên ảnh cắt quanh từng thí sinh thay vì trên cả khung hình.

Mô hình nhận ảnh ở kích thước cố định (Medium: 576), nên cắt vùng quanh một thí sinh rồi đưa vào mô hình
tức là phóng to vùng đó lên nhiều lần: điện thoại 12 pixel có thể thành 50-80 pixel.

Script ghi hai file dự đoán cùng định dạng export_preds_rfdetr.py để chấm chung một lần:
  pred_<tag>_ca_khung.json : chạy trên cả khung hình như bình thường
  pred_<tag>_cat_anh.json  : thí sinh lấy từ cả khung; điện thoại, máy tính lấy từ ảnh cắt quanh từng thí sinh

    python tools/eval_crops.py --images test_camera --weights weights/checkpoint_best_total.pth \
        --out predictions_crops/ --tag rfdetr_m
    python benchmark/evaluate_detections.py --gt test_camera/_annotations.coco.json \
        --pred "predictions_crops/pred_*.json" --out predictions_crops/

--students gt dùng khung thí sinh trong nhãn (giới hạn trên, để biết cách này tốt nhất được bao nhiêu);
mặc định dùng thí sinh do mô hình tìm được (giống lúc chạy thật).
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai_service.detector import class_labels  # noqa: E402

TARGETS = ("phone", "calculator")
PERSON = "student"


def expand_box(box, factor, width, height):
    """Nới khung [x1,y1,x2,y2] quanh tâm theo hệ số, cắt theo biên ảnh, trả về số nguyên."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw, hh = (x2 - x1) * factor / 2, (y2 - y1) * factor / 2
    return (max(0, int(cx - hw)), max(0, int(cy - hh)), min(width, int(round(cx + hw))), min(height, int(round(cy + hh))))


def map_back(box, offset):
    """Khung trong ảnh cắt -> khung trong ảnh gốc."""
    ox, oy = offset
    x1, y1, x2, y2 = box
    return [x1 + ox, y1 + oy, x2 + ox, y2 + oy]


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def nms(dets, thr=0.5):
    """dets: [(class_name, score, [x1,y1,x2,y2])]. Bỏ khung trùng cùng lớp (do hai vùng cắt chồng lên nhau)."""
    keep = []
    for d in sorted(dets, key=lambda d: -d[1]):
        if all(k[0] != d[0] or iou(k[2], d[2]) < thr for k in keep):
            keep.append(d)
    return keep


def to_pred_rows(file_name, dets, limit=100):
    rows = []
    for name, score, (x1, y1, x2, y2) in sorted(dets, key=lambda d: -d[1])[:limit]:
        rows.append({"file_name": file_name, "class_name": name, "score": float(score),
                     "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]})
    return rows


def crop_predictions(image, students, predict_fn, factor=1.5, nms_thr=0.5):
    """image: PIL.Image; students: [[x1,y1,x2,y2]]; predict_fn(PIL) -> [(class_name, score, xyxy)]."""
    W, H = image.size
    found = []
    for box in students:
        x1, y1, x2, y2 = expand_box(box, factor, W, H)
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        for name, score, b in predict_fn(image.crop((x1, y1, x2, y2))):
            if name in TARGETS:
                found.append((name, score, map_back(b, (x1, y1))))
    return nms(found, nms_thr)


def rfdetr_predict_fn(model, threshold=0.001):
    names = model.class_names

    def predict(img):
        det = model.predict(img, threshold=threshold)
        labels = class_labels(det, names)
        return [(str(labels[k]), float(det.confidence[k]), det.xyxy[k].tolist()) for k in range(len(det))]

    return predict


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="So sánh tìm điện thoại trên cả khung và trên ảnh cắt quanh thí sinh")
    ap.add_argument("--images", required=True, help="thư mục ảnh có _annotations.coco.json")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--size", default="medium", choices=["nano", "small", "medium"])
    ap.add_argument("--resolution", type=int)
    ap.add_argument("--students", choices=["pred", "gt"], default="pred")
    ap.add_argument("--student-conf", type=float, default=0.5, help="ngưỡng nhận thí sinh khi --students pred")
    ap.add_argument("--expand", type=float, default=1.5, help="hệ số nới khung thí sinh trước khi cắt")
    ap.add_argument("--tag", default="rfdetr_m")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import rfdetr
    from PIL import Image

    kwargs = {"pretrain_weights": args.weights}
    if args.resolution:
        kwargs["resolution"] = args.resolution
    model = {"nano": rfdetr.RFDETRNano, "small": rfdetr.RFDETRSmall, "medium": rfdetr.RFDETRMedium}[args.size](**kwargs)
    predict = rfdetr_predict_fn(model)

    gt_students = {}
    ann_path = os.path.join(args.images, "_annotations.coco.json")
    if args.students == "gt":
        with open(ann_path, encoding="utf-8") as f:
            coco = json.load(f)
        person_ids = {c["id"] for c in coco["categories"] if c["name"] == PERSON}
        names = {im["id"]: im["file_name"] for im in coco["images"]}
        for a in coco["annotations"]:
            if a["category_id"] in person_ids:
                x, y, w, h = a["bbox"]
                gt_students.setdefault(names[a["image_id"]], []).append([x, y, x + w, y + h])

    paths = sorted(p for p in glob.glob(os.path.join(args.images, "*")) if p.lower().endswith((".jpg", ".jpeg", ".png")))
    full_rows, crop_rows = [], []
    for i, p in enumerate(paths, 1):
        fname = os.path.basename(p)
        img = Image.open(p).convert("RGB")
        full = predict(img)
        full_rows += to_pred_rows(fname, full)
        if args.students == "gt":
            students = gt_students.get(fname, [])
        else:
            students = [b for n, s, b in full if n == PERSON and s >= args.student_conf]
        persons = [d for d in full if d[0] == PERSON]
        crop_rows += to_pred_rows(fname, persons + crop_predictions(img, students, predict, args.expand))
        if i % 20 == 0:
            print(f"{i}/{len(paths)} ảnh")

    os.makedirs(args.out, exist_ok=True)
    for suffix, rows in (("ca_khung", full_rows), ("cat_anh", crop_rows)):
        path = os.path.join(args.out, f"pred_{args.tag}_{suffix}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f)
        print(f"Đã ghi {len(rows)} khung vào {path}")
    print("Chấm: python benchmark/evaluate_detections.py --gt", ann_path,
          f'--pred "{os.path.join(args.out, "pred_*.json")}" --out {args.out}')


if __name__ == "__main__":
    main()
