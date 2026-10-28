"""Xuất file dự đoán RF-DETR cho một thư mục ảnh COCO (ví dụ tập test camera), đúng định dạng evaluate_detections.py.

    python tools/export_preds_rfdetr.py --images test_camera --weights weights/checkpoint_best_total.pth \
        --out predictions_camera/pred_rfdetr_m.json
    python benchmark/evaluate_detections.py --gt test_camera/_annotations.coco.json \
        --pred "predictions_camera/pred_*.json" --out predictions_camera/
"""
import argparse
import glob
import json
import os

from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="thư mục ảnh (thường có _annotations.coco.json)")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--size", default="medium", choices=["nano", "small", "medium"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import rfdetr

    model = {"nano": rfdetr.RFDETRNano, "small": rfdetr.RFDETRSmall, "medium": rfdetr.RFDETRMedium}[args.size](
        pretrain_weights=args.weights)
    names = model.class_names
    paths = sorted(p for p in glob.glob(os.path.join(args.images, "*")) if p.lower().endswith((".jpg", ".jpeg", ".png")))
    preds = []
    for i, p in enumerate(paths, 1):
        det = model.predict(Image.open(p).convert("RGB"), threshold=0.001)   # ngưỡng thấp để tính mAP
        labels = det.data.get("class_name", [names[c] for c in det.class_id])
        order = det.confidence.argsort()[::-1][:100]                           # tối đa 100 khung mỗi ảnh
        for k in order:
            x1, y1, x2, y2 = det.xyxy[k].tolist()
            preds.append({"file_name": os.path.basename(p), "class_name": str(labels[k]),
                          "score": float(det.confidence[k]), "bbox": [x1, y1, x2 - x1, y2 - y1]})
        if i % 50 == 0:
            print(f"{i}/{len(paths)} ảnh")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(preds, f)
    print(f"Đã ghi {len(preds)} khung của {len(paths)} ảnh vào {args.out}")


if __name__ == "__main__":
    main()
