"""Bọc RF-DETR: nhận khung BGR của OpenCV, trả về supervision.Detections kèm tên lớp."""
import numpy as np
from PIL import Image


def name_from_id(names, class_id):
    """Tên lớp theo class_id của RF-DETR.

    class_id là id lớp trong file COCO của Roboflow lúc train: id 0 là lớp cha (không dùng), lớp thật đánh số từ 1
    (calculator 1, phone 2, student 3). rfdetr bản cũ trả class_names dạng {1: "calculator", ...}; bản mới trả
    danh sách ["calculator", "phone", "student"] nên phải lùi một chỉ số, tra names[class_id] sẽ lệch lớp."""
    c = int(class_id)
    if isinstance(names, dict):
        return str(names.get(c, c))
    names = list(names)
    return str(names[c - 1]) if 1 <= c <= len(names) else str(c)


def class_labels(det, names):
    """Tên lớp cho từng khung: ưu tiên tên rfdetr gắn sẵn trong det.data, không có thì tra theo class_id."""
    data = det.data or {}
    if "class_name" in data and len(data["class_name"]) == len(det):
        return [str(x) for x in data["class_name"]]
    return [name_from_id(names, c) for c in det.class_id]


class RFDETRDetector:
    def __init__(self, weights, size="medium", threshold=0.3, optimize=True, fp16=False, resolution=None):
        # Đo trên RTX 5050 laptop (28/10): tối ưu FP32 27 FPS, FP16 21 FPS, không tối ưu 14 FPS -> mặc định FP32
        import rfdetr
        import torch

        model_cls = {"nano": rfdetr.RFDETRNano, "small": rfdetr.RFDETRSmall, "medium": rfdetr.RFDETRMedium}[size]
        kwargs = {"pretrain_weights": weights}
        if resolution:  # mô hình train ở resolution khác mặc định (ví dụ 704) thì phải nạp đúng giá trị đó
            kwargs["resolution"] = int(resolution)
        self.model = model_cls(**kwargs)
        if optimize:
            try:
                if hasattr(self.model, "optimize_for_inference"):
                    self.model.optimize_for_inference()
                else:  # rfdetr 1.11 đổi tên thành inference()
                    dtype = torch.float16 if fp16 and torch.cuda.is_available() else torch.float32
                    self.model.inference(compile=True, dtype=dtype)
            except Exception as e:  # tối ưu lỗi thì vẫn chạy được, chỉ chậm hơn
                print(f"[detector] bỏ qua bước tối ưu: {e}")
        self.names = self.model.class_names
        self.threshold = threshold

    def name_of(self, class_id):
        return name_from_id(self.names, class_id)

    def __call__(self, frame_bgr):
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        return self.model.predict(Image.fromarray(rgb), threshold=self.threshold)
