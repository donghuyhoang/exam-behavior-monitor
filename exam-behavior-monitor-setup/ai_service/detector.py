"""Bọc RF-DETR: nhận khung BGR của OpenCV, trả về supervision.Detections kèm tên lớp."""
import numpy as np
from PIL import Image


class RFDETRDetector:
    def __init__(self, weights, size="medium", threshold=0.3, optimize=True, fp16=False):
        # Đo trên RTX 5050 laptop (28/10): tối ưu FP32 27 FPS, FP16 21 FPS, không tối ưu 14 FPS -> mặc định FP32
        import rfdetr
        import torch

        model_cls = {"nano": rfdetr.RFDETRNano, "small": rfdetr.RFDETRSmall, "medium": rfdetr.RFDETRMedium}[size]
        self.model = model_cls(pretrain_weights=weights)
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
        c = int(class_id)
        if isinstance(self.names, dict):
            return str(self.names.get(c, c))
        return str(self.names[c]) if 0 <= c < len(self.names) else str(c)

    def __call__(self, frame_bgr):
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        return self.model.predict(Image.fromarray(rgb), threshold=self.threshold)
