"""Bọc RF-DETR: nhận khung BGR của OpenCV, trả về supervision.Detections kèm tên lớp."""
import numpy as np
from PIL import Image


class RFDETRDetector:
    def __init__(self, weights, size="medium", threshold=0.3, optimize=True):
        import rfdetr

        model_cls = {"nano": rfdetr.RFDETRNano, "small": rfdetr.RFDETRSmall, "medium": rfdetr.RFDETRMedium}[size]
        self.model = model_cls(pretrain_weights=weights)
        if optimize:
            try:
                self.model.optimize_for_inference()
            except Exception as e:
                print(f"[detector] bỏ qua optimize_for_inference: {e}")
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
