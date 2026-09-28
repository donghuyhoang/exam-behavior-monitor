import argparse
import torch
import cv2
import numpy as np
import torchvision.transforms as T
from PIL import Image
from main import get_args_parser
from models import build_model
from google.colab.patches import cv2_imshow

# ==========================================
# 1. CẤU HÌNH VÀ TẢI MÔ HÌNH (RAW CODE)
# ==========================================
parser = argparse.ArgumentParser('Deformable DETR', parents=[get_args_parser()])
args = parser.parse_args([])
args.dataset_file = 'coco'

# Xây dựng mô hình (hệ thống sẽ tự nhận num_classes = 4 mà bạn đã sửa trong deformable_detr.py)
model, criterion, postprocessors = build_model(args)

# Nạp bộ trọng số bạn vừa train xong (đang lưu trên Drive)
weight_path = '/content/drive/MyDrive/PBL4_Output/checkpoint.pth'
checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint['model'], strict=False)

model.eval() # Chuyển sang chế độ test
model.cuda() # Chạy trên GPU T4 của Colab

# ==========================================
# 2. ĐỌC VÀ TIỀN XỬ LÝ ẢNH THỬ NGHIỆM
# ==========================================
image_path = '/content/exam-behavior-monitor/Datasets/test/IMG_0323_JPG_jpg.rf.d3060d28898627ee371c4182ea0aec3f.jpg' # ĐỔI TÊN FILE NÀY THÀNH ẢNH CỦA BẠN TRÊN COLAB
img_pil = Image.open(image_path).convert('RGB')
w, h = img_pil.size

# Transform ảnh theo đúng chuẩn ImageNet
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
img_tensor = transform(img_pil).unsqueeze(0).cuda()

# ==========================================
# 3. CHẠY SUY LUẬN (INFERENCE)
# ==========================================
with torch.no_grad():
    outputs = model(img_tensor)

target_sizes = torch.tensor([[h, w]]).cuda()
results = postprocessors['bbox'](outputs, target_sizes)[0]

# ==========================================
# 4. LỌC ID VÀ VẼ KẾT QUẢ LÊN ẢNH
# ==========================================
img_cv2 = cv2.imread(image_path)

# Cập nhật ID chuẩn từ Roboflow (bỏ qua 0 vì là Supercategory rỗng)
id2label = {
    1: "calculator",
    2: "phone",
    3: "student"
}

# Chọn màu vẽ cho từng class để dễ nhìn
colors = {
    1: (255, 255, 0),  # calculator: Xanh ngọc / Cyan
    2: (0, 255, 255),  # phone: Vàng / Yellow
    3: (0, 0, 255)     # student: Đỏ / Red
}

for score, label, box in zip(results['scores'], results['labels'], results['boxes']):
    confidence = round(score.item(), 2)
    class_id = label.item()
    
    # Chỉ vẽ nếu AI tự tin > 40% VÀ class_id nằm trong danh sách 1, 2, 3
    if confidence > 0.40 and class_id in id2label:
        class_name = id2label[class_id]
        color = colors.get(class_id, (0, 255, 0)) # Mặc định màu xanh lá
        
        box = [int(i) for i in box.tolist()]
        x_min, y_min, x_max, y_max = box
        
        # Vẽ Bounding Box
        cv2.rectangle(img_cv2, (x_min, y_min), (x_max, y_max), color, 2)
        
        # Vẽ nền cho text để chữ không bị chìm vào ảnh
        text = f"{class_name}: {confidence}"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img_cv2, (x_min, y_min - text_h - 10), (x_min + text_w, y_min), color, -1)
        
        # Ghi chữ màu đen lên nền màu
        cv2.putText(img_cv2, text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

# Hiển thị ảnh ngay trong Colab
cv2_imshow(img_cv2)
