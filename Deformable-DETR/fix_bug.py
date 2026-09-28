import os

print("=== BẮT ĐẦU QUÁ TRÌNH TỰ ĐỘNG VÁ LỖI DEFORMABLE DETR ===")

# Xác định đường dẫn tương đối dựa trên vị trí của file fix_bug.py hiện tại
current_dir = os.path.dirname(os.path.abspath(__file__))

# 1. VÁ LỖI FILE util/misc.py (Lỗi phiên bản torchvision)
misc_path = os.path.join(current_dir, "util", "misc.py")
if os.path.exists(misc_path):
    with open(misc_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    content = content.replace("float(torchvision.__version__[:3]) < 0.5", "False")
    content = content.replace("float(torchvision.__version__[:3]) < 0.7", "False")
    content = content.replace("torchvision.ops.misc.interpolate", "torch.nn.functional.interpolate")
    
    with open(misc_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("[OK] Đã vá xong file: util/misc.py")
else:
    print("[BỎ QUA] Không tìm thấy file: util/misc.py")


# 2. VÁ LỖI FILE main.py (Lỗi weights_only khi load checkpoint PyTorch 2.6+)
main_path = os.path.join(current_dir, "main.py")
if os.path.exists(main_path):
    with open(main_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Thay thế dòng load checkpoint cũ bằng cú pháp có weights_only=False
    old_load = "torch.load(args.resume, map_location='cpu')"
    new_load = "torch.load(args.resume, map_location='cpu', weights_only=False)"
    
    if old_load in content:
        content = content.replace(old_load, new_load)
        with open(main_path, "w", encoding="utf-8") as f:
            f.write(content)
        print("[OK] Đã vá xong file: main.py")
    else:
        print("[CẢNH BÁO] Không tìm thấy đoạn code torch.load tương ứng trong main.py (có thể đã được fix trước đó).")
else:
    print("[BỎ QUA] Không tìm thấy file: main.py")


# 3. VÁ LỖI MÃ NGUỒN C++/CUDA (Lỗi value.type() trong ms_deform_attn_cuda.cu)
cuda_path = os.path.join(current_dir, "models", "ops", "src", "cuda", "ms_deform_attn_cuda.cu")
if os.path.exists(cuda_path):
    with open(cuda_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Đổi hàm type() cũ thành scalar_type() tương thích với PyTorch mới
    content = content.replace("value.type()", "value.scalar_type()")
    
    with open(cuda_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("[OK] Đã vá xong file C++: ms_deform_attn_cuda.cu")
else:
    print("[BỎ QUA] Không tìm thấy file C++ CUDA. (Nếu bạn chưa copy thư mục models/ops vào, hãy kiểm tra lại).")

# 4. VÁ LỖI CẤU TRÚC THƯ MỤC DATASET TRONG datasets/coco.py
coco_dataset_path = os.path.join(current_dir, "datasets", "coco.py")
if os.path.exists(coco_dataset_path):
    with open(coco_dataset_path, "r", encoding="utf-8") as f:
        coco_content = f.read()
    
    # Tìm đoạn định nghĩa PATHS gốc và thay thế bằng thư mục train/val của bạn
    old_paths = '''    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }'''
    
    new_paths = '''    # Đã cấu hình lại để khớp với thư mục train/val do Roboflow xuất ra
    PATHS = {
        "train": (root / "train", root / "train" / "_annotations.coco.json"),
        "val": (root / "valid", root / "valid" / "_annotations.coco.json"),
    }'''
    
    if old_paths in coco_content:
        coco_content = coco_content.replace(old_paths, new_paths)
        with open(coco_dataset_path, "w", encoding="utf-8") as f:
            f.write(coco_content)
        print("[OK] Đã cấu hình lại đường dẫn dataset trong: datasets/coco.py")
    else:
        print("[LƯU Ý] Đoạn PATHS trong datasets/coco.py có thể đã được thay đổi từ trước.")
else:
    print("[BỎ QUA] Không tìm thấy file: datasets/coco.py")
    
print("=== HOÀN TẤT! TẤT CẢ CÁC FILE ĐÃ ĐƯỢC VÁ LỖI THÀNH CÔNG ===")