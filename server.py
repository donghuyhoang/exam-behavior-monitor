import sys, types

# ================== MOCK MODULE C++ GIẢ LẬP ==================
fake_msda = types.ModuleType("MultiScaleDeformableAttention")
sys.modules["MultiScaleDeformableAttention"] = fake_msda

import argparse, glob, os, threading, time, traceback
import cv2, numpy as np, torch
import torchvision.transforms as T
from PIL import Image
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

# Import hàm PyTorch thuần
from models.ops.functions.ms_deform_attn_func import ms_deform_attn_core_pytorch
import models.ops.modules.ms_deform_attn as ms_deform_module

# Override đè hàm C++ bằng hàm PyTorch thuần
ms_deform_module.MSDeformAttnFunction = type("Dummy", (), {
    "apply": staticmethod(
        lambda v, sp, idx, loc, w, step=None: ms_deform_attn_core_pytorch(v, sp, loc, w)
    )
})

from main import get_args_parser
from models import build_model

# ================== CẤU HÌNH ==================
WEIGHT_PATH = r"F:\PBL4\Deformable-DETR\checkpoint.pth" # đường dẫn tới weight bạn vừa train xong
IMAGE_DIR = r"F:\PBL4\Datasets\test" # đường dẫn tới folder chứa ảnh test
SECONDS_PER_IMAGE = 2.0
FRAME_W, FRAME_H = 960, 540
THRESHOLD = 0.1
SHORT_SIDE = 640
PHONE_ID = 2
id2label = {1: "calculator", 2: "phone", 3: "student"}
colors = {1: (255, 255, 0), 2: (0, 255, 255), 3: (0, 0, 255)}  # BGR
device = "cuda" if torch.cuda.is_available() else "cpu"

# Các tham số này PHẢI giống lúc train, nếu không weight sẽ không khớp
WITH_BOX_REFINE = False   # đặt True nếu lúc train có --with_box_refine
TWO_STAGE = False         # đặt True nếu lúc train có --two_stage

# ================== LOAD MODEL ==================
parser = argparse.ArgumentParser("Deformable DETR", parents=[get_args_parser()])
args = parser.parse_args([])
args.dataset_file = "coco"
args.device = device
args.with_box_refine = WITH_BOX_REFINE
args.two_stage = TWO_STAGE

model, criterion, postprocessors = build_model(args)
ckpt = torch.load(WEIGHT_PATH, map_location="cpu", weights_only=False)
msg = model.load_state_dict(ckpt["model"], strict=False)
print("=== LOAD WEIGHT ===")
print("missing keys   :", len(msg.missing_keys), msg.missing_keys[:10])
print("unexpected keys:", len(msg.unexpected_keys), msg.unexpected_keys[:10])
print("(nếu 2 số này lớn => sai cấu hình args so với lúc train)")
model.to(device).eval()

transform = T.Compose([
    T.Resize(SHORT_SIDE),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ================== NGUỒN ẢNH ==================
image_paths = sorted(
    glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
    + glob.glob(os.path.join(IMAGE_DIR, "*.jpeg"))
    + glob.glob(os.path.join(IMAGE_DIR, "*.png"))
)
assert image_paths, f"Folder không có ảnh: {IMAGE_DIR}"

def letterbox(img, w, h):
    ih, iw = img.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[(h - nh) // 2:(h - nh) // 2 + nh, (w - nw) // 2:(w - nw) // 2 + nw] = resized
    return canvas

# ================== TRẠNG THÁI DÙNG CHUNG ==================
app = FastAPI()
lock = threading.Lock()
latest_frame = None
latest_id = -1
annotated = None        # frame đã vẽ khung, do infer_loop tạo ra
detections = []
det_id = -1
state = {"phone": False, "conf": 0.0}

def draw(frame, dets):
    for cid, s, x1, y1, x2, y2 in dets:
        color = colors.get(cid, (0, 255, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{id2label.get(cid, 'obj')}: {s:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
        cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

def camera_loop():
    global latest_frame, latest_id
    i = 0
    while True:
        frame = cv2.imread(image_paths[i % len(image_paths)])
        if frame is not None:
            frame = letterbox(frame, FRAME_W, FRAME_H)
            with lock:
                latest_frame, latest_id = frame, i
        i += 1
        time.sleep(SECONDS_PER_IMAGE)

@torch.inference_mode()
def infer_loop():
    global detections, det_id, state, annotated
    last_done = -1
    while True:
        try:
            with lock:
                if latest_frame is None or latest_id == last_done:
                    frame, fid = None, None
                else:
                    frame, fid = latest_frame.copy(), latest_id

            if frame is None:
                time.sleep(0.03)
                continue

            t0 = time.time()
            h, w = frame.shape[:2]
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            x = transform(img).unsqueeze(0).to(device)

            outputs = model(x)
            target_sizes = torch.tensor([[h, w]], device=device)
            res = postprocessors["bbox"](outputs, target_sizes)[0]

            found = []
            for score, label, box in zip(res["scores"], res["labels"], res["boxes"]):
                s, cid = float(score), int(label)
                if s > THRESHOLD and cid in id2label:
                    found.append((cid, s, *map(int, box.tolist())))

            phones = [d[1] for d in found if d[0] == PHONE_ID]

            # Vẽ ngay trên frame đã dùng để infer => luôn khớp ảnh
            out = frame.copy()
            draw(out, found)

            with lock:
                annotated = out
                detections = found
                det_id = fid
                state["phone"] = len(phones) > 0
                state["conf"] = max(phones, default=0.0)

            last_done = fid
            print(f"[infer] frame={fid} time={time.time() - t0:.2f}s "
                  f"boxes={len(found)} max_score={float(res['scores'].max()):.3f}")

        except Exception:
            traceback.print_exc()
            time.sleep(1)

def gen():
    while True:
        with lock:
            src = annotated if annotated is not None else latest_frame
            frame = None if src is None else src.copy()

        if frame is None:
            time.sleep(0.03)
            continue

        _, jpg = cv2.imencode(".jpg", frame)
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
        time.sleep(0.03)

@app.get("/video")
def video():
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/status")
def status():
    with lock:
        return dict(state)

threading.Thread(target=camera_loop, daemon=True).start()
threading.Thread(target=infer_loop, daemon=True).start()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)