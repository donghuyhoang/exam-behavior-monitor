# PBL4: hệ thống camera giám sát phòng thi

Camera (RTSP) → `ai_service` (RF-DETR + theo dõi + luật vi phạm) → MQTT → `backend` (FastAPI, SQLite, ảnh bằng chứng) → giao diện giám thị.

## Cấu trúc

| Thư mục | Ai | Nội dung |
|---|---|---|
| `ai_service/` | B | `main.py` vòng chạy chính, `stream.py` đọc camera/video, `detector.py` RF-DETR, `tracker.py` gán track_id, `rules.py` luật vi phạm, `publisher.py` MQTT, `mjpeg.py` phát video, `bench_fps.py` đo tốc độ |
| `backend/` | C | `app.py` API + MQTT + SQLite + WebSocket, `fake_events.py` sự kiện giả |
| `ui/` | C | `index.html` giao diện tham khảo, backend phục vụ ở `/ui/` |
| `behavior/` | NT (P1) | `head_pose.py` MediaPipe phát hiện quay đầu |
| `tools/` | NT, A | `probe_camera.py` dò camera, `export_preds_rfdetr.py` xuất dự đoán để chấm điểm, `mosquitto.conf` |
| `tests/` | mọi người | kiểm thử luật và bộ theo dõi, không cần GPU |

`weights/` (checkpoint), `data/` (cơ sở dữ liệu, ảnh bằng chứng), `.env` (mật khẩu camera) không commit.

## Cài trên laptop máy chủ (Windows)

```
py install 3.12                      # nếu chưa có Python 3.12
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-ai.txt -r requirements-backend.txt
copy .env.example .env               # rồi sửa RTSP_URL
```

Chép checkpoint RF-DETR vào `weights/checkpoint_best_total.pth`.

## Chạy (mỗi lệnh một cửa sổ, từ thư mục gốc)

```
python tools\mqtt_broker.py          # hoặc Mosquitto: "C:\Program Files\mosquitto\mosquitto.exe" -c tools\mosquitto.conf -v
uvicorn backend.app:app --host 0.0.0.0 --port 8000
python -m ai_service.main --show
```

Hoặc bật cả ba bằng một lệnh: `powershell -ExecutionPolicy Bypass -File run_demo.ps1` (thêm `-Source videos\test1.mp4` để chạy video dự phòng).
Dịch vụ AI mất khoảng 30–50 giây để nạp và biên dịch mô hình trước khi có hình. Đo trên laptop RTX 5050: mô hình 27 FPS, cả vòng xử lý khoảng 24 FPS.

Mở http://localhost:8000/ui/. Máy khác trong cùng mạng mở `http://<ip-laptop>:8000/ui/` (cho phép cổng 1883, 8000, 8001 trong Windows Firewall, mạng Private).

## Không có camera hoặc GPU

```
python -m ai_service.main --source videos/mau_000.mp4 --dry-run --save-video runs/out.mp4   # Colab
python -m backend.fake_events --every 3                                                     # làm backend/giao diện
python tests/test_rules.py && python tests/test_tracker.py
```
