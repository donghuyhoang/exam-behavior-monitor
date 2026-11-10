# Danh sách lỗi

Mỗi lỗi một dòng. Trạng thái: mới / đang sửa / đã sửa (ghi commit) / để sau.

Dòng 1–8: lỗi gặp khi ghép cả hệ thống trên laptop nhóm trưởng (28/10): camera → dịch vụ AI → MQTT → backend → giao diện, chạy với RF-DETR thật trên `test1.mp4`.

| # | Ngày | Lỗi: thấy gì, làm thế nào để lặp lại | Khối | Ai sửa | Hạn | Trạng thái |
|---|---|---|---|---|---|---|
| 1 | 28/10 | `ai_service` không chạy được từ repo: `publisher.py` dòng 25 lỗi cú pháp (chuỗi `"\n"` bị xuống dòng thật) | ai_service | NT soạn bản sửa, B áp | 29/10 | đã sửa (e655f5e) |
| 2 | 28/10 | Laptop chỉ có Python 3.14 và PyTorch bản CPU, `torch.cuda.is_available()` ra False, không chạy được mô hình trên RTX 5050 | laptop | NT | 28/10 | đã sửa: cài Python 3.12, PyTorch 2.11 bản CUDA 12.8 (môi trường laptop, không có commit) |
| 3 | 28/10 | rfdetr 1.11 bỏ hàm `optimize_for_inference()`, mô hình chạy không tối ưu chỉ 14 FPS | ai_service | NT | 28/10 | đã sửa (e655f5e): dùng `inference(compile=True)`, 27 FPS; cả vòng xử lý khoảng 24 FPS |
| 4 | 28/10 | supervision 0.31 bỏ `sv.ByteTrack`, không gán được track_id | ai_service | NT | 28/10 | đã sửa (a59f4d1): bộ theo dõi riêng `ai_service/tracker.py` |
| 5 | 28/10 | Laptop chưa cài Mosquitto, dịch vụ AI và backend không gửi nhận được tin MQTT | hạ tầng | NT | 28/10 | đã sửa (e655f5e): máy chủ MQTT bằng Python `tools/mqtt_broker.py`, bật cả hệ thống bằng `run_demo.ps1` |
| 6 | 28/10 | Mở giao diện trước khi dịch vụ AI sẵn sàng thì khung video đen mãi, luồng MJPEG không tự nối lại | ui | NT | 28/10 | đã sửa (e655f5e): giao diện tự chuyển sang tải `/snapshot.jpg` |
| 7 | 28/10 | `tools/tune_rules.py` báo lỗi khi ghi kết quả tiếng Việt ra file hoặc ống dẫn trên Windows (bảng mã cp1252) | tools | NT | 28/10 | đã sửa (e655f5e) |
| 8 | 28/10 | Trên camera 640×360, điện thoại cầm tay bị nhận là `calculator`, nên chế độ chỉ báo `phone` không báo lần nào; trên 97 ảnh camera chỉ bắt được 1/304 điện thoại | mô hình | B, A | 6/11 | đang sửa: tạm báo cả `calculator` (`alert_classes`), train lại với ảnh camera tuần 2/11 |
| 9 | 29/10 | File `sua_repo_exam-behavior-monitor.patch` bị commit nhầm vào thư mục gốc repo cùng bản sửa | repo | NT | 3/11 | đã sửa: xoá cùng commit thêm file này |
| 10 | 10/11 | `nghiemthu1.mp4` là video dịch vụ AI ghi ra (`--save-video`, 25 khung/giây cố định) trong khi máy chỉ xử lý 11–15 khung/giây, nên video chạy nhanh 1,6–2,5 lần (trung bình 1,91). `scenes_nghiemthu1.csv` ghi theo giờ thật nên lệch giờ video; ảnh test cắt từ video này còn khung AI vẽ sẵn | nghiệm thu | NT | 10/11 | đã sửa đáp án: đổi sang giờ video theo giờ camera in trên hình (`docs/khung/nghiemthu1_bang_gio_camera.csv`). Từ lần 2 ghi thêm video gốc bằng ffmpeg; ảnh test phải cắt lại từ video gốc |
| 11 | 10/11 | Sau `5693259`, `RFDETRDetector.name_of` coi id lớp đánh số từ 1. rfdetr 1.11.2 trên laptop đánh số từ 0 (calculator 0, phone 1, student 2), nên `name_of` lệch một lớp: student thành phone, phone thành calculator. Dịch vụ AI không bị vì ưu tiên tên lớp rfdetr gắn sẵn (`det.data["class_name"]`), nhưng code nào gọi thẳng `name_of` sẽ sai nhãn | ai_service | A | | mới |
| 12 | | | | | | |
