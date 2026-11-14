# Raspberry Pi làm máy chủ phòng thi

Laptop chỉ còn chạy AI (cần GPU). Raspberry Pi giữ phần **lưu và báo**: Mosquitto, backend (SQLite, ảnh bằng chứng, giao diện web) và trạm báo có 3 đèn, 2 nút cho giám thị (không dùng còi).

```
Camera ──RTSP──> Laptop: ai_service (RF-DETR + MediaPipe) ──MQTT: sự kiện + ảnh──> Raspberry Pi
                     │ video :8001                                                ├─ Mosquitto :1883
                     └──────────────> điện thoại giám thị <── web :8000 ───────────├─ backend (SQLite, ảnh)
                                                                                    └─ trạm báo: 3 đèn, 2 nút
```

Laptop tắt hay treo thì Pi báo ngay, lịch sử và ảnh vẫn còn trên Pi.

## Cần chuẩn bị

| Món | Ghi chú |
|---|---|
| Raspberry Pi 4 (2GB là đủ), thẻ microSD 32GB | backend dùng khoảng 70 MB RAM |
| Nguồn USB-C 5V 3A | củ sạc điện thoại yếu làm Pi thiếu điện, có thể treo |
| LED đỏ + vàng + xanh lá, 3 điện trở 330Ω, 2 nút nhấn 6×6mm, breadboard, dây cắm đực–cái | |
| Laptop có khe thẻ nhớ (hoặc đầu đọc thẻ) | để ghi hệ điều hành |

Không cần màn hình, bàn phím: mọi thứ làm từ laptop qua Wi-Fi.

## Bước 1. Ghi hệ điều hành vào thẻ

1. Cài **Raspberry Pi Imager** trên laptop (raspberrypi.com/software).
2. Chọn thiết bị **Raspberry Pi 4**, hệ điều hành **Raspberry Pi OS (other) → Raspberry Pi OS Lite (64-bit)**, rồi chọn thẻ nhớ.
3. Ở bước tuỳ chỉnh (Customisation / Edit settings):
   - Hostname: `pbl4-pi`
   - Tên người dùng và mật khẩu: tự đặt, ghi lại
   - Wi-Fi: **cùng mạng với laptop** (router phòng hoặc hotspot của nhóm), Country: VN
   - Múi giờ: Asia/Ho_Chi_Minh
   - Bật **SSH**, đăng nhập bằng mật khẩu
4. Ghi thẻ, cắm thẻ vào Pi, cắm nguồn, đợi khoảng 2 phút.

## Bước 2. Vào Pi từ laptop

Mở PowerShell trên laptop:

```
ssh <tên-người-dùng>@pbl4-pi.local
```

Không vào được thì mở trang quản lý router hoặc danh sách thiết bị của hotspot, tìm `pbl4-pi` để lấy IP, rồi `ssh <tên>@<ip>`.

## Bước 3. Lấy code và cài

Trên Pi:

```
sudo apt-get install -y git
git clone https://github.com/donghuyhoang/exam-behavior-monitor.git
cd exam-behavior-monitor/exam-behavior-monitor-setup
bash pi/setup.sh
```

Repo để private thì `git clone` sẽ hỏi tài khoản: dùng GitHub token, hoặc chép thư mục từ laptop sang bằng `scp -r`.

`setup.sh` chạy khoảng 5–10 phút (Pi phải có Internet). Nó cài Mosquitto, môi trường Python, rồi tạo 2 dịch vụ `pbl4-backend` và `pbl4-station` tự chạy mỗi khi cắm điện. Cuối cùng nó in ra **IP của Pi**: ghi lại.

Pi không cần file `.env` và không giữ mật khẩu camera.

## Thử máy chủ trước khi đấu đèn

Chưa đấu đèn thì trạm báo vẫn chạy nhưng không làm gì, không sao cả.

1. Trên Pi: `systemctl status mosquitto pbl4-backend --no-pager`, cả hai phải là `active (running)`.
2. Trên laptop mở trình duyệt: `http://<ip-của-pi>:8000/ui/`. Thấy giao diện (chưa có cảnh báo) là backend đã chạy.
3. Trên laptop: thêm `MQTT_HOST=<ip-của-pi>` vào `.env`, rồi chạy

   ```
   python -m backend.fake_events --every 3 --count 5
   ```

   Màn hình in `(kèm ảnh)` sau mỗi cảnh báo. Trang web ở bước 2 hiện thêm 5 cảnh báo giả, bấm vào thấy ảnh là laptop đã gửi được sự kiện và ảnh sang Pi.
4. Thử với camera thật: `python -m ai_service.main --show`.
5. Tắt laptop rồi mở trang web trên điện thoại: lịch sử và ảnh vẫn còn.

## Bước 4. Đấu dây

Tắt Pi trước khi đấu: `sudo poweroff`, đợi đèn trên Pi ngừng nháy rồi rút điện.

| Linh kiện | Nối vào chân GPIO | Chân số (đếm trên hàng 40 chân) |
|---|---|---|
| LED đỏ (qua điện trở 330Ω) | GPIO17 | 11 |
| LED vàng (qua điện trở 330Ω) | GPIO27 | 13 |
| LED xanh lá (qua điện trở 330Ω) | GPIO22 | 15 |
| Nút Xác nhận | GPIO5 | 29 |
| Nút Bỏ qua | GPIO6 | 31 |
| Thanh GND của breadboard | GND | 6 |

- Chân số 1 nằm ở đầu hàng chân **xa các cổng USB** (phía khe thẻ nhớ), ở hàng trong. Hàng trong (gần giữa mạch) là số lẻ 1, 3, 5…; hàng ngoài (sát mép mạch) là số chẵn 2, 4, 6…. Chân 1 là 3.3V, chân 2 bên cạnh là 5V: đừng nhầm.
- LED: chân dài (+) nối về phía điện trở và GPIO, chân ngắn (−) nối GND.
- Nút nhấn: một chân nối GPIO, chân **chéo đối diện** nối GND.

Thử dây (trên Pi, trong thư mục `exam-behavior-monitor-setup`):

```
sudo systemctl stop pbl4-station
.venv/bin/python -m pi.station --self-test
sudo systemctl start pbl4-station
```

Đúng dây thì đèn đỏ, vàng, xanh lần lượt sáng, nhấn nút nào thì in ra tên nút đó.

## Bước 5. Laptop

1. `git pull` để lấy code mới.
2. Thêm vào `.env` của laptop: `MQTT_HOST=<ip-của-pi>`
3. Từ giờ **không chạy** `tools/mqtt_broker.py` và `uvicorn backend...` trên laptop nữa, chỉ chạy AI:

```
python -m ai_service.main --show
```

Khi `MQTT_HOST` là máy khác, dịch vụ AI tự gửi kèm ảnh bằng chứng sang Pi. Ảnh vẫn được lưu một bản trên laptop.
Windows Firewall chỉ cần mở cổng 8001 (video) cho mạng Private.

## Bước 6. Giám thị mở giao diện

Trên điện thoại hoặc máy bất kỳ cùng mạng:

```
http://<ip-của-pi>:8000/ui/?video=http://<ip-laptop>:8001/video
```

Lưu thành dấu trang. Phần `?video=` để giao diện lấy video trực tiếp từ laptop.

## Đèn nghĩa là gì

| Thấy gì | Nghĩa là |
|---|---|
| Lúc cắm điện: 3 đèn cùng sáng 1 giây | trạm báo đã chạy |
| Xanh sáng | dịch vụ AI trên laptop đang chạy bình thường |
| Đỏ nháy nhanh 3 giây | vừa có cảnh báo mới |
| Đỏ sáng liền | còn 1 cảnh báo chưa xử lý |
| Đỏ nháy chậm | còn từ 2 cảnh báo chưa xử lý trở lên |
| Nhấn Xác nhận / Bỏ qua | xử lý cảnh báo **cũ nhất**, ghi `confirmed` / `dismissed`, web cập nhật ngay |
| Vàng sáng liền | AI vẫn chạy nhưng mất camera |
| Vàng nháy | mất AI: laptop tắt, treo hoặc mất mạng (quá 15 giây không có tin), hoặc laptop chưa chạy AI |

Không có còi nên đặt trạm báo ở chỗ giám thị luôn nhìn thấy, ví dụ ngay trên bàn giám thị.
Muốn thêm còi sau này: ghi `buzzer: 4` ở mục `pi_station` trong `config.yaml`, còi 3 chân nối VCC vào 3.3V (chân 1), I/O vào GPIO4 (chân 7).

Giám thị xử lý cảnh báo trên web thì trạm cũng tự tắt đèn đỏ của cảnh báo đó.

## Bài thử nghiệm thu

| Bài thử | Làm | Đạt khi |
|---|---|---|
| Báo khi mất AI | đang chạy thì tắt `ai_service` hoặc gập laptop | đèn vàng nháy trong vòng 15 giây; bật lại thì xanh sáng lại |
| Lịch sử không mất theo laptop | có 3–4 cảnh báo rồi tắt hẳn laptop | mở web trên điện thoại vẫn thấy đủ cảnh báo và ảnh |
| Báo tại bàn giám thị | thí sinh giơ điện thoại, quay video cả thí sinh lẫn đèn | đèn đỏ nháy trong vòng 3 giây |
| Tỷ lệ báo đúng do người chấm | giám thị bấm Xác nhận / Bỏ qua từng cảnh báo | cuối buổi đếm `confirmed` và `dismissed` trong lịch sử |
| Tự chạy lại sau mất điện | rút điện Pi rồi cắm lại | khoảng 1 phút sau đèn sáng lại, lịch sử còn nguyên |
| Chạy dài 30 phút | trên Pi: `.venv/bin/python tools/monitor_ram.py --minutes 30 --out runs/ram_pi.csv` | RAM ổn định |

## Lệnh hay dùng trên Pi

```
journalctl -u pbl4-station -f            # xem trạm báo đang làm gì
journalctl -u pbl4-backend -f            # xem backend
sudo systemctl restart pbl4-backend pbl4-station
mosquitto_sub -h localhost -t pbl4/room1/status -v   # xem laptop có gửi trạng thái không
vcgencmd get_throttled                   # 0x0 là nguồn đủ điện
hostname -I                              # xem IP của Pi
```

Cập nhật code trên Pi: `git pull` rồi `sudo systemctl restart pbl4-backend pbl4-station`.

## Lỗi hay gặp

| Hiện tượng | Cách xử lý |
|---|---|
| `ssh ...pbl4-pi.local` không vào được | dùng IP thay cho tên; kiểm tra Pi và laptop cùng Wi-Fi |
| Vàng nháy mãi dù laptop đang chạy AI | `MQTT_HOST` trong `.env` của laptop sai IP; IP của Pi đổi sau khi khởi động lại thì đặt IP cố định cho Pi trong router |
| Web có cảnh báo nhưng ảnh trống | laptop chưa `git pull` code mới (chưa gửi kèm ảnh) |
| Đèn không sáng | LED cắm ngược chân; chạy lại `--self-test` |
| Pi tự khởi động lại, `vcgencmd get_throttled` khác 0x0 | nguồn yếu, dùng nguồn USB-C 5V 3A |
