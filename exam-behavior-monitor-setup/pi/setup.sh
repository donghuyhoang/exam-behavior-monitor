#!/usr/bin/env bash
# Cài máy chủ phòng thi trên Raspberry Pi (Raspberry Pi OS Lite 64-bit): Mosquitto, backend, trạm báo.
# Chạy một lần, trong thư mục exam-behavior-monitor-setup:   bash pi/setup.sh
# Pi cần có Internet lúc cài. Chạy lại lần nữa cũng không sao.
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_USER="${SUDO_USER:-$USER}"
VENV="$APP_DIR/.venv"

echo "[1/5] Cài gói hệ thống (Mosquitto, Python, thư viện GPIO)"
sudo apt-get update
sudo apt-get install -y mosquitto mosquitto-clients python3-venv python3-gpiozero python3-lgpio

echo "[2/5] Cho Mosquitto nhận kết nối từ laptop trong cùng mạng"
sudo cp "$APP_DIR/tools/mosquitto.conf" /etc/mosquitto/conf.d/pbl4.conf
sudo systemctl enable mosquitto
sudo systemctl restart mosquitto

echo "[3/5] Môi trường Python (dùng chung gpiozero của hệ thống)"
python3 -m venv --system-site-packages "$VENV"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install fastapi "uvicorn[standard]" "paho-mqtt>=2.0" pyyaml psutil pillow

echo "[4/5] Tạo dịch vụ tự chạy khi cắm điện"
sudo tee /etc/systemd/system/pbl4-backend.service >/dev/null <<EOF
[Unit]
Description=PBL4 backend: API, SQLite, giao diện giám thị
After=network-online.target mosquitto.service
Wants=network-online.target

[Service]
User=$RUN_USER
WorkingDirectory=$APP_DIR
ExecStart=$VENV/bin/uvicorn backend.app:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

sudo tee /etc/systemd/system/pbl4-station.service >/dev/null <<EOF
[Unit]
Description=PBL4 trạm báo: 3 đèn, 2 nút cho giám thị
After=pbl4-backend.service mosquitto.service

[Service]
User=$RUN_USER
WorkingDirectory=$APP_DIR
ExecStart=$VENV/bin/python -m pi.station
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable pbl4-backend pbl4-station
sudo systemctl restart pbl4-backend pbl4-station

echo "[5/5] Xong. Địa chỉ IP của Pi (ghi vào .env trên laptop: MQTT_HOST=...):"
hostname -I
systemctl --no-pager --lines=0 status mosquitto pbl4-backend pbl4-station | grep -E "●|Active:" || true
