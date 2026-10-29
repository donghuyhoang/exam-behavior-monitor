# Bật toàn bộ hệ thống cho buổi demo, mỗi phần một cửa sổ. Chạy từ thư mục gốc repo:
#   powershell -ExecutionPolicy Bypass -File run_demo.ps1                        # camera thật (RTSP_URL trong .env)
#   powershell -ExecutionPolicy Bypass -File run_demo.ps1 -Source videos\test1.mp4   # video dự phòng
#   powershell -ExecutionPolicy Bypass -File run_demo.ps1 -Mosquitto              # dùng Mosquitto thay máy chủ MQTT bằng Python
param([string]$Source = "", [switch]$Mosquitto)

$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
Set-Location $PSScriptRoot

if ($Mosquitto) {
    Start-Process "C:\Program Files\mosquitto\mosquitto.exe" -ArgumentList "-c", "tools\mosquitto.conf", "-v"
} else {
    Start-Process $py -ArgumentList "tools\mqtt_broker.py"
}
Start-Sleep -Seconds 3
Start-Process $py -ArgumentList "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"
Start-Sleep -Seconds 4

$aiArgs = @("-m", "ai_service.main", "--show")
if ($Source) { $aiArgs += @("--source", $Source) }
Start-Process $py -ArgumentList $aiArgs
Start-Sleep -Seconds 8
Start-Process "http://localhost:8000/ui/"
