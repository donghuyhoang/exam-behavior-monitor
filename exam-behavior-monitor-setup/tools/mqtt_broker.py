"""Máy chủ MQTT dự phòng viết bằng Python (amqtt), dùng khi chưa cài được Mosquitto.

    python tools/mqtt_broker.py                 # lắng nghe 0.0.0.0:1883, giống Mosquitto với tools/mosquitto.conf
    python tools/mqtt_broker.py --port 18830

Chỉ chạy trong mạng riêng của nhóm (router, hotspot); máy chủ này không đòi mật khẩu.
"""
import argparse
import asyncio

from amqtt.broker import Broker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=1883)
    args = ap.parse_args()
    config = {
        "listeners": {"default": {"type": "tcp", "bind": f"{args.host}:{args.port}"}},
        "plugins": {"amqtt.plugins.authentication.AnonymousAuthPlugin": {"allow_anonymous": True}},
    }

    async def run():
        broker = Broker(config)
        await broker.start()
        print(f"Máy chủ MQTT đang chạy ở {args.host}:{args.port} (Ctrl+C để dừng)", flush=True)
        await asyncio.Event().wait()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
