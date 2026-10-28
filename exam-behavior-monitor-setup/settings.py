"""Đọc config.yaml dùng chung cho mọi khối; giá trị dạng ${TEN_BIEN} lấy từ file .env hoặc biến môi trường."""
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent


def load_env(path=ROOT / ".env"):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_config(path=ROOT / "config.yaml"):
    load_env()
    text = os.path.expandvars(Path(path).read_text(encoding="utf-8"))
    return yaml.safe_load(text)
