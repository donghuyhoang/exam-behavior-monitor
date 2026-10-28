"""Backend: nhận sự kiện qua MQTT, lưu SQLite, đẩy cảnh báo mới qua WebSocket, phục vụ API và ảnh bằng chứng."""
import asyncio
import json
import sqlite3
import time
from contextlib import asynccontextmanager
from typing import Literal, Optional

import paho.mqtt.client as mqtt
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from settings import ROOT, load_config

cfg = load_config()
SNAP_DIR = ROOT / cfg["output"]["snapshot_dir"]
DB_PATH = ROOT / cfg["backend"]["db_path"]
SNAP_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
TOPIC_EVENTS, TOPIC_STATUS = cfg["mqtt"]["topic_events"], cfg["mqtt"]["topic_status"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY, ts TEXT NOT NULL, camera_id TEXT, type TEXT, confidence REAL,
    track_id INTEGER, student_track_id INTEGER, bbox TEXT, duration_s REAL, snapshot_path TEXT,
    status TEXT NOT NULL DEFAULT 'new'
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events (ts);
"""


def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def save_event(e):
    with db() as con:
        con.execute(
            "INSERT OR IGNORE INTO events (id, ts, camera_id, type, confidence, track_id, student_track_id,"
            " bbox, duration_s, snapshot_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (e["event_id"], e["ts"], e.get("camera_id"), e.get("type"), e.get("confidence"), e.get("track_id"),
             e.get("student_track_id"), json.dumps(e.get("bbox")), e.get("duration_s"), e.get("snapshot")))


def row_out(r):
    d = dict(r)
    d["event_id"] = d.pop("id")
    d["bbox"] = json.loads(d["bbox"]) if d["bbox"] else None
    d["snapshot_url"] = f"/{d['snapshot_path']}" if d["snapshot_path"] else None
    return d


def get_event(event_id):
    with db() as con:
        r = con.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
    return row_out(r) if r else None


class Hub:
    def __init__(self):
        self.clients = set()

    async def broadcast(self, message):
        for ws in list(self.clients):
            try:
                await ws.send_json(message)
            except Exception:
                self.clients.discard(ws)


hub = Hub()
latest_status = {}


def handle_message(topic, payload, loop):
    try:
        data = json.loads(payload)
    except (ValueError, TypeError):
        print("[mqtt] bỏ qua tin nhắn không phải JSON trên", topic)
        return
    if topic == TOPIC_EVENTS:
        if "event_id" not in data or "ts" not in data:
            print("[mqtt] sự kiện thiếu event_id/ts:", data)
            return
        save_event(data)
        message = {"kind": "event", "data": get_event(data["event_id"])}
    else:
        latest_status.update(data)
        latest_status["received_at"] = time.time()
        message = {"kind": "status", "data": dict(latest_status)}
    asyncio.run_coroutine_threadsafe(hub.broadcast(message), loop)


def start_mqtt(loop):
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="pbl4-backend")

    def on_connect(c, userdata, flags, reason_code, properties):
        print("[mqtt] kết nối:", reason_code)
        c.subscribe([(TOPIC_EVENTS, 1), (TOPIC_STATUS, 0)])

    client.on_connect = on_connect
    client.on_message = lambda c, userdata, msg: handle_message(msg.topic, msg.payload, loop)
    client.connect_async(cfg["mqtt"]["host"], int(cfg["mqtt"]["port"]), keepalive=30)
    client.loop_start()
    return client


@asynccontextmanager
async def lifespan(app):
    with db() as con:
        con.executescript(SCHEMA)
    client = start_mqtt(asyncio.get_running_loop())
    yield
    client.loop_stop()
    client.disconnect()


app = FastAPI(title="PBL4 backend", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/snapshots", StaticFiles(directory=SNAP_DIR), name="snapshots")
app.mount("/ui", StaticFiles(directory=ROOT / "ui", html=True), name="ui")


@app.get("/api/events")
def list_events(from_: Optional[str] = Query(None, alias="from"), to: Optional[str] = None,
                type: Optional[str] = None, status: Optional[str] = None, limit: int = 200):
    sql, args = "SELECT * FROM events WHERE 1=1", []
    for cond, value in (("ts >= ?", from_), ("ts <= ?", to), ("type = ?", type), ("status = ?", status)):
        if value:
            sql += f" AND {cond}"
            args.append(value)
    sql += " ORDER BY ts DESC LIMIT ?"
    args.append(min(limit, 1000))
    with db() as con:
        return [row_out(r) for r in con.execute(sql, args).fetchall()]


@app.get("/api/events/{event_id}")
def read_event(event_id: str):
    e = get_event(event_id)
    if e is None:
        raise HTTPException(404, "Không có sự kiện này")
    return e


class StatusUpdate(BaseModel):
    status: Literal["new", "confirmed", "dismissed"]


@app.patch("/api/events/{event_id}")
async def update_event(event_id: str, body: StatusUpdate):
    with db() as con:
        changed = con.execute("UPDATE events SET status = ? WHERE id = ?", (body.status, event_id)).rowcount
    if not changed:
        raise HTTPException(404, "Không có sự kiện này")
    e = get_event(event_id)
    await hub.broadcast({"kind": "update", "data": e})
    return e


@app.get("/api/status")
def read_status():
    received = latest_status.get("received_at", 0)
    return {**latest_status, "online": time.time() - received < 15}


@app.websocket("/ws/events")
async def ws_events(ws: WebSocket):
    await ws.accept()
    hub.clients.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        hub.clients.discard(ws)
