"""Nguồn khung hình: camera RTSP, webcam hoặc file video, dùng chung một giao diện read()."""
import os
import threading
import time

import cv2


def _is_live(source):
    return isinstance(source, int) or str(source).startswith(("rtsp://", "http://", "https://"))


def _open(source):
    if isinstance(source, str) and source.startswith("rtsp://"):
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        return cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    return cv2.VideoCapture(source)


class FrameSource:
    def __init__(self, source, reconnect_s=2.0):
        if isinstance(source, str) and source.strip().isdigit():
            source = int(source)
        if isinstance(source, str) and "${" in source:
            raise ValueError(f"Chưa đặt biến môi trường cho nguồn video: {source}. Điền RTSP_URL trong .env hoặc dùng --source.")
        self.source = source
        self.live = _is_live(source)
        self.reconnect_s = reconnect_s
        self.cap = _open(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Không mở được nguồn video: {source}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 0.0
        self.frame_idx = 0
        self.camera_ok = True
        if self.live:
            self._lock = threading.Lock()
            self._new = threading.Event()
            self._frame = None
            self._captured_at = 0.0
            self._stop = False
            threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while not self._stop:
            ok, frame = self.cap.read()
            if not ok:
                self.camera_ok = False
                self.cap.release()
                time.sleep(self.reconnect_s)
                self.cap = _open(self.source)
                continue
            self.camera_ok = True
            with self._lock:
                self._frame, self._captured_at = frame, time.time()
            self._new.set()

    def read(self, timeout=2.0):
        if not self.live:
            ok, frame = self.cap.read()
            if not ok:
                return False, None, None, None
            self.frame_idx += 1
            fps = self.fps if self.fps > 0 else 25.0
            return True, frame, self.frame_idx / fps, time.time()
        if not self._new.wait(timeout):
            return False, None, None, None
        with self._lock:
            frame, captured_at = self._frame, self._captured_at
            self._new.clear()
        self.frame_idx += 1
        return True, frame, time.monotonic(), captured_at

    def close(self):
        if self.live:
            self._stop = True
        self.cap.release()
