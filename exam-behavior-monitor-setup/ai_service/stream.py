"""Nguồn khung hình: camera RTSP, webcam hoặc file video, dùng chung một giao diện read()."""
import os
import threading
import time

import cv2
import numpy as np
VLC_FIRST_FRAME_S = 8.0   # chờ khung hình đầu tiên từ camera tối đa 8 giây
VLC_STALL_S = 5.0         # quá 5 giây không có khung mới thì coi là mất camera

def _is_live(source):
    return isinstance(source, int) or str(source).startswith(("rtsp://", "http://", "https://"))


class _VLCCap:
    """Đọc RTSP bằng LibVLC trong tiến trình (dùng cho camera Yoosee mà OpenCV không mở được)."""

    def __init__(self, url, width=1280, height=720, first_frame_s=8.0, stall_s=5.0):
        import ctypes
        import vlc

        self.width = width
        self.height = height
        self.stall_s = stall_s              # quá chừng này giây không có khung mới thì coi là mất camera
        self._buf = (ctypes.c_ubyte * (width * height * 3))()
        self._frame = None
        self._got_frame = False
        self._lock = threading.Lock()
        self._new = threading.Event()       # bật mỗi khi VLC giải mã xong một khung MỚI
        self._fps = 25.0
        self._player = None
        self._instance = None

        LOCK_CB = vlc.CallbackDecorators.VideoLockCb
        UNLOCK_CB = vlc.CallbackDecorators.VideoUnlockCb
        DISPLAY_CB = vlc.CallbackDecorators.VideoDisplayCb

        @LOCK_CB
        def _lock(opaque, planes):
            planes[0] = ctypes.cast(self._buf, ctypes.c_void_p)
            return None

        @UNLOCK_CB
        def _unlock(opaque, picture, planes):
            arr = np.frombuffer(self._buf, dtype=np.uint8).reshape((self.height, self.width, 3))
            bgr = arr[:, :, ::-1].copy()    # nếu màu trên giao diện bị đảo đỏ/xanh thì bỏ [:, :, ::-1]
            with self._lock:
                self._frame = bgr
                self._got_frame = True
            self._new.set()

        @DISPLAY_CB
        def _display(opaque, picture):
            pass

        self._cbs = (_lock, _unlock, _display)  # giữ tham chiếu tránh Python garbage collect
        self._instance = vlc.Instance("--no-audio", "--network-caching=300", "--avcodec-hw=none")
        self._player = self._instance.media_player_new()
        media = self._instance.media_new(url)
        self._player.set_media(media)
        self._player.video_set_callbacks(_lock, _unlock, _display, None)
        self._player.video_set_format(b"RV24", width, height, width * 3)
        self._player.play()

        # Chờ khung hình đầu tiên tối đa first_frame_s giây
        self._new.wait(first_frame_s)

    def isOpened(self):
        return self._got_frame

    def read(self):
        # Chờ khung MỚI thay vì trả lại khung cũ ngay. Bản trước trả khung cũ liên tục nên:
        # vòng đọc quay hết một nhân CPU, AI chấm lại cùng một khung (FPS hiển thị cao giả),
        # và rút điện camera thì hình đứng yên, giao diện không báo mất kết nối, không tự nối lại.
        if self._player is None or not self._new.wait(self.stall_s):
            return False, None
        self._new.clear()
        with self._lock:
            return True, self._frame

    def native_size(self):
        """Kích thước gốc của luồng (VLC luôn trả khung đã đổi về width×height). (0, 0) nếu chưa biết."""
        try:
            return tuple(self._player.video_get_size(0))
        except Exception:
            return 0, 0

    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_FPS:
            return self._fps
        elif prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        return 0.0

    def release(self):
        if self._player:
            self._player.stop()
            self._player.release()
            self._player = None
        if self._instance:
            self._instance.release()
            self._instance = None


def _open(source):
    if isinstance(source, str) and source.startswith("rtsp://"):
        try:
            cap = _VLCCap(source, first_frame_s=VLC_FIRST_FRAME_S, stall_s=VLC_STALL_S)
        except Exception as e:  # chưa cài VLC 64-bit hoặc chưa pip install python-vlc
            print(f"[stream] LibVLC không khả dụng ({e}), chuyển sang OpenCV")
        else:
            if cap.isOpened():
                print("[stream] Đã kết nối camera qua LibVLC")
            else:
                print("[stream] LibVLC chưa nhận được khung nào (camera tắt, sai địa chỉ hoặc sai mật khẩu)")
            # Có VLC thì không rơi xuống OpenCV: OpenCV không đọc được camera này, chỉ làm lần nối lại chậm thêm ~30 giây
            return cap

        # 2. Không có VLC: dùng OpenCV
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