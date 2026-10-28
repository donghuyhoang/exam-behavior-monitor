"""Phát video đã vẽ khung dạng MJPEG: giao diện chỉ cần <img src="http://<ip-laptop>:8001/video">."""
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2


class MjpegServer:
    def __init__(self, port=8001, width=1280, quality=80):
        self.width = width
        self.quality = quality
        self.jpeg = None
        self.cond = threading.Condition()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.path.startswith("/snapshot.jpg"):
                    with server.cond:
                        jpg = server.jpeg
                    if jpg is None:
                        self.send_error(503, "Chưa có khung hình")
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpg)))
                    self.end_headers()
                    self.wfile.write(jpg)
                    return
                if not self.path.startswith("/video"):
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                last = None
                try:
                    while True:
                        with server.cond:
                            server.cond.wait_for(lambda: server.jpeg is not None and server.jpeg is not last, timeout=5)
                            jpg = server.jpeg
                        if jpg is None or jpg is last:
                            continue
                        last = jpg
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass

        self.httpd = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        print(f"[mjpeg] http://0.0.0.0:{port}/video")

    def update(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        if w > self.width:
            frame_bgr = cv2.resize(frame_bgr, (self.width, int(h * self.width / w)))
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if ok:
            with self.cond:
                self.jpeg = buf.tobytes()
                self.cond.notify_all()

    def close(self):
        self.httpd.shutdown()
