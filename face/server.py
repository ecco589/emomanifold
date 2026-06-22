# -*- coding: utf-8 -*-
"""
Minimal HTTP bridge for M3:
  - serves the web UI in ./webui
  - POST /apply  {"v":..,"a":..,"time":..}  -> FKCN -> fuse -> face_driver

Run:  python server.py        (dry-run on macOS, live on Linux with serial)
Then open http://127.0.0.1:8770/

Python 3.8+ standard library only (+ face_driver deps).
"""
import json
import os
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from face_driver import FaceDriver, EMOTION_LABELS

HOST = "127.0.0.1"
PORT = 8770
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
WEBUI_DIR = os.path.join(_CURRENT_DIR, "webui")

_driver = FaceDriver()

_MIME = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # quiet

    def _send(self, code, body, content_type="application/json; charset=utf-8"):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self, path):
        if path == "/" or path == "":
            path = "/index.html"
        rel = path.lstrip("/")
        full = os.path.normpath(os.path.join(WEBUI_DIR, rel))
        if not full.startswith(WEBUI_DIR) or not os.path.isfile(full):
            self._send(404, "not found", "text/plain; charset=utf-8")
            return
        ext = os.path.splitext(full)[1].lower()
        ctype = _MIME.get(ext, "application/octet-stream")
        with open(full, "rb") as f:
            self._send(200, f.read(), ctype)

    def do_GET(self):
        if self.path.startswith("/status"):
            self._send(200, json.dumps({
                "dry_run": _driver.dry_run,
                "labels": EMOTION_LABELS,
            }))
            return
        self._serve_static(self.path)

    def do_POST(self):
        if not self.path.startswith("/apply"):
            self._send(404, "not found", "text/plain; charset=utf-8")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            v = float(payload.get("v", 0.0))
            a = float(payload.get("a", 0.0))
            time_ms = int(payload.get("time", 300))
            weights, factors = _driver.factors_from_va(v, a)
            _driver.apply_factors(factors, time_ms)
            self._send(200, json.dumps({
                "ok": True,
                "dry_run": _driver.dry_run,
                "weights": weights,
                "factors": factors,
            }))
        except Exception as e:  # noqa
            self._send(500, json.dumps({"ok": False, "error": str(e)}))


def main():
    url = "http://%s:%d/" % (HOST, PORT)
    print("FaceServer ->", url, "| mode:", "dry-run" if _driver.dry_run else "live", flush=True)
    try:
        webbrowser.open(url)
    except Exception:
        pass
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
