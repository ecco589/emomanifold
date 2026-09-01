#!/usr/bin/env python3
"""Static model host with CORS + Range (no root required)."""
from __future__ import annotations

import os
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

ROOT = os.path.expanduser("~/emomanifold-models")
HOST = os.environ.get("EMO_MODEL_HOST", "0.0.0.0")
PORT = int(os.environ.get("EMO_MODEL_PORT", "8787"))


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Range")
        self.send_header("Access-Control-Expose-Headers", "Content-Length, Content-Range, Accept-Ranges")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.end_headers()

    def log_message(self, fmt: str, *args) -> None:
        print(f"[models] {self.address_string()} {fmt % args}")


def main() -> None:
    os.makedirs(ROOT, exist_ok=True)
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Serving {ROOT} on http://{HOST}:{PORT}/")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
