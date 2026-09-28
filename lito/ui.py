"""Minimal stdlib HTTP UI — no frameworks."""

from __future__ import annotations

import json
import mimetypes
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .agent import Agent

_STATIC = Path(__file__).resolve().parent / "static"
_agent = Agent()
_lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    server_version = "Lito/0.2"

    def log_message(self, fmt: str, *args) -> None:  # quieter
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code: int, body: bytes, content_type: str = "text/plain; charset=utf-8") -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            return self._file(_STATIC / "index.html")
        if path.startswith("/static/"):
            return self._file(_STATIC / path[len("/static/") :])
        # bare assets
        name = path.lstrip("/")
        cand = _STATIC / name
        if cand.is_file():
            return self._file(cand)
        self._send(404, b"not found")

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path not in {"/api/chat", "/chat"}:
            self._send(404, b"not found")
            return
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        try:
            data = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._send(400, b'{"error":"bad json"}', "application/json")
            return
        text = (data.get("message") or data.get("text") or "").strip()
        with _lock:
            reply = _agent.handle(text)
        payload = json.dumps(
            {"ok": reply.ok, "reply": reply.text, "text": reply.text},
            ensure_ascii=False,
        ).encode("utf-8")
        self._send(200, payload, "application/json; charset=utf-8")

    def _file(self, path: Path) -> None:
        if not path.is_file():
            self._send(404, b"not found")
            return
        data = path.read_bytes()
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype.endswith("javascript") or ctype == "application/json":
            ctype += "; charset=utf-8"
        self._send(200, data, ctype)


def run_server(host: str = "0.0.0.0", port: int = 8765) -> None:
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"Lito UI  http://{host}:{port}  (Ctrl+C to stop)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        httpd.server_close()
