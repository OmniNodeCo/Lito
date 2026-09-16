"""Minimal local web UI served by stdlib http.server.

Single process, no frameworks, no websockets - keeps RAM tiny.
"""

from __future__ import annotations

import html
import json
import re
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from . import memory
from .actions import status_payload
from .brain import Brain
from .config import load_config


def _static_dir() -> Path:
    # PyInstaller one-file unpacks data under sys._MEIPASS
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        cand = Path(meipass) / "lito" / "static"
        if cand.is_dir():
            return cand
    return Path(__file__).resolve().parent / "static"


STATIC_DIR = _static_dir()

# Shared brain (stateless enough)
_brain = Brain()
_lock = threading.Lock()


def _read_static(name: str) -> bytes | None:
    path = STATIC_DIR / name
    if not path.is_file():
        return None
    try:
        return path.read_bytes()
    except OSError:
        return None


class Handler(BaseHTTPRequestHandler):
    server_version = "Lito/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        # Quiet by default - uncomment for debug
        pass

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        # Allow same-origin preview embedding
        self.send_header("X-Frame-Options", "SAMEORIGIN")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: Any) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self._send(code, data, "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            data = _read_static("index.html")
            if data is None:
                self._send(500, b"missing ui", "text/plain")
                return
            self._send(200, data, "text/html; charset=utf-8")
            return
        if path == "/style.css":
            data = _read_static("style.css") or b""
            self._send(200, data, "text/css; charset=utf-8")
            return
        if path == "/app.js":
            data = _read_static("app.js") or b""
            self._send(200, data, "application/javascript; charset=utf-8")
            return
        if path == "/api/status":
            self._json(200, status_payload())
            return
        if path == "/api/history":
            self._json(200, {"messages": memory.recent_history(50)})
            return
        if path == "/api/health":
            self._json(200, {"ok": True})
            return
        self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length") or 0)
        # Cap body - keep memory bounded
        if length > 32_000:
            self._json(413, {"ok": False, "error": "message too long"})
            return
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._json(400, {"ok": False, "error": "invalid json"})
            return

        if path == "/api/chat":
            text = str(payload.get("text") or "").strip()
            if not text:
                self._json(400, {"ok": False, "error": "empty"})
                return
            with _lock:
                reply = _brain.handle(text)
            memory.append_history("user", text)
            memory.append_history("assistant", reply.text)
            self._json(
                200,
                {
                    "ok": reply.ok,
                    "text": reply.text,
                    "kind": reply.kind,
                    "html": _md_lite(reply.text),
                },
            )
            return

        self._json(404, {"ok": False, "error": "not found"})

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def _md_lite(text: str) -> str:
    """Tiny markdown-ish renderer for chat bubbles (bold, code, breaks)."""
    s = html.escape(text)

    def fence(m: re.Match[str]) -> str:
        return f"<pre><code>{m.group(1)}</code></pre>"

    s = re.sub(r"```(?:\w+)?\n?(.*?)```", fence, s, flags=re.S)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = s.replace("\n", "<br>\n")
    return s


class LitoServer:
    def __init__(self, host: str | None = None, port: int | None = None) -> None:
        cfg = load_config()
        self.host = host or str(cfg.get("host", "127.0.0.1"))
        self.port = int(port or cfg.get("port", 8765))
        self.httpd: ThreadingHTTPServer | None = None

    def start(self, open_browser: bool = True, block: bool = True) -> None:
        # Bind 0.0.0.0 when host is all-interfaces for preview environments
        self.httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        actual_port = self.httpd.server_address[1]
        url = f"http://127.0.0.1:{actual_port}/"
        print(f"Lito UI -> {url}")
        print(f"Bound  -> {self.host}:{actual_port}")
        print("Tip: type 'help' in the chat. Ctrl+C to stop.")
        if open_browser:
            try:
                webbrowser.open(url)
            except Exception:
                pass
        if block:
            try:
                self.httpd.serve_forever(poll_interval=0.5)
            except KeyboardInterrupt:
                print("\nLito stopped.")
            finally:
                self.httpd.server_close()
        else:
            t = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.5}, daemon=True)
            t.start()

    def stop(self) -> None:
        if self.httpd:
            self.httpd.shutdown()
