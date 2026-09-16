"""Terminal interface for Lito - even lighter than the web UI."""

from __future__ import annotations

import sys

from . import memory
from .actions import lito_ram_usage, status_payload
from .brain import Brain


BANNER = r"""
  _     _ _
 | |   (_) |_ ___
 | |   | | __/ _ \
 | |___| | || (_) |
 |_____|_|\__\___/   low-RAM desktop AI
"""


def _c(code: str, text: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"


def run_cli() -> int:
    brain = Brain()
    print(_c("92", BANNER.strip()))
    st = status_payload()
    print(
        _c("90", f"  RAM {st.get('ram_human') or '?'} - {st.get('apps_known', 0)} apps - type help - Ctrl+C quit")
    )
    print()
    while True:
        try:
            line = input(_c("92", "you › ") + _c("0", ""))
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            return 0
        line = line.strip()
        if not line:
            continue
        if line.lower() in {"quit", "exit", ":q"}:
            print("bye.")
            return 0
        reply = brain.handle(line)
        memory.append_history("user", line)
        memory.append_history("assistant", reply.text)
        # Plain terminal: strip light markdown
        text = reply.text.replace("**", "")
        color = "91" if not reply.ok else "96"
        print(_c(color, "lito ›"))
        for row in text.splitlines() or [""]:
            print(f"  {row}")
        print()
