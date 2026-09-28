"""Public Agent API — one object, low RAM, actually thinks."""

from __future__ import annotations

import os
from dataclasses import dataclass

from .reasoner import LLMReasoner, LocalReasoner, Trace
from .tools import build_tools


@dataclass
class Reply:
    text: str
    ok: bool = True
    trace: Trace | None = None

    @property
    def kind(self) -> str:
        return "think"


class Agent:
    """Lightest thinking AI.

    - Default: local multi-step reasoner + tools (no model weights).
    - Optional: LITO_LLM_URL for a real LLM kept out-of-process.
    """

    def __init__(self) -> None:
        self.tools = build_tools()
        self.local = LocalReasoner(self.tools)
        self.llm = LLMReasoner(self.tools)
        self.show_thoughts = os.environ.get("LITO_SHOW_THOUGHTS", "1") != "0"
        # Prefer LLM when configured unless forced local
        self.prefer_llm = os.environ.get("LITO_FORCE_LOCAL", "") != "1"

    def handle(self, text: str) -> Reply:
        text = (text or "").strip()
        if self.prefer_llm and self.llm.available():
            # Fast local shortcuts still win for math/open/memory (cheaper)
            if self._local_fastpath(text):
                trace = self.local.think(text)
            else:
                trace = self.llm.think(text)
        else:
            trace = self.local.think(text)
        out = trace.format(show_thoughts=self.show_thoughts)
        return Reply(text=out, ok=True, trace=trace)

    def _local_fastpath(self, text: str) -> bool:
        low = text.lower()
        if low in {"help", "hi", "hello", "hey", "time", "date", "sysinfo", "status"}:
            return True
        if low.startswith(
            ("remember ", "note ", "recall ", "open ", "run ", "shell ", "calc ")
        ):
            return True
        return False
