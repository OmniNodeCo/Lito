"""Public Agent API — custom Lito-Nano neural stack + tools."""

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
    """Lightest smart agent with a **custom** micro neural brain.

    Stack (all pure Python, no torch/numpy):
      • IntentNet  — supervised intent classifier (tool routing)
      • NanoLM     — tiny generative LM (chat / polish)
      • Tools      — calc, search, memory, shell, …
      • LocalReasoner fallback
      • Optional LITO_LLM_URL external model
    """

    def __init__(self) -> None:
        self.tools = build_tools()
        self.local = LocalReasoner(self.tools)
        self.llm = LLMReasoner(self.tools)
        self.show_thoughts = os.environ.get("LITO_SHOW_THOUGHTS", "1") != "0"
        self.force_local = os.environ.get("LITO_FORCE_LOCAL", "") == "1"
        self.prefer_external = os.environ.get("LITO_PREFER_EXTERNAL", "") == "1"

        self.nano = None
        self.intent = None
        self.nano_reasoner = None

        if not self.force_local:
            try:
                from .nano.runtime import try_load_brain, try_load_intent
                from .nano.reason import NanoReasoner

                self.intent = try_load_intent()
                self.nano = try_load_brain()
                if self.intent is not None or self.nano is not None:
                    self.nano_reasoner = NanoReasoner(
                        self.tools, intent=self.intent, brain=self.nano
                    )
            except Exception:
                self.nano_reasoner = None

    def handle(self, text: str) -> Reply:
        text = (text or "").strip()
        if self.prefer_external and self.llm.available() and not self.force_local:
            trace = self.llm.think(text)
        elif self.nano_reasoner is not None and not self.force_local:
            try:
                trace = self.nano_reasoner.think(text)
            except Exception:
                trace = self.local.think(text)
        else:
            trace = self.local.think(text)
        return Reply(text=trace.format(show_thoughts=self.show_thoughts), ok=True, trace=trace)
