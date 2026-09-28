"""Agent API — neural intent + generative replies (no preset cards)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .reasoner import LLMReasoner, Trace
from .tools import build_tools


@dataclass
class Reply:
    text: str
    ok: bool = True
    trace: Trace | None = None

    @property
    def kind(self) -> str:
        return "generate"


class Agent:
    """Custom micro brain that **generates** answers.

    - IntentNet routes tools when needed
    - Markov / WordLM / NanoLM write free-form text
    - Tools ground math/search/memory in facts
    - No static help menus or canned status cards as the primary reply
    """

    def __init__(self) -> None:
        self.tools = build_tools()
        self.llm = LLMReasoner(self.tools)
        self.show_thoughts = os.environ.get("LITO_SHOW_THOUGHTS", "1") != "0"
        self.force_local = os.environ.get("LITO_FORCE_LOCAL", "") == "1"
        self.prefer_external = os.environ.get("LITO_PREFER_EXTERNAL", "") == "1"

        self.intent = None
        self.nano = None
        self.markov = None
        self.wordlm = None
        self.nano_reasoner = None

        if not self.force_local:
            try:
                from .nano.reason import NanoReasoner
                from .nano.runtime import default_weights_dir, try_load_brain, try_load_intent

                self.intent = try_load_intent()
                self.nano = try_load_brain()
                wdir = default_weights_dir()
                mk = wdir / "lito-markov.json"
                if mk.exists():
                    from .nano.markov import MarkovGen

                    self.markov = MarkovGen.load(mk)
                else:
                    # train quickly in-memory / save
                    from .nano.markov import train_and_save

                    self.markov = train_and_save(mk)
                wl = wdir / "lito-wordlm.bin"
                if wl.exists():
                    try:
                        from .nano.wordlm import WordLM

                        self.wordlm = WordLM.load(wl)
                    except Exception:
                        self.wordlm = None

                self.nano_reasoner = NanoReasoner(
                    self.tools,
                    intent=self.intent,
                    brain=self.nano,
                    wordlm=self.wordlm,
                    markov=self.markov,
                )
            except Exception as exc:
                self.nano_reasoner = None
                self._init_error = exc
                # still try a bare markov so we never go fully mute
                try:
                    from .nano.markov import MarkovGen, train_and_save
                    from .nano.reason import NanoReasoner
                    from .nano.runtime import default_weights_dir, try_load_intent

                    mk = default_weights_dir() / "lito-markov.json"
                    markov = MarkovGen.load(mk) if mk.exists() else train_and_save(mk)
                    self.markov = markov
                    self.intent = self.intent or try_load_intent()
                    self.nano_reasoner = NanoReasoner(
                        self.tools, intent=self.intent, markov=markov
                    )
                except Exception as exc2:
                    self._init_error = (exc, exc2)

    def handle(self, text: str) -> Reply:
        text = (text or "").strip()
        if self.prefer_external and self.llm.available() and not self.force_local:
            trace = self.llm.think(text)
        elif self.nano_reasoner is not None:
            try:
                trace = self.nano_reasoner.think(text)
            except Exception as exc:
                trace = Trace(mode="error", answer=f"I hit a snag generating that: {exc}")
        else:
            # last ditch generative-ish line
            trace = Trace(
                mode="bare",
                answer=f"I hear “{text}”. My generative core is offline; restart me or retrain weights.",
            )
        out = trace.format(show_thoughts=self.show_thoughts)
        return Reply(text=out, ok=True, trace=trace)
