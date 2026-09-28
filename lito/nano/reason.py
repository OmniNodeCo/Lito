"""Nano neural intent + tools — smartest tiny custom brain."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from ..reasoner import LocalReasoner, Step, Trace
from ..tools import Tool, call_tool

if TYPE_CHECKING:
    from .intent import IntentNet
    from .runtime import NanoBrain


def _args(name: str, text: str) -> dict[str, Any]:
    t = text.strip()
    if name == "calc":
        m = re.search(r"([0-9][0-9\.\s\+\-\*\/\%\(\)\^x]*)", t)
        return {"expr": (m.group(1).strip() if m else t)}
    if name == "search":
        q = re.sub(
            r"^(what is|what's|who is|search(?: the)?(?: web)?|look up|explain|define|tell me about|google)\s+",
            "",
            t,
            flags=re.I,
        ).strip("?.! ")
        return {"query": q or t}
    if name == "remember":
        m = re.match(r"remember(?:\s+that)?\s+(.+?)\s+(?:is|=)\s+(.+)$", t, re.I)
        if m:
            return {"key": m.group(1).strip(), "value": m.group(2).strip()}
        return {"key": t, "value": ""}
    if name == "recall":
        m = re.match(r"^(?:recall|remind me(?: of)?|what(?:'s| is) my)\s+(.+)$", t, re.I)
        return {"key": (m.group(1) if m else t).strip("?.! ")}
    if name == "open":
        m = re.match(r"^(?:open|launch|start)\s+(.+)$", t, re.I)
        return {"target": (m.group(1) if m else t).strip()}
    if name == "shell":
        m = re.match(r"^(?:run|shell|exec)\s+(.+)$", t, re.I)
        return {"cmd": (m.group(1) if m else t).strip()}
    if name == "note":
        m = re.match(r"^(?:note|jot|take a note)\s*[:\-]?\s*(.+)$", t, re.I)
        return {"text": (m.group(1) if m else t).strip()}
    if name == "find":
        m = re.match(r"^(?:find file|find|locate)\s+(.+)$", t, re.I)
        return {"name": (m.group(1) if m else t).strip()}
    return {}


class NanoReasoner:
    """Custom neural stack:
    1) IntentNet (supervised MLP) picks tool — smart & tiny
    2) Tools execute with ground truth
    3) Optional NanoLM free-gen for chat polish
    """

    def __init__(
        self,
        tools: dict[str, Tool],
        intent: "IntentNet | None" = None,
        brain: "NanoBrain | None" = None,
    ):
        self.tools = tools
        self.intent = intent
        self.brain = brain
        self.fallback = LocalReasoner(tools)

    def think(self, user_text: str) -> Trace:
        text = (user_text or "").strip()
        trace = Trace(mode="nano")

        parts = []
        if self.intent:
            parts.append(f"IntentNet {self.intent.param_count():,}w")
        if self.brain:
            parts.append(self.brain.info)
        trace.steps.append(Step("**Lito-Nano** custom brain — " + " · ".join(parts) if parts else "nano"))

        route = "chat"
        conf = 0.0
        if self.intent is not None:
            route, probs = self.intent.predict(text)
            conf = max(probs)
            top = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:3]
            from .intent import TOOLS

            board = ", ".join(f"{TOOLS[i]}={probs[i]:.2f}" for i in top)
            trace.steps.append(Step(f"neural intent: **{route}** ({conf:.0%})  [{board}]"))
        else:
            trace.steps.append(Step("intent net missing — local fallback"))
            fb = self.fallback.think(text)
            trace.steps.extend(fb.steps)
            trace.answer = fb.answer
            trace.mode = "local"
            return trace

        # Low confidence → local reasoner assist
        if conf < 0.28:
            trace.steps.append(Step("low confidence — hybrid local reasoner"))
            fb = self.fallback.think(text)
            trace.answer = fb.answer
            trace.mode = "nano+local"
            return trace

        if route in {"chat", "help"}:
            if route == "help" or text.lower() in {"help", "?", "commands", "what can you do"}:
                fb = self.fallback.think("help")
                nparams = 0
                if self.intent:
                    nparams += self.intent.param_count()
                if self.brain:
                    nparams += self.brain.model.param_count()
                header = (
                    f"I'm **Lito-Nano** — your custom micro-LLM stack "
                    f"(**{nparams:,}** neural weights, pure Python, no torch).\n"
                    f"I classify intent neurally, then run tools.\n\n"
                )
                trace.answer = header + fb.answer
                return trace
            # chat / thanks
            low = text.lower().strip()
            if low in {"thanks", "thank you", "thx", "ty"}:
                trace.answer = "You're welcome. Ask me anything else."
                return trace
            if self.brain is not None:
                gen = self.brain.generate_chat(text)
                if gen and len(gen) > 10 and gen.count("<") < 2:
                    trace.steps.append(Step("nano LM chat generation"))
                    trace.answer = gen
                    return trace
            fb = self.fallback.think(text if low not in {"hi", "hello", "hey"} else "hello")
            trace.answer = fb.answer
            return trace

        if route not in self.tools:
            fb = self.fallback.think(text)
            trace.answer = fb.answer
            return trace

        args = _args(route, text)
        trace.steps.append(Step("tool", f"{route} {args}", ""))
        obs = call_tool(self.tools, route, args)
        trace.steps[-1].observation = str(obs)[:500]

        if route == "calc" and not str(obs).startswith("calc error"):
            trace.answer = f"**{args.get('expr','')}** = **{obs}**"
        elif route == "shell":
            trace.answer = f"```\n{obs}\n```"
        elif route == "search":
            q = args.get("query", text)
            body = str(obs)
            if "unavailable" in body.lower() and "Abstract" not in body:
                trace.answer = (
                    f"Neural route **search** «{q}».\nNetwork issue:\n```\n{body[:400]}\n```"
                )
            else:
                main = body.split("\n\n")[0][:900]
                polish = ""
                if self.brain is not None:
                    try:
                        polish = self.brain.complete_answer(text, f"intent {route}", f"search {q}", main)
                    except Exception:
                        polish = ""
                if polish and len(polish) > 12 and polish.count("<") < 2:
                    trace.answer = f"{polish.strip()}\n\n{main}"
                else:
                    trace.answer = f"**{q}**\n\n{main}"
        else:
            trace.answer = str(obs)
        return trace
