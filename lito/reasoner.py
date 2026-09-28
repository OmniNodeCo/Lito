"""Local multi-step reasoner — thinks without loading model weights.

Strategy
--------
1. Understand: classify the ask, pull memory, extract entities.
2. Plan: choose 0–N tools with arguments.
3. Act: run tools, collect observations.
4. Reflect: if the answer is weak, try another tool or reframe.
5. Answer: synthesize a clear reply + show a short thought trace.

Optional neural path: if LITO_LLM_URL is set (Ollama / OpenAI-compatible),
the agent can call an external model for harder reasoning while weights
stay out-of-process (this process stays tiny).
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from . import memory as mem
from .tools import Tool, build_tools, call_tool, tools_prompt


@dataclass
class Step:
    thought: str
    action: str = ""
    observation: str = ""


@dataclass
class Trace:
    steps: list[Step] = field(default_factory=list)
    answer: str = ""
    mode: str = "local"  # local | llm

    def format(self, *, show_thoughts: bool = True) -> str:
        parts: list[str] = []
        if show_thoughts and self.steps:
            parts.append("**thinking**")
            for i, s in enumerate(self.steps, 1):
                line = f"{i}. {s.thought}"
                if s.action:
                    line += f"\n   → `{s.action}`"
                if s.observation:
                    obs = s.observation.strip()
                    if len(obs) > 400:
                        obs = obs[:400] + "…"
                    line += f"\n   ↳ {obs}"
                parts.append(line)
            parts.append("")
        parts.append(self.answer)
        return "\n".join(parts).strip()


# ---------------------------------------------------------------------------
# Heuristic planners (fast path, zero ML)
# ---------------------------------------------------------------------------

_MATH_RE = re.compile(
    r"(?:what(?:'s| is)|calculate|compute|solve)?\s*"
    r"([0-9\.\s\+\-\*\/\%\(\)\^x]+|"
    r"sqrt\s*\([^\)]+\)|"
    r"[0-9\.]+\s*(?:plus|minus|times|over|divided by)\s*[0-9\.]+)",
    re.I,
)
_WORD_MATH = (
    (r"\bplus\b", "+"),
    (r"\bminus\b", "-"),
    (r"\btimes\b|\bmultiplied by\b", "*"),
    (r"\bdivided by\b|\bover\b", "/"),
)


def _normalize_math(text: str) -> str | None:
    t = text.strip().rstrip("?.!")
    for pat, rep in _WORD_MATH:
        t = re.sub(pat, rep, t, flags=re.I)
    # bare expression
    if re.fullmatch(r"[0-9\.\s\+\-\*\/\%\(\)\^x]+", t):
        return t
    m = re.match(
        r"^(?:calc(?:ulate)?|compute|solve)\s+(.+)$", t, re.I
    )
    if m:
        expr = m.group(1).strip()
        if re.search(r"[0-9]", expr):
            return expr
    m = re.search(
        r"(?:what(?:'s| is))\s+([0-9\.\s\+\-\*\/\%\(\)\^x]+)$", t, re.I
    )
    if m:
        return m.group(1).strip()
    if re.search(r"[0-9]\s*[\+\-\*\/\^x]\s*[0-9]", t):
        m2 = re.search(r"([0-9\.\s\+\-\*\/\%\(\)\^x]{3,})", t)
        if m2:
            return m2.group(1).strip()
    return None


def _wants_web(text: str) -> bool:
    low = text.lower()
    triggers = (
        "search",
        "look up",
        "google",
        "who is",
        "what is",
        "what's",
        "tell me about",
        "explain",
        "news",
        "latest",
        "weather",
        "define",
        "wikipedia",
        "how does",
        "why is",
        "when did",
        "where is",
    )
    # skip pure memory / calc
    if _normalize_math(text):
        return False
    if low.startswith(("remember ", "note ", "recall ", "open ", "run ", "shell ")):
        return False
    return any(t in low for t in triggers)


def _extract_search_query(text: str) -> str:
    t = text.strip()
    for pat in (
        r"^(?:search(?:\s+the)?\s+web|web search|google|look up|search for)\s+",
        r"^(?:what(?:'s| is)|who is|tell me about|explain|define)\s+",
        r"^(?:ask\s+(?:the\s+)?(?:internet|web)|internet\s+search)\s+",
    ):
        t2 = re.sub(pat, "", t, flags=re.I).strip().rstrip("?.!")
        if t2 != t:
            return t2
    return t.rstrip("?.!")


class LocalReasoner:
    """Deterministic multi-step thinker with tools."""

    def __init__(self, tools: dict[str, Tool] | None = None, *, max_steps: int = 6):
        self.tools = tools or build_tools()
        self.max_steps = max_steps
        self.show_thoughts = os.environ.get("LITO_SHOW_THOUGHTS", "1") != "0"

    def think(self, user_text: str) -> Trace:
        text = (user_text or "").strip()
        trace = Trace(mode="local")
        if not text:
            trace.answer = "Say something — ask a question, or try `help`."
            return trace

        low = text.lower().strip()

        # --- greet / help / meta ---
        if low in {"hi", "hello", "hey", "yo", "sup"}:
            trace.steps.append(Step("Greeting detected — reply briefly."))
            trace.answer = (
                "Hey. I'm **Lito** — a tiny thinking agent.\n"
                "I plan, use tools, and answer. Ask me anything, or say `help`."
            )
            return trace

        if low in {"help", "?", "commands", "what can you do"}:
            trace.steps.append(Step("User wants capabilities — list tools + patterns."))
            trace.answer = self._help()
            return trace

        # --- memory write ---
        m = re.match(r"^(?:remember(?:\s+that)?)\s+(.+?)\s+(?:is|=)\s+(.+)$", text, re.I)
        if m:
            key, val = m.group(1).strip(), m.group(2).strip()
            trace.steps.append(Step(f"Store memory `{key}`.", f"remember({key})", ""))
            obs = call_tool(self.tools, "remember", {"key": key, "value": val})
            trace.steps[-1].observation = obs
            trace.answer = obs
            return trace

        m = re.match(r"^(?:note|jot|take a note)\s*[:\-]?\s+(.+)$", text, re.I)
        if m:
            trace.steps.append(Step("Save a note.", "note(...)", ""))
            obs = call_tool(self.tools, "note", {"text": m.group(1)})
            trace.steps[-1].observation = obs
            trace.answer = obs
            return trace

        if re.match(r"^(?:show|list|my)\s+notes\s*$", low):
            trace.steps.append(Step("List notes.", "notes()", ""))
            obs = call_tool(self.tools, "notes", {})
            trace.steps[-1].observation = obs
            trace.answer = obs
            return trace

        m = re.match(r"^(?:recall|remind me(?: of)?)\s+(.+)$", text, re.I)
        if m:
            key = m.group(1).strip().rstrip("?.!")
            trace.steps.append(Step(f"Recall `{key}`.", f"recall({key})", ""))
            obs = call_tool(self.tools, "recall", {"key": key})
            trace.steps[-1].observation = obs
            # also search if miss
            if "nothing stored" in obs.lower():
                trace.steps.append(Step("Direct recall missed — search memory.", f"memory({key})", ""))
                obs2 = call_tool(self.tools, "memory", {"query": key})
                trace.steps[-1].observation = obs2
                trace.answer = obs2 if "no memory" not in obs2.lower() else obs
            else:
                trace.answer = obs
            return trace

        # --- open / run ---
        m = re.match(r"^(?:open|launch|start)\s+(.+)$", text, re.I)
        if m:
            target = m.group(1).strip()
            trace.steps.append(Step(f"Open `{target}`.", f"open({target})", ""))
            obs = call_tool(self.tools, "open", {"target": target})
            trace.steps[-1].observation = obs
            trace.answer = obs
            return trace

        m = re.match(r"^(?:run|shell|exec)\s+(.+)$", text, re.I)
        if m:
            cmd = m.group(1).strip()
            trace.steps.append(Step("Run shell command (safety filter applies).", f"shell({cmd})", ""))
            obs = call_tool(self.tools, "shell", {"cmd": cmd})
            trace.steps[-1].observation = obs
            trace.answer = f"```\n{obs}\n```"
            return trace

        m = re.match(r"^(?:read|cat|show file)\s+(.+)$", text, re.I)
        if m:
            path = m.group(1).strip()
            trace.steps.append(Step(f"Read `{path}`.", f"read({path})", ""))
            obs = call_tool(self.tools, "read", {"path": path})
            trace.steps[-1].observation = obs
            trace.answer = f"```\n{obs}\n```"
            return trace

        m = re.match(r"^(?:find file|find files|locate)\s+(.+)$", text, re.I)
        if m:
            name = m.group(1).strip()
            trace.steps.append(Step(f"Find files named like `{name}`.", f"find({name})", ""))
            obs = call_tool(self.tools, "find", {"name": name})
            trace.steps[-1].observation = obs
            trace.answer = obs
            return trace

        if re.match(
            r"^(?:sysinfo|system info|status|whoami|ram|memory usage|"
            r"how much ram|free ram|memory)\s*\??$",
            low,
        ):
            trace.steps.append(Step("Report system status.", "sysinfo()", ""))
            obs = call_tool(self.tools, "sysinfo", {})
            trace.steps[-1].observation = obs
            trace.answer = obs
            return trace

        if re.match(r"^(?:time|date|what time|what'?s the time)\s*\??$", low):
            trace.steps.append(Step("Current time.", "time()", ""))
            obs = call_tool(self.tools, "time", {})
            trace.steps[-1].observation = obs
            trace.answer = f"**{obs}**"
            return trace

        # --- math ---
        expr = _normalize_math(text)
        if expr:
            trace.steps.append(Step(f"This looks like math: `{expr}`.", f"calc({expr})", ""))
            obs = call_tool(self.tools, "calc", {"expr": expr})
            trace.steps[-1].observation = obs
            if not obs.startswith("calc error"):
                trace.answer = f"**{expr.strip()}** = **{obs}**"
                return trace
            trace.steps.append(Step("Calc failed — will try other strategies."))

        # --- memory first for "what is X" if we know X ---
        m = re.match(r"^(?:what(?:'s| is)|whats)\s+(.+)$", text, re.I)
        if m:
            key = re.sub(r"^(the|my|our)\s+", "", m.group(1).strip().rstrip("?.!"), flags=re.I)
            got = mem.recall(key)
            if got is not None:
                trace.steps.append(Step(f"Found `{key}` in memory.", f"recall({key})", got))
                trace.answer = f"**{key}** = {got}"
                return trace
            # if short key, also memory search
            hits = mem.search_memory(key, limit=3)
            if hits:
                trace.steps.append(Step("Partial memory hits.", "memory(...)", "\n".join(hits)))

        # --- web / knowledge ---
        if _wants_web(text) or self._needs_knowledge(text):
            q = _extract_search_query(text)
            trace.steps.append(
                Step(
                    f"I need outside knowledge about «{q}». Searching.",
                    f"search({q})",
                    "",
                )
            )
            obs = call_tool(self.tools, "search", {"query": q})
            trace.steps[-1].observation = obs
            trace.answer = self._synthesize_from_search(q, obs)
            return trace

        # --- multi-hop: try memory search + light inference ---
        trace.steps.append(Step("No direct pattern — search local memory, then reason."))
        hits = mem.search_memory(text, limit=5)
        if hits:
            trace.steps.append(Step("Memory hits found.", "memory(...)", "\n".join(hits)))
            trace.answer = (
                "From what I remember:\n"
                + "\n".join(f"- {h}" for h in hits)
                + "\n\nAsk me to **search** the web if you need fresher info."
            )
            return trace

        # --- last resort: if it looks like a question, web search ---
        if text.endswith("?") or len(text.split()) >= 4:
            q = text.rstrip("?.!")
            trace.steps.append(
                Step(
                    "Treating this as a question — gather facts from the web.",
                    f"search({q})",
                    "",
                )
            )
            obs = call_tool(self.tools, "search", {"query": q})
            trace.steps[-1].observation = obs
            if obs and "no results" not in obs.lower() and "unavailable" not in obs.lower()[:80]:
                trace.answer = self._synthesize_from_search(q, obs)
                return trace

        trace.steps.append(Step("Insufficient signal — ask for clarification."))
        trace.answer = (
            "I'm not sure yet. I can:\n"
            "- **search** the web for facts\n"
            "- **calc** math\n"
            "- **remember** / **recall** things\n"
            "- **open** apps/URLs, **run** safe shell, **read** files\n"
            "Try rephrasing, or say `help`."
        )
        return trace

    def _needs_knowledge(self, text: str) -> bool:
        low = text.lower()
        if any(w in low for w in ("how to", "difference between", "compare", "pros and cons")):
            return True
        # capitalised proper nouns often need lookup
        if re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", text) and len(text.split()) <= 12:
            return True
        return False

    def _synthesize_from_search(self, query: str, obs: str) -> str:
        """Turn tool output into a readable answer (extractive, not hallucinated)."""
        if not obs or obs.startswith("no results"):
            return f"I couldn't find solid info on **{query}**."
        # Prefer abstract / wikipedia paragraphs before link dumps
        blocks = [b.strip() for b in re.split(r"\n\s*\n", obs) if b.strip()]
        summary_bits = []
        links = []
        for b in blocks:
            if b.lower().startswith("top links"):
                links.append(b)
            elif b.lower().startswith("related"):
                summary_bits.append(b)
            elif b.lower().startswith("source:") or b.lower().startswith("wiki:"):
                links.append(b)
            else:
                summary_bits.append(b)
        body = "\n\n".join(summary_bits[:3]) if summary_bits else obs[:1200]
        out = f"**{query}**\n\n{body}"
        if links:
            out += "\n\n" + "\n\n".join(links[:2])
        out += "\n\n_Sources gathered live; I don't invent facts beyond them._"
        return out

    def _help(self) -> str:
        return (
            "**Lito** — lightest thinking agent (tiny RAM, stdlib only)\n\n"
            "I **think in steps**: understand → plan → use tools → answer.\n\n"
            f"**Tools**\n{tools_prompt(self.tools)}\n\n"
            "**Examples**\n"
            "- `what is the walrus operator`\n"
            "- `calculate 2^10 + 5`\n"
            "- `remember wifi is orchard-5G`\n"
            "- `open https://example.com`\n"
            "- `run echo hello`\n"
            "- `search web MQTT QoS levels`\n"
            "- `note buy milk`\n\n"
            "**Neural mode (optional)** — point me at a local model without "
            "loading weights here:\n"
            "```\nexport LITO_LLM_URL=http://127.0.0.1:11434/v1/chat/completions\n"
            "export LITO_LLM_MODEL=llama3.2:1b\n```\n"
            "Thought traces: `LITO_SHOW_THOUGHTS=0` to hide."
        )


# ---------------------------------------------------------------------------
# Optional external LLM (weights stay elsewhere)
# ---------------------------------------------------------------------------


class LLMReasoner:
    """OpenAI-compatible chat endpoint (Ollama, llama.cpp server, etc.)."""

    def __init__(self, tools: dict[str, Tool] | None = None):
        self.tools = tools or build_tools()
        self.url = os.environ.get("LITO_LLM_URL", "").strip()
        self.model = os.environ.get("LITO_LLM_MODEL", "llama3.2:1b").strip()
        self.local = LocalReasoner(self.tools)

    def available(self) -> bool:
        return bool(self.url)

    def think(self, user_text: str) -> Trace:
        if not self.url:
            return self.local.think(user_text)

        system = (
            "You are Lito, a careful desktop agent with tiny RAM. "
            "Think step by step. When you need a tool, output ONE line exactly:\n"
            'TOOL name key=value key2=value2\n'
            "Wait for TOOL_RESULT. When done, output:\n"
            "ANSWER your final reply\n"
            "Be concise. Never invent tool results.\n"
            f"Tools:\n{tools_prompt(self.tools)}"
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ]
        trace = Trace(mode="llm")
        for round_i in range(5):
            raw = self._chat(messages)
            if raw is None:
                trace.steps.append(Step("LLM unreachable — falling back to local reasoner."))
                return self.local.think(user_text)
            messages.append({"role": "assistant", "content": raw})
            tool_line = None
            for line in raw.splitlines():
                if line.strip().upper().startswith("TOOL "):
                    tool_line = line.strip()
                    break
            if tool_line:
                name, args = self._parse_tool_line(tool_line)
                trace.steps.append(
                    Step(f"LLM requested tool `{name}`.", tool_line, "")
                )
                obs = call_tool(self.tools, name, args)
                trace.steps[-1].observation = obs
                messages.append({"role": "user", "content": f"TOOL_RESULT {name}:\n{obs}"})
                continue
            m = re.search(r"ANSWER\s+(.+)", raw, re.I | re.S)
            if m:
                trace.answer = m.group(1).strip()
                return trace
            # treat whole message as answer if no TOOL
            if "TOOL " not in raw.upper():
                trace.answer = raw.strip()
                return trace
        trace.answer = raw.strip() if raw else "I lost the thread."
        return trace

    def _parse_tool_line(self, line: str) -> tuple[str, dict[str, str]]:
        body = re.sub(r"^TOOL\s+", "", line, flags=re.I).strip()
        parts = body.split()
        if not parts:
            return "", {}
        name = parts[0]
        args: dict[str, str] = {}
        # key=value pairs; remainder as first schema key
        rest = []
        for p in parts[1:]:
            if "=" in p:
                k, v = p.split("=", 1)
                args[k] = v.strip('"')
            else:
                rest.append(p)
        if rest and not args:
            # single freeform arg
            t = self.tools.get(name)
            if t and t.schema:
                key = t.schema.split(":")[0].strip().split(",")[0]
                args[key] = " ".join(rest).strip('"')
            else:
                args["q"] = " ".join(rest)
        elif rest:
            # append to first value
            first = next(iter(args))
            args[first] = (args[first] + " " + " ".join(rest)).strip()
        return name, args

    def _chat(self, messages: list[dict[str, str]]) -> str | None:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 512,
        }
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self.url,
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": "Lito/0.2"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.loads(resp.read().decode("utf-8", "replace"))
        except Exception:
            return None
        try:
            return payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return None
