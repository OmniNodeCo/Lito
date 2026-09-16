"""Lightweight intent brain.

No ML weights in process. Pattern matching + optional remote LLM
(Ollama / OpenAI-compatible) so RAM stays tiny by default.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable

from . import actions
from .config import load_config


@dataclass(slots=True)
class Reply:
    text: str
    ok: bool = True
    kind: str = "chat"  # chat | action | error | help


# Patterns: (compiled regex, handler name, groupdict keys used)
# Order matters — first match wins.
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^\s*(help|commands|\?)\s*$", re.I), "help"),
    (re.compile(r"^\s*(hi|hello|hey|good\s+(morning|afternoon|evening)|howdy)\b", re.I), "greet"),
    (re.compile(r"^\s*(thanks|thank you|thx)\b", re.I), "thanks"),
    (re.compile(r"^\s*(bye|goodbye|see you|quit|exit)\b", re.I), "bye"),
    (
        re.compile(
            r"^\s*(?:how much ram|ram usage|memory usage|how much memory|"
            r"are you light|footprint)\b",
            re.I,
        ),
        "ram",
    ),
    (
        re.compile(
            r"^\s*(?:system info|sysinfo|about (?:this )?(?:pc|computer|machine)|host info)\s*$",
            re.I,
        ),
        "sysinfo",
    ),
    (re.compile(r"^\s*(?:what(?:'s| is)? the )?time\b|^\s*date\b", re.I), "time"),
    (
        re.compile(
            r"^\s*(?:list|show|what)\s+apps(?:\s+(?:like|matching|for)\s+(.+))?\s*$",
            re.I,
        ),
        "list_apps",
    ),
    (re.compile(r"^\s*(?:find|search)\s+apps?\s+(.+)$", re.I), "list_apps"),
    # Shell before open-app so "run echo hi" is not treated as an app name
    (
        re.compile(
            r"^\s*(?:run|exec(?:ute)?|shell|bash|cmd)\s+(.+)$",
            re.I,
        ),
        "shell",
    ),
    (
        re.compile(
            r"^\s*(?:open|launch|start|run app)\s+(?!https?://)(?!note\b)(.+)$",
            re.I,
        ),
        "open_app",
    ),
    (
        re.compile(
            r"^\s*(?:go to|browse|open url|open link|visit)\s+(\S+)$",
            re.I,
        ),
        "open_url",
    ),
    (re.compile(r"^\s*open\s+(https?://\S+)\s*$", re.I), "open_url"),
    (re.compile(r"^\s*open\s+((?:~|/|\./)\S+)\s*$", re.I), "open_path"),
    (
        re.compile(
            r"^\s*(?:find|search|locate)\s+files?(?:\s+named)?\s+(.+)$",
            re.I,
        ),
        "find_file",
    ),
    (
        re.compile(
            r"^\s*(?:search(?:\s+the)?\s+web|web search|google|duckduckgo|look up)\s+(.+)$",
            re.I,
        ),
        "web_search",
    ),
    (
        re.compile(
            r"^\s*(?:calc(?:ulate)?|math|what(?:'s| is)|compute)\s+(.+)$",
            re.I,
        ),
        "calc",
    ),
    # bare arithmetic
    (re.compile(r"^\s*([0-9\.\s\+\-\*\/\%\(\)\^x]+)\s*$"), "calc"),
    (
        re.compile(
            r"^\s*(?:note|take a note|remember this|jot)\s*[:\-]?\s+(.+)$",
            re.I,
        ),
        "note",
    ),
    (re.compile(r"^\s*(?:show|list|my)\s+notes\s*$", re.I), "show_notes"),
    (
        re.compile(
            r"^\s*remember\s+(?:that\s+)?(.+?)\s+(?:is|=)\s+(.+)$",
            re.I,
        ),
        "remember",
    ),
    (
        re.compile(
            r"^\s*(?:what(?:'s| is)|recall|remind me(?: of)?)\s+(.+)$",
            re.I,
        ),
        "recall",
    ),
    (
        re.compile(
            r"^\s*(?:set\s+)?volume\s+(?:to\s+)?(\d{1,3})\s*%?\s*$",
            re.I,
        ),
        "volume",
    ),
    (re.compile(r"^\s*(?:take a )?screenshot\b", re.I), "screenshot"),
    (
        re.compile(
            r"^\s*(?:fetch|read|summarize url|get)\s+(https?://\S+)\s*$",
            re.I,
        ),
        "fetch",
    ),
    (re.compile(r"^\s*status\s*$", re.I), "status"),
]


def _strip_filler(text: str) -> str:
    t = text.strip()
    # Drop polite filler prefixes
    t = re.sub(
        r"^(please|pls|can you|could you|would you|hey lito|lito[,:]?)\s+",
        "",
        t,
        flags=re.I,
    )
    t = re.sub(r"^(please|pls)\s+", "", t, flags=re.I)
    return t.strip()


class Brain:
    """Rule-based intent router with optional remote LLM fallback."""

    __slots__ = ("cfg",)

    def __init__(self) -> None:
        self.cfg = load_config()

    def reload(self) -> None:
        self.cfg = load_config()

    def handle(self, user_text: str) -> Reply:
        raw = (user_text or "").strip()
        if not raw:
            return Reply("Say something — try `help`.", kind="help")

        text = _strip_filler(raw)

        for pattern, name in _PATTERNS:
            m = pattern.match(text)
            if not m:
                continue
            handler: Callable[..., Reply] = getattr(self, f"_do_{name}")
            try:
                return handler(text, m)
            except Exception as exc:  # noqa: BLE001 — surface to user
                return Reply(f"Something went wrong: {exc}", ok=False, kind="error")

        # Soft open: "firefox please"
        soft = re.match(r"^([a-zA-Z0-9][\w\s\-\.]{0,40})$", text)
        if soft:
            from .apps import registry

            app = registry.find(text)
            if app:
                ok, msg = actions.open_app_by_name(text)
                return Reply(msg, ok=ok, kind="action")

        # Optional remote LLM (weights stay out of this process)
        llm = self._try_llm(raw)
        if llm is not None:
            return llm

        return Reply(
            "I didn't catch that. Try `open <app>`, `note <text>`, `calc 2+2`, "
            "or `help` for everything I can do.",
            ok=False,
            kind="help",
        )

    # --- handlers -----------------------------------------------------------

    def _do_help(self, text: str, m: re.Match) -> Reply:
        return Reply(actions.help_text(), kind="help")

    def _do_greet(self, text: str, m: re.Match) -> Reply:
        name = self.cfg.get("name", "Lito")
        return Reply(
            f"Hey — I'm **{name}**, your low-RAM desktop assistant. "
            f"Ask me to open apps, take notes, run quick tasks, or type `help`."
        )

    def _do_thanks(self, text: str, m: re.Match) -> Reply:
        return Reply("Anytime.")

    def _do_bye(self, text: str, m: re.Match) -> Reply:
        return Reply("Bye! Close the window or press Ctrl+C in the terminal when you're done.")

    def _do_ram(self, text: str, m: re.Match) -> Reply:
        return Reply(actions.lito_ram_usage() + "\n\nNo big model loaded — just Python + rules.")

    def _do_sysinfo(self, text: str, m: re.Match) -> Reply:
        return Reply(actions.system_info(), kind="action")

    def _do_time(self, text: str, m: re.Match) -> Reply:
        return Reply(actions.what_time())

    def _do_list_apps(self, text: str, m: re.Match) -> Reply:
        q = (m.group(1) or "").strip() if m.lastindex else ""
        return Reply(actions.list_apps_text(q), kind="action")

    def _do_open_app(self, text: str, m: re.Match) -> Reply:
        name = m.group(1).strip()
        lower = name.lower()
        if lower.startswith(("http://", "https://")):
            from .apps import open_url

            ok, msg = open_url(name)
            return Reply(msg, ok=ok, kind="action")
        if lower.startswith(("folder ", "file ", "directory ")):
            path = name.split(None, 1)[1]
            from .apps import open_path

            ok, msg = open_path(path)
            return Reply(msg, ok=ok, kind="action")
        ok, msg = actions.open_app_by_name(name)
        return Reply(msg, ok=ok, kind="action")

    def _do_open_url(self, text: str, m: re.Match) -> Reply:
        from .apps import open_url

        ok, msg = open_url(m.group(1).strip())
        return Reply(msg, ok=ok, kind="action")

    def _do_open_path(self, text: str, m: re.Match) -> Reply:
        from .apps import open_path

        ok, msg = open_path(m.group(1).strip())
        return Reply(msg, ok=ok, kind="action")

    def _do_find_file(self, text: str, m: re.Match) -> Reply:
        return Reply(actions.search_files(m.group(1).strip()), kind="action")

    def _do_web_search(self, text: str, m: re.Match) -> Reply:
        q = m.group(1).strip()
        if text.lower().startswith("google"):
            ok, msg = actions.google_search_open(q)
        else:
            ok, msg = actions.web_search_open(q)
        return Reply(msg, ok=ok, kind="action")

    def _do_calc(self, text: str, m: re.Match) -> Reply:
        expr = m.group(1).strip() if m.lastindex else text
        # If it was "what is X" and X isn't math-like, fall through via fail
        ok, msg = actions.safe_calc(expr)
        if not ok and not re.search(r"[\d\+\-\*\/]", expr):
            # maybe recall
            return self._do_recall(text, re.match(r"(.+)", expr))  # type: ignore[arg-type]
        return Reply(msg, ok=ok, kind="action")

    def _do_note(self, text: str, m: re.Match) -> Reply:
        return Reply(actions.take_note(m.group(1)), kind="action")

    def _do_show_notes(self, text: str, m: re.Match) -> Reply:
        return Reply(actions.show_notes(), kind="action")

    def _do_remember(self, text: str, m: re.Match) -> Reply:
        return Reply(actions.do_remember(m.group(1).strip(), m.group(2).strip()), kind="action")

    def _do_recall(self, text: str, m: re.Match) -> Reply:
        key = m.group(1).strip().rstrip("?.!")
        key = re.sub(r"^(the|my|our)\s+", "", key, flags=re.I)
        # "what is the time" already handled; "what is 2+2" calc
        if re.fullmatch(r"[0-9\.\s\+\-\*\/\%\(\)\^x]+", key):
            ok, msg = actions.safe_calc(key)
            return Reply(msg, ok=ok, kind="action")
        return Reply(actions.do_recall(key), kind="action")

    def _do_shell(self, text: str, m: re.Match) -> Reply:
        safe = bool(self.cfg.get("safe_shell", True))
        ok, msg = actions.run_shell(m.group(1), safe=safe)
        return Reply(f"```\n{msg}\n```", ok=ok, kind="action")

    def _do_volume(self, text: str, m: re.Match) -> Reply:
        level = int(m.group(1))
        ok, msg = actions.set_volume(level)
        return Reply(msg, ok=ok, kind="action")

    def _do_screenshot(self, text: str, m: re.Match) -> Reply:
        ok, msg = actions.screenshot()
        return Reply(msg, ok=ok, kind="action")

    def _do_fetch(self, text: str, m: re.Match) -> Reply:
        ok, msg = actions.fetch_url_text(m.group(1))
        return Reply(msg if ok else msg, ok=ok, kind="action")

    def _do_status(self, text: str, m: re.Match) -> Reply:
        s = actions.status_payload()
        return Reply(
            f"**{s['name']}** · RAM {s.get('ram_human') or '?'} · "
            f"{s['apps_known']} apps · {s['platform']} · Python {s['python']}",
            kind="action",
        )

    # --- optional remote LLM ------------------------------------------------

    def _try_llm(self, user_text: str) -> Reply | None:
        """Call an external OpenAI-compatible endpoint if configured.

        LITO_LLM_URL example: http://127.0.0.1:11434/v1/chat/completions
        LITO_LLM_MODEL example: llama3.2:1b
        Weights stay in the other process — Lito only holds the HTTP response.
        """
        url = os.environ.get("LITO_LLM_URL") or self.cfg.get("llm_url")
        if not url:
            return None
        model = os.environ.get("LITO_LLM_MODEL") or self.cfg.get("llm_model") or "llama3.2:1b"
        system = (
            "You are Lito, a concise desktop assistant. "
            "If the user wants to open an app or run a local action, reply with ONLY a JSON line: "
            '{"action":"open_app","name":"..."} or {"action":"note","text":"..."} or '
            '{"action":"shell","cmd":"..."} or {"action":"calc","expr":"..."}. '
            "Otherwise answer briefly in plain text. No markdown fences."
        )
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_text},
            ],
            "temperature": 0.2,
            "max_tokens": 300,
        }
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {os.environ.get('LITO_LLM_KEY', 'ollama')}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, OSError):
            return None
        try:
            content = payload["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError):
            return None
        # Try action JSON
        action_reply = self._maybe_action_json(content)
        if action_reply:
            return action_reply
        return Reply(content)

    def _maybe_action_json(self, content: str) -> Reply | None:
        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
        if not (text.startswith("{") and "action" in text):
            # find embedded json
            m = re.search(r"\{[^{}]+\}", text)
            if not m:
                return None
            text = m.group(0)
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(obj, dict) or "action" not in obj:
            return None
        act = str(obj["action"]).lower()
        if act == "open_app":
            ok, msg = actions.open_app_by_name(str(obj.get("name", "")))
            return Reply(msg, ok=ok, kind="action")
        if act == "note":
            return Reply(actions.take_note(str(obj.get("text", ""))), kind="action")
        if act == "shell":
            safe = bool(self.cfg.get("safe_shell", True))
            ok, msg = actions.run_shell(str(obj.get("cmd", "")), safe=safe)
            return Reply(f"```\n{msg}\n```", ok=ok, kind="action")
        if act == "calc":
            ok, msg = actions.safe_calc(str(obj.get("expr", "")))
            return Reply(msg, ok=ok, kind="action")
        if act == "open_url":
            from .apps import open_url

            ok, msg = open_url(str(obj.get("url", "")))
            return Reply(msg, ok=ok, kind="action")
        return None
