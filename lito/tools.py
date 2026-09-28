"""Agent tools — each is a small pure function. Stdlib only."""

from __future__ import annotations

import ast
import math
import os
import platform
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from . import memory as mem

# ---------------------------------------------------------------------------
# Safe calculator (AST only — no eval of names/calls except math)
# ---------------------------------------------------------------------------

_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
_ALLOWED_UNARY = (ast.UAdd, ast.USub)
_MATH_FUNCS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "pi": math.pi,
    "e": math.e,
}


def _safe_eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Num):  # py3.9
        return float(node.n)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, _ALLOWED_UNARY):
        v = _safe_eval_node(node.operand)
        return v if isinstance(node.op, ast.UAdd) else -v
    if isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BINOPS):
        a, b = _safe_eval_node(node.left), _safe_eval_node(node.right)
        if isinstance(node.op, ast.Add):
            return a + b
        if isinstance(node.op, ast.Sub):
            return a - b
        if isinstance(node.op, ast.Mult):
            return a * b
        if isinstance(node.op, ast.Div):
            return a / b
        if isinstance(node.op, ast.FloorDiv):
            return a // b
        if isinstance(node.op, ast.Mod):
            return a % b
        if isinstance(node.op, ast.Pow):
            if abs(b) > 1000 or abs(a) > 1e6:
                raise ValueError("exponent too large")
            return a**b
    if isinstance(node, ast.Name) and node.id in _MATH_FUNCS:
        val = _MATH_FUNCS[node.id]
        if callable(val):
            raise ValueError(f"use {node.id}(...)")
        return float(val)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fn = _MATH_FUNCS.get(node.func.id)
        if not callable(fn):
            raise ValueError("call not allowed")
        args = [_safe_eval_node(a) for a in node.args]
        return float(fn(*args))
    raise ValueError("unsupported expression")


def tool_calc(expr: str) -> str:
    expr = expr.strip().replace("^", "**").replace("x", "*")
    try:
        tree = ast.parse(expr, mode="eval")
        val = _safe_eval_node(tree)
        if abs(val - round(val)) < 1e-12:
            return str(int(round(val)))
        return f"{val:.10g}"
    except Exception as exc:
        return f"calc error: {exc}"


# ---------------------------------------------------------------------------
# Shell (allowlist-ish — block obvious destruction)
# ---------------------------------------------------------------------------

_BLOCK = re.compile(
    r"(rm\s+-rf\s+/|mkfs|dd\s+if=|:\(\)\s*\{|shutdown|reboot|format\s+"
    r"|del\s+/[sf]|Remove-Item\s+-Recurse\s+-Force\s+C:\\)",
    re.I,
)


def tool_shell(cmd: str, timeout: float = 20.0) -> str:
    cmd = (cmd or "").strip()
    if not cmd:
        return "empty command"
    if _BLOCK.search(cmd):
        return "blocked: dangerous command pattern"
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(Path.home()),
        )
    except subprocess.TimeoutExpired:
        return f"timed out after {timeout}s"
    except OSError as exc:
        return str(exc)
    out = ((proc.stdout or "") + (proc.stderr or "")).strip()
    if len(out) > 4000:
        out = out[:4000] + "\n…(truncated)"
    return out or f"(exit {proc.returncode})"


def tool_sysinfo() -> str:
    lines = [
        f"OS: {platform.system()} {platform.release()} ({platform.machine()})",
        f"Python: {platform.python_version()}",
        f"Host: {platform.node()}",
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Home: {Path.home()}",
        f"CWD: {Path.cwd()}",
    ]
    # RAM if available
    try:
        if Path("/proc/meminfo").exists():
            info = Path("/proc/meminfo").read_text()
            total = re.search(r"MemTotal:\s+(\d+)", info)
            avail = re.search(r"MemAvailable:\s+(\d+)", info)
            if total and avail:
                t_mb = int(total.group(1)) / 1024
                a_mb = int(avail.group(1)) / 1024
                lines.append(f"RAM: {a_mb:.0f} MB free / {t_mb:.0f} MB total")
    except OSError:
        pass
    return "\n".join(lines)


def tool_time_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z").strip()


def tool_read_file(path: str, max_bytes: int = 30_000) -> str:
    p = Path(path).expanduser()
    if not p.exists():
        return f"not found: {p}"
    if p.is_dir():
        try:
            names = sorted(x.name for x in p.iterdir())[:80]
            return f"directory {p} ({len(names)} shown):\n" + "\n".join(names)
        except OSError as exc:
            return str(exc)
    try:
        data = p.read_bytes()[:max_bytes]
        text = data.decode("utf-8", errors="replace")
        if len(data) >= max_bytes:
            text += "\n…(truncated)"
        return text
    except OSError as exc:
        return str(exc)


def tool_write_note(text: str) -> str:
    mem.note(text)
    return f"noted ({len(text)} chars)"


def tool_list_notes() -> str:
    notes = mem.list_notes(15)
    if not notes:
        return "no notes yet"
    return "\n".join(
        f"- {datetime.fromtimestamp(n['ts']).strftime('%m-%d %H:%M')}: {n['text']}"
        for n in notes
    )


def tool_remember(key: str, value: str) -> str:
    mem.remember(key, value)
    return f"remembered **{key}** = {value}"


def tool_recall(key: str) -> str:
    v = mem.recall(key)
    return f"{key} = {v}" if v is not None else f"nothing stored for '{key}'"


def tool_memory_search(query: str) -> str:
    hits = mem.search_memory(query)
    return "\n".join(hits) if hits else "no memory matches"


def tool_open_path(target: str) -> str:
    target = target.strip().strip('"')
    if target.startswith(("http://", "https://")):
        return _open_url(target)
    p = Path(target).expanduser()
    if not p.exists():
        # try which
        w = shutil.which(target)
        if w:
            return _launch([w])
        return f"path not found: {target}"
    return _open_local(p)


def _open_url(url: str) -> str:
    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.Popen(["open", url], start_new_session=True)
        elif system == "Windows":
            os.startfile(url)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", url], start_new_session=True)
        return f"opened {url}"
    except OSError as exc:
        return str(exc)


def _open_local(p: Path) -> str:
    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.Popen(["open", str(p)], start_new_session=True)
        elif system == "Windows":
            os.startfile(str(p))  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(p)], start_new_session=True)
        return f"opened {p}"
    except OSError as exc:
        return str(exc)


def _launch(argv: list[str]) -> str:
    try:
        subprocess.Popen(argv, start_new_session=True)
        return f"launched {' '.join(argv)}"
    except OSError as exc:
        return str(exc)


def tool_web_search(query: str) -> str:
    """DuckDuckGo instant answer + a few HTML titles (stdlib HTTP)."""
    q = query.strip()
    if not q:
        return "empty query"
    parts: list[str] = []
    # Instant Answer API
    try:
        url = "https://api.duckduckgo.com/?" + urllib.parse.urlencode(
            {"q": q, "format": "json", "no_html": 1, "skip_disambig": 1}
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Lito/0.2"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            import json

            data = json.loads(resp.read().decode("utf-8", "replace"))
        if data.get("AbstractText"):
            parts.append(data["AbstractText"][:800])
            if data.get("AbstractURL"):
                parts.append(f"source: {data['AbstractURL']}")
        if data.get("Answer"):
            parts.append(str(data["Answer"])[:400])
        related = data.get("RelatedTopics") or []
        tips = []
        for item in related[:5]:
            if isinstance(item, dict) and item.get("Text"):
                tips.append(f"- {item['Text'][:160]}")
        if tips:
            parts.append("related:\n" + "\n".join(tips))
    except Exception as exc:
        parts.append(f"(instant answer unavailable: {exc})")

    # Wikipedia summary
    try:
        title = urllib.parse.quote(q.replace(" ", "_"))
        wurl = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
        req = urllib.request.Request(wurl, headers={"User-Agent": "Lito/0.2"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            import json

            w = json.loads(resp.read().decode("utf-8", "replace"))
        extract = (w.get("extract") or "")[:600]
        if extract and extract not in "\n".join(parts):
            parts.append(f"wikipedia: {extract}")
            if w.get("content_urls", {}).get("desktop", {}).get("page"):
                parts.append(f"wiki: {w['content_urls']['desktop']['page']}")
    except Exception:
        pass

    # Lightweight HTML scrape for top links
    try:
        hurl = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": q})
        req = urllib.request.Request(hurl, headers={"User-Agent": "Lito/0.2"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            html = resp.read(80_000).decode("utf-8", "replace")
        links = re.findall(
            r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            html,
            flags=re.I | re.S,
        )
        if links:
            lines = []
            for href, title in links[:5]:
                title = re.sub(r"<[^>]+>", "", title).strip()
                href = urllib.parse.unquote(href)
                # DDG redirect
                m = re.search(r"uddg=([^&]+)", href)
                if m:
                    href = urllib.parse.unquote(m.group(1))
                lines.append(f"- {title[:80]} — {href[:120]}")
            parts.append("top links:\n" + "\n".join(lines))
    except Exception:
        pass

    if not parts:
        return f"no results for: {q}"
    return "\n\n".join(parts)


def tool_fetch(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    req = urllib.request.Request(url, headers={"User-Agent": "Lito/0.2"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read(40_000)
            ctype = resp.headers.get("Content-Type", "")
    except Exception as exc:
        return f"fetch error: {exc}"
    text = raw.decode("utf-8", "replace")
    if "html" in ctype.lower() or text.lstrip().lower().startswith("<!DOCTYPE") or "<html" in text[:200].lower():
        text = re.sub(r"(?is)<script.*?>.*?</script>", " ", text)
        text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
        text = re.sub(r"(?is)<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
    return text[:3500] + ("…" if len(text) > 3500 else "")


def tool_find_files(name: str, root: str = "") -> str:
    base = Path(root).expanduser() if root else Path.home()
    name_l = name.lower()
    hits: list[str] = []
    deadline = time.monotonic() + 3.0
    try:
        for dirpath, dirnames, filenames in os.walk(base):
            if time.monotonic() > deadline or len(hits) >= 25:
                break
            # prune heavy dirs
            dirnames[:] = [
                d
                for d in dirnames
                if d not in {".git", "node_modules", "__pycache__", ".cache", "Library"}
                and not d.startswith(".")
            ]
            for fn in filenames:
                if name_l in fn.lower():
                    hits.append(str(Path(dirpath) / fn))
                    if len(hits) >= 25:
                        break
    except OSError as exc:
        return str(exc)
    if not hits:
        return f"no files matching '{name}' under {base}"
    return f"found {len(hits)}:\n" + "\n".join(hits)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass
class Tool:
    name: str
    description: str
    schema: str  # human-readable args
    fn: Callable[..., str]


def build_tools() -> dict[str, Tool]:
    tools = [
        Tool("calc", "Evaluate a math expression safely", "expr:str", tool_calc),
        Tool("shell", "Run a short shell command (blocked if dangerous)", "cmd:str", tool_shell),
        Tool("sysinfo", "OS, RAM, time, paths", "", tool_sysinfo),
        Tool("time", "Current local date/time", "", tool_time_now),
        Tool("read", "Read a text file or list a directory", "path:str", tool_read_file),
        Tool("note", "Save a short note", "text:str", tool_write_note),
        Tool("notes", "List recent notes", "", tool_list_notes),
        Tool("remember", "Store key=value in long-term memory", "key:str, value:str", tool_remember),
        Tool("recall", "Recall a remembered key", "key:str", tool_recall),
        Tool("memory", "Search notes and memories", "query:str", tool_memory_search),
        Tool("open", "Open a URL, file, folder, or app name", "target:str", tool_open_path),
        Tool("search", "Search the web (answers + links)", "query:str", tool_web_search),
        Tool("fetch", "Fetch a URL and return text", "url:str", tool_fetch),
        Tool("find", "Find files by name under home", "name:str", tool_find_files),
    ]
    return {t.name: t for t in tools}


def tools_prompt(tools: dict[str, Tool]) -> str:
    lines = []
    for t in tools.values():
        args = f"({t.schema})" if t.schema else "()"
        lines.append(f"- {t.name}{args}: {t.description}")
    return "\n".join(lines)


def call_tool(tools: dict[str, Tool], name: str, args: dict[str, Any] | None = None) -> str:
    t = tools.get(name)
    if not t:
        return f"unknown tool: {name}"
    args = args or {}
    try:
        if not t.schema:
            return t.fn()
        keys = [p.split(":")[0].strip() for p in t.schema.split(",") if p.strip()]
        if not args:
            return "missing arguments"
        vals = []
        for k in keys:
            if k in args:
                vals.append(args[k])
            elif len(keys) == 1:
                vals.append(next(iter(args.values())))
            else:
                return f"missing arg: {k}"
        return t.fn(*vals)
    except Exception as exc:
        return f"tool error: {exc}"
