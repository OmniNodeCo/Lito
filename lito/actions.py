"""Task actions Lito can perform. All stdlib — no heavy deps."""

from __future__ import annotations

import ast
import datetime as dt
import os
import platform
import re
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from . import memory
from .apps import launch, open_path, open_url, registry

# Shell denylist when safe_shell is on
_DANGEROUS = re.compile(
    r"(rm\s+-rf\s+/|mkfs|dd\s+if=|fork\s*\(|shutdown|reboot|poweroff|"
    r">\s*/dev/sd|chmod\s+-R\s+777\s+/|chown\s+-R\s+/)",
    re.I,
)


def system_info() -> str:
    uname = platform.uname()
    mem = _mem_info()
    disk = shutil.disk_usage(Path.home())
    lines = [
        f"**Host:** {socket.gethostname()}",
        f"**OS:** {uname.system} {uname.release} ({uname.machine})",
        f"**Python:** {platform.python_version()}",
        f"**User:** {os.environ.get('USER') or os.environ.get('USERNAME') or 'unknown'}",
        f"**Home:** {Path.home()}",
        f"**CPU:** {os.cpu_count() or '?'} cores",
    ]
    if mem:
        lines.append(mem)
    lines.append(
        f"**Disk:** { _fmt_bytes(disk.used) } used / { _fmt_bytes(disk.total) } "
        f"({disk.used * 100 // max(disk.total, 1)}%)"
    )
    lines.append(f"**Time:** {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')}".rstrip())
    return "\n".join(lines)


def _mem_info() -> str | None:
    # Linux
    try:
        info: dict[str, int] = {}
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith(("MemTotal:", "MemAvailable:", "MemFree:")):
                    parts = line.split()
                    info[parts[0].rstrip(":")] = int(parts[1]) * 1024
        if "MemTotal" in info:
            total = info["MemTotal"]
            avail = info.get("MemAvailable", info.get("MemFree", 0))
            used = total - avail
            return (
                f"**RAM:** { _fmt_bytes(used) } used / { _fmt_bytes(total) } "
                f"({used * 100 // total}%) — { _fmt_bytes(avail) } free"
            )
    except OSError:
        pass
    # macOS
    if platform.system() == "Darwin":
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
            return f"**RAM:** { _fmt_bytes(int(out)) } total"
        except (OSError, ValueError, subprocess.CalledProcessError):
            pass
    return None


def _fmt_bytes(n: int) -> str:
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} PB"


def lito_ram_usage() -> str:
    """Report this process RSS — proof we stay light."""
    rss = _self_rss()
    if rss is None:
        return "Could not read process memory on this OS."
    return f"Lito is using about **{ _fmt_bytes(rss) }** of RAM right now."


def _self_rss() -> int | None:
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    if platform.system() == "Darwin":
        try:
            import resource

            # ru_maxrss is bytes on macOS
            return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        except Exception:
            return None
    try:
        import resource

        # Linux: ru_maxrss is KB
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    except Exception:
        return None


def safe_calc(expr: str) -> tuple[bool, str]:
    """Evaluate a math expression safely via AST (no eval of names/calls)."""
    expr = expr.strip().rstrip("?!")
    expr = expr.replace("^", "**")
    # Allow common words
    expr = re.sub(r"\bplus\b", "+", expr, flags=re.I)
    expr = re.sub(r"\bminus\b", "-", expr, flags=re.I)
    expr = re.sub(r"\btimes\b", "*", expr, flags=re.I)
    expr = re.sub(r"\bdivided by\b", "/", expr, flags=re.I)
    expr = re.sub(r"\bx\b", "*", expr, flags=re.I)
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return False, "I couldn't parse that as math."

    allowed: tuple[type, ...] = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.UAdd,
        ast.USub,
        ast.Load,
    )
    # Python 3.7/3.8 may still emit Num
    if hasattr(ast, "Num"):
        allowed = (*allowed, ast.Num)  # type: ignore[attr-defined]

    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            return False, "Only basic arithmetic is allowed."
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            return False, "Only numbers are allowed."

    try:
        result = eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, {})  # noqa: S307
    except Exception as exc:
        return False, f"Math error: {exc}"
    if isinstance(result, float) and result == int(result):
        result = int(result)
    return True, f"= **{result}**"


def run_shell(command: str, safe: bool = True, timeout: float = 15.0) -> tuple[bool, str]:
    command = command.strip()
    if not command:
        return False, "Empty command."
    if safe and _DANGEROUS.search(command):
        return False, "Blocked potentially dangerous command. (Disable safe_shell in config to override.)"
    run_kwargs: dict = {
        "shell": True,
        "capture_output": True,
        "text": True,
        "timeout": timeout,
        "cwd": str(Path.home()),
    }
    # Avoid popping a console window on Windows GUI sessions
    if platform.system() == "Windows":
        run_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        proc = subprocess.run(command, **run_kwargs)
    except subprocess.TimeoutExpired:
        return False, f"Command timed out after {timeout}s."
    except OSError as exc:
        return False, str(exc)
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    parts = []
    if out:
        parts.append(out[:4000])
    if err:
        parts.append(f"(stderr)\n{err[:1000]}")
    if not parts:
        parts.append(f"(exit {proc.returncode}, no output)")
    ok = proc.returncode == 0
    return ok, "\n".join(parts)


def search_files(query: str, root: str | None = None, limit: int = 25) -> str:
    """Fast filename search under home (or root). Caps work for low CPU/RAM."""
    base = Path(root or Path.home()).expanduser()
    if not base.is_dir():
        return f"Not a directory: {base}"
    q = query.lower().strip()
    if not q:
        return "Give me a filename fragment to search for."
    hits: list[str] = []
    skip_dirs = {
        ".git",
        "node_modules",
        "__pycache__",
        ".cache",
        ".local",
        ".npm",
        ".venv",
        "venv",
        ".Trash",
        "Library",
        ".cargo",
        "target",
        "dist",
        "build",
    }
    # Bound walk time/work
    start = time.monotonic()
    max_secs = 2.5
    try:
        for dirpath, dirnames, filenames in os.walk(base):
            if time.monotonic() - start > max_secs or len(hits) >= limit:
                break
            dirnames[:] = [d for d in dirnames if d not in skip_dirs and not d.startswith(".")]
            for name in filenames:
                if q in name.lower():
                    hits.append(str(Path(dirpath) / name))
                    if len(hits) >= limit:
                        break
    except OSError as exc:
        return f"Search error: {exc}"
    if not hits:
        return f"No files matching `{query}` under `{base}` (searched ~{max_secs}s)."
    body = "\n".join(f"- `{h}`" for h in hits)
    more = " (capped)" if len(hits) >= limit else ""
    return f"Found **{len(hits)}** file(s){more}:\n{body}"


def web_search_open(query: str) -> tuple[bool, str]:
    url = "https://duckduckgo.com/?q=" + urllib.parse.quote_plus(query)
    return open_url(url)


def google_search_open(query: str) -> tuple[bool, str]:
    url = "https://www.google.com/search?q=" + urllib.parse.quote_plus(query)
    return open_url(url)


def fetch_url_text(url: str, max_bytes: int = 50_000) -> tuple[bool, str]:
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    req = urllib.request.Request(url, headers={"User-Agent": "Lito/0.1 (desktop-ai)"})
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            raw = resp.read(max_bytes)
            ctype = resp.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}: {exc.reason}"
    except urllib.error.URLError as exc:
        return False, f"Network error: {exc.reason}"
    except Exception as exc:
        return False, str(exc)
    text = raw.decode("utf-8", errors="replace")
    # Strip tags lightly
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", text)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return True, text[:3000] + ("…" if len(text) > 3000 else "")


def what_time() -> str:
    now = dt.datetime.now()
    return now.strftime("It's **%A, %B %d, %Y** — **%H:%M:%S**.")


def set_volume_hint() -> str:
    system = platform.system()
    if system == "Linux":
        if shutil.which("pactl"):
            return "Try: `set volume to 50` or I can run `pactl` for you."
        return "No PulseAudio `pactl` found."
    if system == "Darwin":
        return "On macOS I can run: `osascript -e 'set volume output volume N'`."
    return "Volume control varies by OS."


def set_volume(level: int) -> tuple[bool, str]:
    level = max(0, min(100, level))
    system = platform.system()
    try:
        if system == "Linux" and shutil.which("pactl"):
            subprocess.run(
                ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{level}%"],
                check=False,
                capture_output=True,
            )
            return True, f"Volume set to **{level}%**."
        if system == "Darwin":
            subprocess.run(
                ["osascript", "-e", f"set volume output volume {level}"],
                check=False,
                capture_output=True,
            )
            return True, f"Volume set to **{level}%**."
        if system == "Windows":
            return False, "Native volume control on Windows isn't wired yet — open Settings > System > Sound."
    except OSError as exc:
        return False, str(exc)
    return False, "No volume backend available on this system."


def take_note(text: str) -> str:
    note = memory.add_note(text)
    return f"Saved note #{note['id']}: {note['text']}"


def show_notes() -> str:
    notes = memory.list_notes()
    if not notes:
        return "No notes yet. Say `note buy milk` to add one."
    lines = ["**Notes:**"]
    for n in notes:
        ts = dt.datetime.fromtimestamp(n.get("ts", 0)).strftime("%m-%d %H:%M")
        lines.append(f"- #{n.get('id')} ({ts}) {n.get('text')}")
    return "\n".join(lines)


def do_remember(key: str, value: str) -> str:
    memory.remember(key, value)
    return f"I'll remember **{key}** = {value}"


def do_recall(key: str) -> str:
    val = memory.recall(key)
    if val is None:
        return f"I don't have anything stored for **{key}**."
    return f"**{key}** = {val}"


def list_apps_text(query: str = "") -> str:
    apps = registry.search(query, limit=30) if query else registry.all()[:40]
    if not apps:
        return "No matching apps found on this system."
    lines = [f"**Apps** ({len(apps)} shown):"]
    for a in apps:
        desc = f" — {a.description}" if a.description else ""
        lines.append(f"- **{a.name}** `{a.command}`{desc}")
    return "\n".join(lines)


def open_app_by_name(name: str) -> tuple[bool, str]:
    app = registry.find(name)
    if not app:
        # Try partial search
        hits = registry.search(name, limit=5)
        if not hits:
            return False, f"I couldn't find an app called **{name}**. Try `list apps`."
        if len(hits) == 1:
            return launch(hits[0])
        names = ", ".join(h.name for h in hits)
        return False, f"Did you mean: {names}?"
    return launch(app)


def screenshot() -> tuple[bool, str]:
    """Best-effort screenshot to ~/Pictures or home."""
    dest_dir = Path.home() / "Pictures"
    if not dest_dir.is_dir():
        dest_dir = Path.home()
    dest = dest_dir / f"lito-shot-{int(time.time())}.png"
    system = platform.system()
    try:
        if system == "Linux":
            for tool, args in (
                ("gnome-screenshot", ["-f", str(dest)]),
                ("scrot", [str(dest)]),
                ("import", ["-window", "root", str(dest)]),  # ImageMagick
            ):
                if shutil.which(tool.split()[0] if False else tool):
                    r = subprocess.run([tool, *args], capture_output=True, timeout=10)
                    if r.returncode == 0 and dest.exists():
                        return True, f"Screenshot saved to `{dest}`."
            return False, "No screenshot tool found (gnome-screenshot, scrot, or import)."
        if system == "Darwin":
            subprocess.run(["screencapture", "-x", str(dest)], check=False, timeout=10)
            if dest.exists():
                return True, f"Screenshot saved to `{dest}`."
            return False, "screencapture failed."
        if system == "Windows":
            return False, "Say `open snipping tool` or use Win+Shift+S on Windows."
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    return False, "Screenshot not supported here."


def cache_scan(*, include_system: bool = False) -> tuple[bool, str]:
    from . import cache as cache_mod

    return cache_mod.handle_cache_command(action="scan", include_system=include_system)


def cache_clear(
    *,
    owner: str | None = None,
    dry_run: bool = False,
    include_system: bool = False,
    unused_only: bool = True,
) -> tuple[bool, str]:
    """Clear caches for unused apps (or a named owner). Skips running apps."""
    from . import cache as cache_mod

    result = cache_mod.clear_caches(
        owner=owner,
        unused_only=unused_only,
        include_system=include_system,
        dry_run=dry_run,
    )
    head = ("**Dry run** — no files deleted.\n" if dry_run else "") + result.summary()
    body = "\n".join(result.lines[:60])
    more = f"\n…({len(result.lines) - 60} more lines)" if len(result.lines) > 60 else ""
    ok = result.failed == 0
    return ok, f"{head}\n\n{body}{more}"


def check_update() -> tuple[bool, str]:
    from . import update as update_mod

    try:
        info = update_mod.check_for_update()
        update_mod.write_last_check(info)
    except Exception as exc:  # noqa: BLE001
        return False, f"Update check failed: {exc}"
    return True, update_mod.format_update_status(info)


def install_update(*, restart: bool = False) -> tuple[bool, str]:
    from . import update as update_mod

    try:
        info = update_mod.check_for_update()
    except Exception as exc:  # noqa: BLE001
        return False, f"Update check failed: {exc}"
    if info is None:
        return True, update_mod.format_update_status(None)
    return update_mod.apply_update(info, restart=restart)


def open_release_page() -> tuple[bool, str]:
    from . import update as update_mod

    try:
        info = update_mod.check_for_update()
    except Exception:
        info = None
    if info and info.html_url:
        return open_url(info.html_url)
    repo = update_mod.DEFAULT_REPO
    return open_url(f"https://github.com/{repo}/releases/latest")


def help_text() -> str:
    return """**Lito** — tiny desktop AI (stdlib only, ~few MB RAM)

**Apps & files**
- `open firefox` / `launch code` / `start spotify`
- `open https://example.com`
- `open ~/Documents`
- `list apps` / `find app terminal`
- `find file report.pdf`

**Cache cleaner** (chat or voice)
- `scan caches` — map caches → app/system owner, mark in-use vs unused
- `clear unused caches` — delete only unused/orphaned **user** caches
- `clear unused caches dry run` — preview, delete nothing
- `clear cache for firefox` — one owner (skipped if that app is running)
- `free up cache space` / `clean app caches` — same as clear unused
- Never wipes caches for apps that are currently running
- System paths (`/var/cache/...`) only with `including system`

**Updates** (GitHub Releases)
- `check update` — compare with latest release
- `install update` — download matching OS binary & apply
- `open release page` — browser to latest release
- Auto-check on startup when `LITO_AUTO_UPDATE=1` (default)

**Tasks**
- `note pick up groceries` / `show notes`
- `remember wifi is blueorchid` / `what is wifi` / `recall wifi`
- `calc 22 * 7 + 3`
- `run echo hello` (safe shell)
- `search web python walrus operator`
- `set volume 40`
- `screenshot`
- `time` / `system info` / `how much ram`

**Chat**
- Type naturally — I'll match intent with a light rule engine (no big model in RAM).
- Optional: set `LITO_LLM_URL` to a local Ollama-style endpoint for smarter chat without loading weights into Lito.

**Tips**
- Config lives in `~/.lito/config.json`
- Data: notes, memory, history in `~/.lito/`
- Standalone exe builds: see GitHub Releases / `scripts/build_exe.py`
"""


def status_payload() -> dict[str, Any]:
    from . import __version__
    from . import update as update_mod

    rss = _self_rss()
    return {
        "name": "Lito",
        "version": __version__,
        "ram_bytes": rss,
        "ram_human": _fmt_bytes(rss) if rss else None,
        "apps_known": len(registry.all()),
        "platform": platform.system(),
        "platform_tag": update_mod.current_platform_tag(),
        "python": platform.python_version(),
        "frozen": update_mod.is_frozen(),
    }
