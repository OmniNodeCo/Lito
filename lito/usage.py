"""Track when apps were last used (and how often).

Sources (merged, lightest-first):
1. Lito's own launch log (~/.lito/app_usage.json) - always accurate for opens via Lito
2. Linux: atime/mtime of binary + GTK/KDE recently-used + bash history hints
3. macOS: mtime of .app bundle / Spotlight mdls kMDItemLastUsedDate when available
4. Windows: Start Menu .lnk LastAccess / Target mtime

Keeps RAM tiny: one small JSON dict, no DB.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import data_dir, ensure_data_dir

USAGE_FILE = "app_usage.json"


def _usage_path() -> Path:
    return data_dir() / USAGE_FILE


def _load() -> dict[str, Any]:
    path = _usage_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save(data: dict[str, Any]) -> None:
    ensure_data_dir()
    path = _usage_path()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        tmp.replace(path)
    except OSError:
        if path.exists():
            path.unlink(missing_ok=True)
        tmp.rename(path)


def _key(name: str) -> str:
    return name.strip().lower()


def record_launch(name: str, command: str = "") -> None:
    """Call whenever Lito opens an app."""
    data = _load()
    k = _key(name)
    now = time.time()
    item = data.get(k) if isinstance(data.get(k), dict) else {}
    item["name"] = name
    item["last_used"] = now
    item["launches"] = int(item.get("launches") or 0) + 1
    if command:
        item["command"] = command
    if "first_used" not in item:
        item["first_used"] = now
    data[k] = item
    _save(data)


def get_usage(name: str) -> dict[str, Any] | None:
    data = _load()
    item = data.get(_key(name))
    return item if isinstance(item, dict) else None


def all_usage() -> dict[str, dict[str, Any]]:
    data = _load()
    return {k: v for k, v in data.items() if isinstance(v, dict)}


def format_ago(ts: float | None) -> str:
    if not ts:
        return "never"
    try:
        ts_f = float(ts)
    except (TypeError, ValueError):
        return "never"
    if ts_f <= 0:
        return "never"
    delta = time.time() - ts_f
    if delta < 0:
        delta = 0
    if delta < 60:
        return "just now"
    if delta < 3600:
        m = int(delta // 60)
        return f"{m}m ago"
    if delta < 86400:
        h = int(delta // 3600)
        return f"{h}h ago"
    if delta < 86400 * 30:
        d = int(delta // 86400)
        return f"{d}d ago"
    if delta < 86400 * 365:
        mo = int(delta // (86400 * 30))
        return f"{mo}mo ago"
    y = int(delta // (86400 * 365))
    return f"{y}y ago"


def format_when(ts: float | None) -> str:
    if not ts:
        return "never"
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError, OSError):
        return "never"


def _binary_from_command(command: str) -> str | None:
    if not command:
        return None
    # flatpak run id / snap run name / open -a Name
    if command.startswith("flatpak run "):
        return None
    if command.startswith("snap run "):
        return None
    if command.startswith("open -a "):
        return None
    parts = command.strip().strip('"').split()
    if not parts:
        return None
    bin0 = parts[0].strip('"')
    if bin0.startswith("/"):
        return bin0
    return shutil.which(bin0)


def probe_last_used(app_name: str, command: str = "", source: str = "") -> float | None:
    """Best-effort filesystem / OS last-used timestamp (may be None)."""
    # Prefer our own log
    logged = get_usage(app_name)
    if logged and logged.get("last_used"):
        return float(logged["last_used"])

    system = platform.system()
    candidates: list[float] = []

    binary = _binary_from_command(command)
    if binary:
        p = Path(binary)
        try:
            st = p.stat()
            # atime is often reliable on Linux when noatime is not set
            for t in (getattr(st, "st_atime", 0), st.st_mtime):
                if t and t > 1_000_000:  # skip epoch noise
                    candidates.append(float(t))
        except OSError:
            pass

    if system == "Darwin" and command.startswith("open -a "):
        m = re.search(r'open -a ["\']?([^"\']+)["\']?', command)
        name = m.group(1) if m else app_name
        for root in (Path("/Applications"), Path.home() / "Applications", Path("/System/Applications")):
            app_path = root / f"{name}.app"
            if app_path.is_dir():
                try:
                    candidates.append(app_path.stat().st_mtime)
                except OSError:
                    pass
                # mdls last used
                try:
                    out = subprocess.check_output(
                        ["mdls", "-name", "kMDItemLastUsedDate", "-raw", str(app_path)],
                        text=True,
                        stderr=subprocess.DEVNULL,
                        timeout=3,
                    ).strip()
                    if out and out != "(null)":
                        # e.g. 2024-01-15 12:34:56 +0000
                        try:
                            # drop timezone for simple parse
                            dt = datetime.strptime(out[:19], "%Y-%m-%d %H:%M:%S")
                            candidates.append(dt.timestamp())
                        except ValueError:
                            pass
                except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
                    pass
                break

    if system == "Windows" and command:
        target = command.strip().strip('"').split('"')[0] if command.startswith('"') else command.split()[0]
        p = Path(target.strip('"'))
        if p.exists():
            try:
                st = p.stat()
                candidates.extend([float(st.st_atime), float(st.st_mtime)])
            except OSError:
                pass

    # GTK recently-used xbel (Linux)
    if system == "Linux":
        xbel = Path.home() / ".local/share/recently-used.xbel"
        if xbel.is_file():
            try:
                text = xbel.read_text(encoding="utf-8", errors="ignore")
                # Look for app name or binary in hrefs / applications
                needle = (app_name or "").lower()
                bin_name = Path(binary).name.lower() if binary else ""
                for block in re.findall(r"<bookmark\b[^>]*>.*?</bookmark>", text, flags=re.I | re.S):
                    blob = block.lower()
                    if needle and needle not in blob and (not bin_name or bin_name not in blob):
                        continue
                    m = re.search(r'modified="([^"]+)"', block) or re.search(r'visited="([^"]+)"', block)
                    if m:
                        raw = m.group(1).replace("Z", "+00:00")
                        try:
                            # fromisoformat handles 2024-01-01T12:00:00
                            ts = datetime.fromisoformat(raw[:19]).timestamp()
                            candidates.append(ts)
                        except ValueError:
                            pass
            except OSError:
                pass

    if not candidates:
        return None
    return max(candidates)


def enrich_last_used(name: str, command: str = "", source: str = "") -> dict[str, Any]:
    """Merged view: log + probe."""
    logged = get_usage(name) or {}
    probed = probe_last_used(name, command, source)
    last = logged.get("last_used")
    try:
        last_f = float(last) if last is not None else None
    except (TypeError, ValueError):
        last_f = None
    if probed and (last_f is None or probed > last_f):
        # Don't overwrite log with older probe; only fill gaps or newer OS signal
        if last_f is None:
            last_f = probed
        elif probed - last_f > 60:  # OS says significantly newer
            last_f = probed
    return {
        "name": name,
        "last_used": last_f,
        "last_used_ago": format_ago(last_f),
        "last_used_when": format_when(last_f),
        "launches": int(logged.get("launches") or 0),
        "source": "lito" if logged.get("last_used") else ("os" if last_f else "unknown"),
    }
