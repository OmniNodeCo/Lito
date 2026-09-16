"""Discover app/system caches and clear ones that are safe to drop.

Strategy
--------
1. Scan well-known cache roots (user + optional system).
2. Map each cache directory to an **owner** (app name, package manager,
   or "system").
3. Detect which owners are **currently running** (process list).
4. Mark caches as:
   - **active**    - owning app is running -> skip (in use)
   - **recent**    - not running, but last used within the idle window -> keep
   - **unused**    - not running and not used recently (or never tracked) -> safe
   - **orphaned**  - cache exists but owner binary not installed -> safe
   - **protected** - critical system paths we never delete
5. Last-used comes from Lito's launch log + OS probes (see usage.py).
6. Delete only unused/orphaned (or a named target) under the user home
   by default. System paths require an explicit flag.

Keeps RAM low: streams directory sizes, no heavy indexing.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

# ---------------------------------------------------------------------------
# Owner catalogue: name -> process match tokens + known relative cache paths
# Paths may contain ~ and are expanded later. Globs are single-level (*) only.
# ---------------------------------------------------------------------------

# (owner_id, display_name, process_tokens, path_globs, kind)
# kind: app | pm | system | browser | dev
_OWNER_SPECS: tuple[tuple[str, str, tuple[str, ...], tuple[str, ...], str], ...] = (
    # Browsers
    (
        "firefox",
        "Firefox",
        ("firefox", "firefox-bin", "firefox-esr"),
        (
            "~/.cache/mozilla",
            "~/.mozilla/firefox/*/cache2",
            "~/.mozilla/firefox/*/startupCache",
            "~/Library/Caches/Firefox",
            "~/Library/Caches/Mozilla",
        ),
        "browser",
    ),
    (
        "chrome",
        "Google Chrome",
        ("chrome", "google-chrome", "Google Chrome"),
        (
            "~/.cache/google-chrome",
            "~/.config/google-chrome/Default/Service Worker/CacheStorage",
            "~/.config/google-chrome/Default/Code Cache",
            "~/.config/google-chrome/ShaderCache",
            "~/Library/Caches/Google/Chrome",
            "~/AppData/Local/Google/Chrome/User Data/Default/Cache",
            "~/AppData/Local/Google/Chrome/User Data/ShaderCache",
        ),
        "browser",
    ),
    (
        "chromium",
        "Chromium",
        ("chromium", "chromium-browser"),
        (
            "~/.cache/chromium",
            "~/.config/chromium/Default/Code Cache",
            "~/.config/chromium/ShaderCache",
            "~/Library/Caches/Chromium",
        ),
        "browser",
    ),
    (
        "brave",
        "Brave",
        ("brave", "brave-browser"),
        ("~/.cache/BraveSoftware", "~/Library/Caches/BraveSoftware"),
        "browser",
    ),
    (
        "edge",
        "Microsoft Edge",
        ("msedge", "microsoft-edge", "Microsoft Edge"),
        (
            "~/.cache/microsoft-edge",
            "~/Library/Caches/com.microsoft.edgemac",
            "~/AppData/Local/Microsoft/Edge/User Data/Default/Cache",
        ),
        "browser",
    ),
    (
        "safari",
        "Safari",
        ("Safari",),
        ("~/Library/Caches/com.apple.Safari",),
        "browser",
    ),
    # Editors / IDEs
    (
        "code",
        "VS Code",
        ("code", "Code"),
        (
            "~/.config/Code/Cache",
            "~/.config/Code/CachedData",
            "~/.config/Code/CachedExtensions",
            "~/.config/Code/Code Cache",
            "~/.config/Code/GPUCache",
            "~/.config/Code/logs",
            "~/Library/Application Support/Code/Cache",
            "~/Library/Application Support/Code/CachedData",
            "~/Library/Caches/com.microsoft.VSCode",
            "~/AppData/Roaming/Code/Cache",
            "~/AppData/Roaming/Code/CachedData",
        ),
        "dev",
    ),
    (
        "cursor",
        "Cursor",
        ("cursor", "Cursor"),
        (
            "~/.config/Cursor/Cache",
            "~/.config/Cursor/CachedData",
            "~/.config/Cursor/GPUCache",
            "~/Library/Application Support/Cursor/Cache",
            "~/Library/Application Support/Cursor/CachedData",
        ),
        "dev",
    ),
    # Chat / media
    (
        "slack",
        "Slack",
        ("slack", "Slack"),
        (
            "~/.config/Slack/Cache",
            "~/.config/Slack/Code Cache",
            "~/.config/Slack/GPUCache",
            "~/Library/Application Support/Slack/Cache",
            "~/Library/Caches/com.tinyspeck.slackmacgap",
        ),
        "app",
    ),
    (
        "discord",
        "Discord",
        ("discord", "Discord"),
        (
            "~/.config/discord/Cache",
            "~/.config/discord/Code Cache",
            "~/.config/discord/GPUCache",
            "~/Library/Application Support/discord/Cache",
            "~/Library/Caches/com.hnc.Discord",
        ),
        "app",
    ),
    (
        "spotify",
        "Spotify",
        ("spotify", "Spotify"),
        (
            "~/.cache/spotify",
            "~/Library/Caches/com.spotify.client",
            "~/AppData/Local/Spotify/Storage",
        ),
        "app",
    ),
    (
        "telegram",
        "Telegram",
        ("telegram-desktop", "Telegram"),
        ("~/.cache/telegram-desktop", "~/Library/Caches/ru.keepcoder.Telegram"),
        "app",
    ),
    (
        "steam",
        "Steam",
        ("steam", "steamwebhelper"),
        ("~/.steam/steam/appcache", "~/.local/share/Steam/appcache", "~/Library/Caches/com.valvesoftware.steam"),
        "app",
    ),
    (
        "vlc",
        "VLC",
        ("vlc",),
        ("~/.cache/vlc", "~/Library/Caches/org.videolan.vlc"),
        "app",
    ),
    (
        "obsidian",
        "Obsidian",
        ("obsidian", "Obsidian"),
        (
            "~/.config/obsidian/Cache",
            "~/.config/obsidian/Code Cache",
            "~/.config/obsidian/GPUCache",
            "~/Library/Application Support/obsidian/Cache",
        ),
        "app",
    ),
    # Package managers / dev toolchains (system-ish, user-scoped)
    (
        "npm",
        "npm",
        ("npm", "node"),
        ("~/.npm/_cacache", "~/.npm/_logs"),
        "pm",
    ),
    (
        "yarn",
        "Yarn",
        ("yarn",),
        ("~/.cache/yarn", "~/.yarn/cache"),
        "pm",
    ),
    (
        "pnpm",
        "pnpm",
        ("pnpm",),
        ("~/.local/share/pnpm/store", "~/.cache/pnpm"),
        "pm",
    ),
    (
        "pip",
        "pip",
        ("pip", "pip3"),
        ("~/.cache/pip",),
        "pm",
    ),
    (
        "cargo",
        "Cargo",
        ("cargo", "rustc"),
        ("~/.cargo/registry/cache", "~/.cargo/git/db"),
        "pm",
    ),
    (
        "go",
        "Go modules",
        ("go",),
        ("~/go/pkg/mod/cache", "~/.cache/go-build"),
        "pm",
    ),
    (
        "composer",
        "Composer",
        ("composer",),
        ("~/.composer/cache", "~/.cache/composer"),
        "pm",
    ),
    (
        "gradle",
        "Gradle",
        ("gradle", "java"),
        ("~/.gradle/caches",),
        "pm",
    ),
    (
        "thumbnails",
        "Thumbnail cache",
        (),  # no process - always "unused" if present
        ("~/.cache/thumbnails", "~/.thumbnails"),
        "system",
    ),
    (
        "fontconfig",
        "Fontconfig",
        (),
        ("~/.cache/fontconfig",),
        "system",
    ),
    (
        "mesa",
        "Mesa shader cache",
        (),
        ("~/.cache/mesa_shader_cache", "~/.cache/mesa_shader_cache_db"),
        "system",
    ),
    (
        "tracker",
        "GNOME Tracker",
        ("tracker-miner", "tracker3"),
        ("~/.cache/tracker3", "~/.cache/tracker"),
        "system",
    ),
    (
        "flatpak",
        "Flatpak",
        ("flatpak",),
        ("~/.cache/flatpak", "~/.var/app/*/cache"),
        "pm",
    ),
    (
        "snap",
        "Snap",
        ("snapd",),
        ("~/snap/*/common/.cache",),
        "pm",
    ),
    (
        "apt",
        "APT package cache",
        (),
        ("/var/cache/apt/archives",),
        "system",
    ),
    (
        "pacman",
        "Pacman package cache",
        (),
        ("/var/cache/pacman/pkg",),
        "system",
    ),
    (
        "dnf",
        "DNF package cache",
        (),
        ("/var/cache/dnf",),
        "system",
    ),
    (
        "brew",
        "Homebrew",
        (),
        ("~/Library/Caches/Homebrew",),
        "pm",
    ),
    (
        "lito",
        "Lito",
        ("lito",),  # don't clear our own live process data carelessly
        (),  # never auto-map; reserved
        "app",
    ),
)

# Directory name -> owner_id for generic ~/.cache/<name> discovery
_CACHE_DIR_ALIASES: dict[str, str] = {
    "mozilla": "firefox",
    "google-chrome": "chrome",
    "chromium": "chromium",
    "BraveSoftware": "brave",
    "microsoft-edge": "edge",
    "spotify": "spotify",
    "pip": "pip",
    "yarn": "yarn",
    "pnpm": "pnpm",
    "thumbnails": "thumbnails",
    "fontconfig": "fontconfig",
    "mesa_shader_cache": "mesa",
    "mesa_shader_cache_db": "mesa",
    "tracker": "tracker",
    "tracker3": "tracker",
    "flatpak": "flatpak",
    "telegram-desktop": "telegram",
    "vlc": "vlc",
    "composer": "composer",
    "go-build": "go",
    "Homebrew": "brew",
}

# Never delete these (absolute resolved prefixes)
_PROTECTED_NAMES = frozenset(
    {
        "/",
        "/home",
        "/Users",
        "/var",
        "/var/cache",
        "/usr",
        "/etc",
        "/bin",
        "/sbin",
        "/boot",
        "/dev",
        "/proc",
        "/sys",
        "/root",
    }
)

_PROTECTED_SUFFIXES = frozenset(
    {
        ".ssh",
        ".gnupg",
        ".lito",  # our own data
        "Documents",
        "Desktop",
        "Downloads",
        "Pictures",
        "Music",
        "Movies",
        "Videos",
    }
)


# Default: apps used within this many days keep their cache on "clear unused"
DEFAULT_IDLE_DAYS = 7.0


@dataclass
class CacheEntry:
    path: Path
    owner_id: str
    owner_name: str
    kind: str  # app | pm | system | browser | dev
    size_bytes: int = 0
    running: bool = False
    owner_installed: bool = True
    status: str = "unused"  # active | recent | unused | orphaned | protected | empty
    system_scope: bool = False  # outside user home
    notes: str = ""
    last_used: float | None = None  # epoch seconds
    last_used_ago: str = ""
    idle_days: float | None = None  # days since last_used (None if unknown)

    def as_dict(self) -> dict:
        return {
            "path": str(self.path),
            "owner_id": self.owner_id,
            "owner_name": self.owner_name,
            "kind": self.kind,
            "size_bytes": self.size_bytes,
            "size_human": _fmt(self.size_bytes),
            "running": self.running,
            "owner_installed": self.owner_installed,
            "status": self.status,
            "system_scope": self.system_scope,
            "notes": self.notes,
            "last_used": self.last_used,
            "last_used_ago": self.last_used_ago,
            "idle_days": self.idle_days,
        }


@dataclass
class CleanResult:
    scanned: int = 0
    cleared: int = 0
    skipped: int = 0
    failed: int = 0
    freed_bytes: int = 0
    lines: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Scanned **{self.scanned}** - cleared **{self.cleared}** - "
            f"skipped **{self.skipped}** - failed **{self.failed}** - "
            f"freed **{_fmt(self.freed_bytes)}**"
        )


def _fmt(n: int | float) -> str:
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} PB"


def _home() -> Path:
    return Path.home()


def _expand(p: str) -> Path:
    """Expand leading ~ against Lito's home (patchable in tests)."""
    if p.startswith("~/") or p == "~":
        rest = p[2:] if p.startswith("~/") else ""
        base = _home()
        return (base / rest).resolve(strict=False) if rest else base.resolve(strict=False)
    return Path(p).expanduser().resolve(strict=False)


def _is_under(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _is_protected(path: Path) -> bool:
    try:
        resolved = path.resolve(strict=False)
    except OSError:
        resolved = path
    s = str(resolved)
    if s in _PROTECTED_NAMES or resolved.as_posix() in _PROTECTED_NAMES:
        return True
    # Never wipe entire home or entire Library
    home = _home().resolve(strict=False)
    if resolved == home:
        return True
    if resolved.name in _PROTECTED_SUFFIXES and _is_under(resolved, home):
        # allow ~/.cache/thumbnails etc.; block bare ~/.ssh
        if resolved.parent == home or resolved.parent.name in {".config", "Library"}:
            if resolved.name in {".ssh", ".gnupg", ".lito", "Documents", "Desktop"}:
                return True
    # Must look like a cache-ish path
    parts_l = {p.lower() for p in resolved.parts}
    cache_markers = {
        "cache",
        "caches",
        "cacheddata",
        "codecache",
        "gpucache",
        "shadercache",
        "_cacache",
        "appcache",
        "thumbnails",
        ".thumbnails",
        "pip",
        "archives",
    }
    # Allow known package-manager cache roots even without "cache" in name
    known_ok = any(
        x in s
        for x in (
            "/.npm/",
            "/.cargo/registry",
            "/.cargo/git",
            "/.gradle/caches",
            "/go/pkg/mod/cache",
            "/.steam/",
            "appcache",
            "ShaderCache",
            "Code Cache",
            "CachedData",
            "GPUCache",
        )
    )
    if known_ok:
        return False
    if not (parts_l & cache_markers) and "cache" not in s.lower():
        return True
    return False


def _dir_size(path: Path, deadline: float, max_files: int = 50_000) -> int:
    """Bounded size walk - stops early to stay light."""
    total = 0
    n = 0
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    try:
        for root, dirs, files in os.walk(path, followlinks=False):
            if time.monotonic() > deadline or n >= max_files:
                break
            # skip huge nested VCS junk if any
            dirs[:] = [d for d in dirs if d not in {".git", "node_modules"}]
            for name in files:
                n += 1
                if n >= max_files or time.monotonic() > deadline:
                    break
                fp = Path(root) / name
                try:
                    if fp.is_symlink():
                        continue
                    st = fp.lstat()
                except OSError:
                    continue
                total += st.st_size
    except OSError:
        pass
    return total


def _expand_globs(pattern: str) -> list[Path]:
    """Expand ~ (via _home) and a single * path segment (no recursive **)."""
    if pattern.startswith("~/") or pattern == "~":
        rest = pattern[2:] if pattern.startswith("~/") else ""
        base_path = _home() / rest if rest else _home()
    elif pattern.startswith("~\\"):
        rest = pattern[2:].lstrip("\\")
        base_path = _home() / rest if rest else _home()
    else:
        base_path = Path(os.path.expanduser(pattern))

    raw = str(base_path)
    if "*" not in raw:
        return [base_path] if base_path.exists() else []

    # Progressive walk that is OS-agnostic (works with drive letters on Windows)
    parts = list(base_path.parts)
    if not parts:
        return []
    matches: list[Path] = [Path(parts[0])]
    # On POSIX absolute paths parts[0] is '/'; on Windows it's 'C:\\'
    for part in parts[1:]:
        next_matches: list[Path] = []
        for base in matches:
            if not base.exists():
                continue
            if part == "*":
                try:
                    next_matches.extend(c for c in base.iterdir() if c.is_dir())
                except OSError:
                    pass
            else:
                cand = base / part
                if cand.exists():
                    next_matches.append(cand)
        matches = next_matches
        if not matches:
            break
    return [m for m in matches if m.exists()]


def _running_processes() -> set[str]:
    """Lowercased process name set (best-effort, cross-platform)."""
    names: set[str] = set()
    system = platform.system()
    try:
        if system == "Windows":
            try:
                out = subprocess.check_output(
                    ["tasklist", "/FO", "CSV", "/NH"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=8,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except TypeError:
                # creationflags unsupported
                out = subprocess.check_output(
                    ["tasklist", "/FO", "CSV", "/NH"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=8,
                )
            for line in out.splitlines():
                # "name.exe","pid",...
                m = re.match(r'"([^"]+)"', line.strip())
                if m:
                    name = m.group(1).lower()
                    if name.endswith(".exe"):
                        name = name[:-4]
                    names.add(name)
        else:
            # ps is widely available; fall back to /proc
            try:
                out = subprocess.check_output(
                    ["ps", "-A", "-o", "comm="],
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                )
                for line in out.splitlines():
                    n = line.strip()
                    if n:
                        names.add(Path(n).name.lower())
            except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
                proc = Path("/proc")
                if proc.is_dir():
                    for ent in proc.iterdir():
                        if not ent.name.isdigit():
                            continue
                        try:
                            comm = (ent / "comm").read_text(encoding="utf-8").strip()
                            if comm:
                                names.add(comm.lower())
                            # also cmdline first token
                            cmd = (ent / "cmdline").read_bytes().split(b"\x00")[0].decode("utf-8", "ignore")
                            if cmd:
                                names.add(Path(cmd).name.lower())
                        except OSError:
                            continue
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass
    return names


def _owner_running(tokens: tuple[str, ...], procs: set[str]) -> bool:
    if not tokens:
        return False
    for t in tokens:
        tl = t.lower()
        if tl in procs:
            return True
        # substring match for "Google Chrome" etc.
        for p in procs:
            if tl in p or p in tl:
                return True
    return False


def _owner_installed(tokens: tuple[str, ...]) -> bool:
    if not tokens:
        return True  # system caches without a binary
    for t in tokens:
        # skip display names with spaces for which()
        if " " in t:
            continue
        if shutil.which(t):
            return True
        # macOS app bundle heuristic
        mac = Path(f"/Applications/{t}.app")
        if mac.exists():
            return True
        mac2 = Path(f"/Applications/{t}")
        if mac2.exists():
            return True
    return False


def _owners_index() -> dict[str, tuple[str, tuple[str, ...], str]]:
    """owner_id -> (display, tokens, kind)"""
    return {oid: (name, toks, kind) for oid, name, toks, _paths, kind in _OWNER_SPECS}


def _parse_idle_days(value: float | int | str | None, default: float = DEFAULT_IDLE_DAYS) -> float:
    """Parse idle window. Accepts days (float), or strings like '7d', '48h', '2w'."""
    if value is None:
        return float(default)
    if isinstance(value, (int, float)):
        return max(0.0, float(value))
    s = str(value).strip().lower()
    if not s:
        return float(default)
    m = re.match(r"^([0-9]*\.?[0-9]+)\s*([a-z]*)$", s)
    if not m:
        try:
            return max(0.0, float(s))
        except ValueError:
            return float(default)
    n = float(m.group(1))
    unit = m.group(2) or "d"
    if unit in {"", "d", "day", "days"}:
        return max(0.0, n)
    if unit in {"h", "hr", "hour", "hours"}:
        return max(0.0, n / 24.0)
    if unit in {"w", "wk", "week", "weeks"}:
        return max(0.0, n * 7.0)
    if unit in {"m", "mo", "month", "months"}:
        return max(0.0, n * 30.0)
    return max(0.0, n)


def _last_used_for_owner(owner_id: str, owner_name: str, tokens: tuple[str, ...]) -> float | None:
    """Last-used for cache decisions: **Lito launch log only**.

    We deliberately skip OS binary atime/mtime here - those flip "recent" for
    any installed tool (pip, node, …) and would block clearing their caches.
    Launch tracking via open/launch is the reliable signal.
    """
    try:
        from . import usage as usage_mod
    except Exception:
        return None
    candidates: list[float] = []
    oid = (owner_id or "").lower()
    oname = (owner_name or "").lower()
    token_l = {t.lower() for t in tokens if t}

    def _take(item: dict) -> None:
        ts = item.get("last_used")
        if ts:
            try:
                candidates.append(float(ts))
            except (TypeError, ValueError):
                pass

    # Direct keys
    for n in (owner_id, owner_name, *tokens):
        if not n:
            continue
        try:
            item = usage_mod.get_usage(n)
            if item:
                _take(item)
        except Exception:
            continue

    # Fuzzy scan of the whole log (name / command contains owner)
    try:
        for key, item in usage_mod.all_usage().items():
            blob = f"{key} {item.get('name', '')} {item.get('command', '')}".lower()
            if oid and oid in blob:
                _take(item)
                continue
            if oname and len(oname) >= 3 and oname in blob:
                _take(item)
                continue
            if any(t and len(t) >= 3 and t in blob for t in token_l):
                _take(item)
    except Exception:
        pass
    return max(candidates) if candidates else None


def _classify_with_usage(
    *,
    running: bool,
    installed: bool,
    protected: bool,
    size: int,
    last_used: float | None,
    idle_days: float,
    kind: str,
) -> tuple[str, str, float | None]:
    """Return (status, notes, idle_days_value)."""
    idle_val: float | None = None
    if last_used:
        idle_val = max(0.0, (time.time() - float(last_used)) / 86400.0)

    if protected:
        return "protected", "protected path", idle_val
    if size == 0:
        return "empty", "", idle_val
    if running:
        return "active", "owner process is running", idle_val
    # package managers / system caches: no "recent" keep unless we have a strong signal
    # but still honor last_used when present
    if last_used is not None and idle_val is not None and idle_val < idle_days:
        ago = ""
        try:
            from . import usage as usage_mod
            ago = usage_mod.format_ago(last_used)
        except Exception:
            ago = f"{idle_val:.1f}d ago"
        return (
            "recent",
            f"used {ago} (within {idle_days:g}-day keep window)",
            idle_val,
        )
    if not installed:
        return "orphaned", "owner app not installed", idle_val
    if last_used is None:
        # Unknown usage: still eligible as unused, but note it
        return "unused", "no last-used signal (eligible)", idle_val
    return (
        "unused",
        f"idle {idle_val:.1f}d (>{idle_days:g}d keep window)",
        idle_val,
    )



def _iter_known_paths() -> Iterator[tuple[str, Path]]:
    for oid, _name, _toks, paths, _kind in _OWNER_SPECS:
        for pat in paths:
            for p in _expand_globs(pat):
                yield oid, p


def _discover_generic_user_cache() -> Iterator[tuple[str, Path]]:
    """Pick up ~/.cache/* and macOS ~/Library/Caches/* not already listed."""
    roots = [
        _home() / ".cache",
        _home() / "Library" / "Caches",
        _home() / "AppData" / "Local",
    ]
    for root in roots:
        if not root.is_dir():
            continue
        try:
            children = list(root.iterdir())
        except OSError:
            continue
        for child in children:
            if not child.is_dir() or child.is_symlink():
                continue
            name = child.name
            # AppData/Local is huge - only known cache-like subdirs
            if root.name == "Local":
                low = name.lower()
                if not any(k in low for k in ("cache", "temp", "chrome", "edge", "spotify", "code")):
                    continue
            oid = _CACHE_DIR_ALIASES.get(name) or _CACHE_DIR_ALIASES.get(name.lower())
            if not oid:
                # heuristic owner id from folder name
                oid = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "unknown"
            yield oid, child


def scan_caches(
    *,
    include_system: bool = False,
    max_secs: float = 4.0,
    idle_days: float | int | str | None = None,
) -> list[CacheEntry]:
    """Return catalogued cache entries with sizes, running status, and last-used.

    idle_days: keep window for "recent" status (default 7 days). Apps used more
    recently than this stay marked recent and are skipped by clear unused.
    """
    home = _home().resolve(strict=False)
    procs = _running_processes()
    index = _owners_index()
    deadline = time.monotonic() + max_secs
    keep_days = _parse_idle_days(idle_days, DEFAULT_IDLE_DAYS)

    # Cache last-used lookups per owner_id for this scan
    last_used_cache: dict[str, float | None] = {}

    seen: set[str] = set()
    entries: list[CacheEntry] = []

    def add(owner_id: str, path: Path) -> None:
        if time.monotonic() > deadline:
            return
        try:
            resolved = path.resolve(strict=False)
        except OSError:
            resolved = path
        key = str(resolved)
        if key in seen:
            return
        if not resolved.exists():
            return
        seen.add(key)

        system_scope = not _is_under(resolved, home)
        if system_scope and not include_system:
            return

        meta = index.get(owner_id)
        if meta:
            owner_name, tokens, kind = meta
        else:
            owner_name, tokens, kind = owner_id, (), "app"

        running = _owner_running(tokens, procs)
        installed = _owner_installed(tokens) if tokens else True
        protected = _is_protected(resolved)

        size = _dir_size(resolved, deadline=min(deadline, time.monotonic() + 1.5))

        if owner_id not in last_used_cache:
            last_used_cache[owner_id] = _last_used_for_owner(owner_id, owner_name, tokens)
        last_used = last_used_cache[owner_id]

        status, notes, idle_val = _classify_with_usage(
            running=running,
            installed=installed if tokens else True,
            protected=protected,
            size=size,
            last_used=last_used,
            idle_days=keep_days,
            kind=kind,
        )

        ago = ""
        if last_used:
            try:
                from . import usage as usage_mod
                ago = usage_mod.format_ago(last_used)
            except Exception:
                ago = ""

        entries.append(
            CacheEntry(
                path=resolved,
                owner_id=owner_id,
                owner_name=owner_name,
                kind=kind,
                size_bytes=size,
                running=running,
                owner_installed=installed,
                status=status,
                system_scope=system_scope,
                notes=notes,
                last_used=last_used,
                last_used_ago=ago,
                idle_days=idle_val,
            )
        )

    for oid, p in _iter_known_paths():
        add(oid, p)
    for oid, p in _discover_generic_user_cache():
        add(oid, p)

    # Sort: reclaimable (unused/orphaned) by size, then recent, then active
    rank = {"orphaned": 0, "unused": 1, "empty": 2, "recent": 3, "active": 4, "protected": 5}
    entries.sort(
        key=lambda e: (
            rank.get(e.status, 9),
            -e.size_bytes,
            e.owner_name.lower(),
            str(e.path),
        )
    )
    return entries


def format_scan(
    entries: list[CacheEntry],
    *,
    limit: int = 40,
    idle_days: float | int | str | None = None,
) -> str:
    keep_days = _parse_idle_days(idle_days, DEFAULT_IDLE_DAYS)
    if not entries:
        return (
            "**Cache scan** - 0 locations found (or none readable).\n"
            "Try again after using some apps, or say `clear unused caches dry run`."
        )
    total = sum(e.size_bytes for e in entries)
    unused = [e for e in entries if e.status in {"unused", "orphaned"}]
    recent = [e for e in entries if e.status == "recent"]
    active = [e for e in entries if e.status == "active"]
    freed_potential = sum(e.size_bytes for e in unused)
    kept_recent = sum(e.size_bytes for e in recent)

    lines = [
        f"**Cache scan** - {len(entries)} locations - **{_fmt(total)}** total",
        f"Reclaimable (not used in **{keep_days:g}+ days** / orphaned): "
        f"**{_fmt(freed_potential)}** ({len(unused)} cache(s))",
        f"Kept recent (used within {keep_days:g}d): **{_fmt(kept_recent)}** "
        f"({len(recent)}) - In use now: **{len(active)}**",
        "",
    ]
    for e in entries[:limit]:
        flag = {
            "active": "[*] in use",
            "recent": "[~] recent",
            "unused": "[ ] idle",
            "orphaned": "[?] orphaned",
            "protected": "[!] protected",
            "empty": "- empty",
        }.get(e.status, e.status)
        scope = " [system]" if e.system_scope else ""
        used = ""
        if e.last_used_ago:
            used = f" · last used {e.last_used_ago}"
        elif e.status in {"unused", "orphaned"}:
            used = " · last used unknown"
        note = f" — {e.notes}" if e.notes and e.status in {"recent", "unused"} else ""
        lines.append(
            f"- {flag} - **{e.owner_name}** ({e.kind}) - {_fmt(e.size_bytes)}"
            f"{used} - `{e.path}`{scope}{note}"
        )
    if len(entries) > limit:
        lines.append(f"\n...and {len(entries) - limit} more.")
    lines.append(
        f"\nSay **`clear unused caches`** to delete idle/orphaned user caches "
        f"(keeps apps used in the last **{keep_days:g} days**). "
        f"Try **`clear caches older than 30 days`**, **`clear cache for firefox`**, "
        f"or add **`dry run`** to preview."
    )
    return "\n".join(lines)


def _safe_delete(path: Path) -> tuple[bool, str, int]:
    """Delete a cache path. Returns (ok, message, freed_bytes)."""
    if _is_protected(path):
        return False, f"protected: `{path}`", 0
    if not path.exists():
        return False, f"missing: `{path}`", 0
    # Final guard: path must still look like cache
    if _is_protected(path):
        return False, f"blocked: `{path}`", 0

    size = _dir_size(path, deadline=time.monotonic() + 2.0)
    try:
        if path.is_symlink() or path.is_file():
            _unlink_retry(path)
        elif path.is_dir():
            shutil.rmtree(path, onerror=_on_rm_error)
        else:
            return False, f"not a file/dir: `{path}`", 0
    except OSError as exc:
        return False, f"{path}: {exc}", 0
    return True, f"cleared `{path}`", size


def _unlink_retry(path: Path, attempts: int = 5) -> None:
    last: Exception | None = None
    for i in range(attempts):
        try:
            path.unlink(missing_ok=True)
            return
        except OSError as exc:
            last = exc
            time.sleep(0.05 * (i + 1))
    if last:
        raise last


def _on_rm_error(func, path, exc_info) -> None:  # noqa: ANN001
    """Windows-friendly rmtree handler: clear read-only bit and retry."""
    try:
        os.chmod(path, 0o700)
        func(path)
    except OSError:
        pass


def clear_caches(
    *,
    owner: str | None = None,
    unused_only: bool = True,
    include_system: bool = False,
    dry_run: bool = False,
    min_size: int = 0,
    idle_days: float | int | str | None = None,
    include_recent: bool = False,
) -> CleanResult:
    """Clear matching caches.

    unused_only=True (default): only status unused/orphaned/empty — never
    active, protected, or **recent** (used within idle_days).
    include_recent=True: also clear "recent" caches (still never active).
    owner: filter by owner_id or owner_name (case-insensitive substring).
    idle_days: keep window (default 7). Only caches idle longer than this
    (or with no last-used signal / orphaned) are cleared when unused_only.
    """
    keep_days = _parse_idle_days(idle_days, DEFAULT_IDLE_DAYS)
    entries = scan_caches(include_system=include_system, idle_days=keep_days)
    result = CleanResult(scanned=len(entries))
    result.lines.append(
        f"Keep window: apps used within **{keep_days:g} days** are skipped"
        + (" (include_recent overrides)" if include_recent else "")
        + "."
    )

    owner_q = (owner or "").strip().lower()
    clearable = {"unused", "orphaned", "empty"}
    if include_recent or not unused_only:
        clearable = clearable | {"recent"}
    # Explicit owner target: user asked for that app — allow recent, never active
    if owner_q:
        clearable = clearable | {"recent"}

    selected: list[CacheEntry] = []
    for e in entries:
        if e.size_bytes < min_size:
            continue
        if owner_q:
            blob = f"{e.owner_id} {e.owner_name}".lower()
            if owner_q not in blob and owner_q not in str(e.path).lower():
                continue
        if e.status == "protected":
            result.skipped += 1
            result.lines.append(f"skip [!] `{e.path}`")
            continue
        if e.status == "active":
            result.skipped += 1
            result.lines.append(
                f"skip [*] **{e.owner_name}** in use - `{e.path}`"
            )
            continue
        if e.status == "recent" and e.status not in clearable:
            result.skipped += 1
            ago = e.last_used_ago or "recently"
            result.lines.append(
                f"skip [~] **{e.owner_name}** used {ago} "
                f"(within {keep_days:g}d) - `{e.path}`"
            )
            continue
        if unused_only and e.status not in clearable:
            result.skipped += 1
            continue
        if e.system_scope and not include_system:
            result.skipped += 1
            result.lines.append(f"skip [system] `{e.path}` (pass include_system)")
            continue
        selected.append(e)

    if not selected:
        result.lines.insert(
            0,
            "Nothing to clear"
            + (f" for **{owner}**." if owner else " - no idle/unused caches matched.")
            + f" (keep window {keep_days:g}d)",
        )
        return result

    for e in selected:
        used = f" · last used {e.last_used_ago}" if e.last_used_ago else " · last used unknown"
        if dry_run:
            result.cleared += 1  # would-clear count
            result.freed_bytes += e.size_bytes
            result.lines.append(
                f"would clear - **{e.owner_name}** - {_fmt(e.size_bytes)}{used} - `{e.path}`"
            )
            continue
        ok, msg, freed = _safe_delete(e.path)
        if ok:
            result.cleared += 1
            result.freed_bytes += freed
            result.lines.append(
                f"cleared - **{e.owner_name}** - {_fmt(freed)}{used} - `{e.path}`"
            )
        else:
            result.failed += 1
            result.lines.append(f"fail - {msg}")

    return result


def handle_cache_command(
    *,
    action: str,
    owner: str | None = None,
    dry_run: bool = False,
    include_system: bool = False,
    idle_days: float | int | str | None = None,
    include_recent: bool = False,
) -> tuple[bool, str]:
    """High-level API used by the brain.

    action: scan | clear | clear_unused | clear_all_unused
    """
    action = (action or "scan").lower().strip()
    keep = _parse_idle_days(idle_days, DEFAULT_IDLE_DAYS)
    if action in {"scan", "list", "show", "check"}:
        entries = scan_caches(include_system=include_system, idle_days=keep)
        return True, format_scan(entries, idle_days=keep)

    if action in {"clear", "clean", "delete", "purge", "clear_unused", "clear_all_unused"}:
        # default always unused_only for safety (respects last-used keep window)
        unused_only = True
        result = clear_caches(
            owner=owner,
            unused_only=unused_only,
            include_system=include_system,
            dry_run=dry_run,
            idle_days=keep,
            include_recent=include_recent,
        )
        head = ("**Dry run** - no files deleted.\n" if dry_run else "") + result.summary()
        body = "\n".join(result.lines[:60])
        more = f"\n...({len(result.lines) - 60} more lines)" if len(result.lines) > 60 else ""
        ok = result.failed == 0
        return ok, f"{head}\n\n{body}{more}"

    return False, f"Unknown cache action `{action}`."


# Voice / natural-language phrase helpers used by brain ---------------------

_VOICE_PHRASES = (
    "clear unused caches",
    "clear unused cache",
    "clean unused caches",
    "clean app caches",
    "clear app caches",
    "purge unused caches",
    "free up cache",
    "free cache space",
    "delete unused caches",
    "clear caches older than 30 days",
    "clear caches not used in 14 days",
    "scan caches",
    "check caches",
    "show caches",
)


def voice_phrases() -> tuple[str, ...]:
    return _VOICE_PHRASES
