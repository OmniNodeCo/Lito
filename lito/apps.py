"""App registry — maps friendly names to launch commands.

Designed for Linux desktops; falls back gracefully on other OSes.
Uses almost no RAM: a plain dict + lazy discovery of .desktop files.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class AppEntry:
    name: str
    command: str
    aliases: tuple[str, ...] = ()
    description: str = ""

    def matches(self, query: str) -> bool:
        q = query.lower().strip()
        if not q:
            return False
        if q == self.name.lower() or q in self.name.lower():
            return True
        return any(q == a or q in a for a in self.aliases)


# Built-in catalogue — only entries whose binary exists are kept at runtime.
_CATALOGUE: tuple[AppEntry, ...] = (
    AppEntry("firefox", "firefox", ("browser", "web", "mozilla"), "Web browser"),
    AppEntry("chrome", "google-chrome", ("chromium", "google chrome"), "Chrome browser"),
    AppEntry("chromium", "chromium", ("chromium-browser",), "Chromium browser"),
    AppEntry("brave", "brave-browser", ("brave",), "Brave browser"),
    AppEntry("code", "code", ("vscode", "vs code", "visual studio code"), "VS Code"),
    AppEntry("cursor", "cursor", (), "Cursor editor"),
    AppEntry("terminal", "x-terminal-emulator", ("term", "console", "shell", "kitty", "alacritty", "gnome-terminal"), "Terminal"),
    AppEntry("kitty", "kitty", (), "Kitty terminal"),
    AppEntry("alacritty", "alacritty", (), "Alacritty terminal"),
    AppEntry("gnome-terminal", "gnome-terminal", (), "GNOME Terminal"),
    AppEntry("files", "xdg-open", ("file manager", "nautilus", "dolphin", "thunar", "explorer", "finder"), "File manager"),
    AppEntry("nautilus", "nautilus", ("files",), "GNOME Files"),
    AppEntry("dolphin", "dolphin", (), "KDE Dolphin"),
    AppEntry("thunar", "thunar", (), "Thunar"),
    AppEntry("spotify", "spotify", ("music",), "Spotify"),
    AppEntry("slack", "slack", (), "Slack"),
    AppEntry("discord", "discord", (), "Discord"),
    AppEntry("telegram", "telegram-desktop", ("tg",), "Telegram"),
    AppEntry("obsidian", "obsidian", ("notes app",), "Obsidian"),
    AppEntry("gimp", "gimp", (), "GIMP"),
    AppEntry("inkscape", "inkscape", (), "Inkscape"),
    AppEntry("vlc", "vlc", ("video", "media player"), "VLC"),
    AppEntry("calculator", "gnome-calculator", ("calc", "math"), "Calculator"),
    AppEntry("settings", "gnome-control-center", ("preferences", "control center"), "Settings"),
    AppEntry("system monitor", "gnome-system-monitor", ("task manager", "htop", "monitor"), "System monitor"),
    AppEntry("text editor", "gedit", ("editor", "notepad", "gedit", "kate", "mousepad"), "Text editor"),
    AppEntry("gedit", "gedit", (), "gedit"),
    AppEntry("kate", "kate", (), "Kate"),
    AppEntry("libreoffice", "libreoffice", ("writer", "office", "docs"), "LibreOffice"),
    AppEntry("writer", "libreoffice --writer", ("word", "document"), "LibreOffice Writer"),
    AppEntry("calc sheet", "libreoffice --calc", ("spreadsheet", "excel"), "LibreOffice Calc"),
    AppEntry("steam", "steam", ("games",), "Steam"),
    AppEntry("blender", "blender", (), "Blender"),
)


def _which(cmd: str) -> str | None:
    # Commands may include args ("libreoffice --writer")
    binary = cmd.split()[0]
    return shutil.which(binary)


def _platform_defaults() -> list[AppEntry]:
    system = platform.system()
    extras: list[AppEntry] = []
    if system == "Darwin":
        extras = [
            AppEntry("safari", "open -a Safari", ("browser",), "Safari"),
            AppEntry("finder", "open -a Finder", ("files", "file manager"), "Finder"),
            AppEntry("terminal", "open -a Terminal", ("term", "console"), "Terminal"),
            AppEntry("iterm", "open -a iTerm", ("iterm2",), "iTerm"),
            AppEntry("notes", "open -a Notes", (), "Notes"),
            AppEntry("mail", "open -a Mail", (), "Mail"),
            AppEntry("messages", "open -a Messages", (), "Messages"),
            AppEntry("music", "open -a Music", ("itunes",), "Music"),
            AppEntry("preview", "open -a Preview", (), "Preview"),
            AppEntry("system settings", "open -a 'System Settings'", ("settings", "preferences"), "System Settings"),
        ]
    elif system == "Windows":
        extras = [
            AppEntry("explorer", "explorer", ("files", "file manager"), "File Explorer"),
            AppEntry("notepad", "notepad", ("editor", "text editor"), "Notepad"),
            AppEntry("cmd", "cmd", ("terminal", "shell"), "Command Prompt"),
            AppEntry("powershell", "powershell", ("pwsh", "terminal"), "PowerShell"),
            AppEntry("calculator", "calc", ("calc",), "Calculator"),
            AppEntry("paint", "mspaint", (), "Paint"),
            AppEntry("edge", "start msedge", ("browser",), "Microsoft Edge"),
        ]
    return extras


def discover_desktop_apps(limit: int = 80) -> list[AppEntry]:
    """Parse a few .desktop files for extra apps. Capped to stay light."""
    dirs = [
        Path("/usr/share/applications"),
        Path.home() / ".local/share/applications",
        Path("/var/lib/flatpak/exports/share/applications"),
    ]
    found: list[AppEntry] = []
    seen: set[str] = set()
    for d in dirs:
        if not d.is_dir():
            continue
        try:
            entries = sorted(d.glob("*.desktop"))
        except OSError:
            continue
        for desk in entries:
            if len(found) >= limit:
                return found
            try:
                text = desk.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            name = cmd = ""
            nodisplay = False
            for line in text.splitlines():
                if line.startswith("Name=") and not name:
                    name = line.split("=", 1)[1].strip()
                elif line.startswith("Exec=") and not cmd:
                    raw = line.split("=", 1)[1].strip()
                    # Strip field codes %f %u etc.
                    parts = [p for p in raw.split() if not p.startswith("%")]
                    cmd = " ".join(parts)
                elif line.startswith("NoDisplay=true"):
                    nodisplay = True
                elif line.startswith("Type=") and "Application" not in line:
                    nodisplay = True
            if nodisplay or not name or not cmd:
                continue
            key = name.lower()
            if key in seen:
                continue
            binary = cmd.split()[0]
            if not (binary.startswith("/") or shutil.which(binary)):
                continue
            seen.add(key)
            found.append(AppEntry(name=name, command=cmd, aliases=(key,)))
    return found


class AppRegistry:
    """Lazy, cached list of launchable apps."""

    __slots__ = ("_apps", "_loaded")

    def __init__(self) -> None:
        self._apps: list[AppEntry] = []
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        apps: list[AppEntry] = []
        seen_cmds: set[str] = set()
        for entry in (*_CATALOGUE, *_platform_defaults()):
            # mac/win open -a style always kept if on that OS
            cmd0 = entry.command.split()[0]
            if cmd0 in ("open", "start", "explorer", "notepad", "calc", "mspaint", "cmd", "powershell"):
                apps.append(entry)
                seen_cmds.add(entry.command)
                continue
            if _which(entry.command):
                apps.append(entry)
                seen_cmds.add(entry.command)
        # Special: files via xdg-open $HOME
        if shutil.which("xdg-open"):
            apps.append(
                AppEntry(
                    "files",
                    f"xdg-open {Path.home()}",
                    ("file manager", "home folder", "downloads"),
                    "Open home folder",
                )
            )
        # Desktop discovery (optional, light)
        if os.environ.get("LITO_NO_DESKTOP_SCAN") != "1":
            for extra in discover_desktop_apps():
                if extra.command not in seen_cmds:
                    apps.append(extra)
                    seen_cmds.add(extra.command)
        self._apps = apps
        self._loaded = True

    def all(self) -> list[AppEntry]:
        self._load()
        return list(self._apps)

    def find(self, query: str) -> AppEntry | None:
        self._load()
        q = query.lower().strip()
        if not q:
            return None
        # Exact alias / name first
        for app in self._apps:
            if q == app.name.lower() or q in app.aliases:
                return app
        # Substring
        for app in self._apps:
            if app.matches(q):
                return app
        # Fuzzy-ish: all tokens present
        tokens = q.split()
        for app in self._apps:
            blob = f"{app.name} {' '.join(app.aliases)}".lower()
            if all(t in blob for t in tokens):
                return app
        return None

    def search(self, query: str, limit: int = 12) -> list[AppEntry]:
        self._load()
        q = query.lower().strip()
        if not q:
            return self._apps[:limit]
        scored: list[tuple[int, AppEntry]] = []
        for app in self._apps:
            blob = f"{app.name} {' '.join(app.aliases)} {app.description}".lower()
            score = 0
            if q == app.name.lower():
                score = 100
            elif q in app.aliases:
                score = 90
            elif blob.startswith(q):
                score = 70
            elif q in blob:
                score = 50
            else:
                toks = q.split()
                if toks and all(t in blob for t in toks):
                    score = 30
            if score:
                scored.append((score, app))
        scored.sort(key=lambda x: (-x[0], x[1].name.lower()))
        return [a for _, a in scored[:limit]]


def launch(app: AppEntry, extra_args: Iterable[str] | None = None) -> tuple[bool, str]:
    """Spawn an app detached from Lito. Returns (ok, message)."""
    cmd = app.command
    args = list(extra_args or ())

    # Expand "files" style
    if cmd.startswith("xdg-open ") or cmd == "xdg-open":
        full = cmd.split() + args
    else:
        full = cmd.split() + args

    system = platform.system()
    try:
        if system == "Windows":
            # start is a shell builtin
            if full[0] == "start":
                subprocess.Popen(" ".join(full), shell=True, close_fds=True)
            else:
                subprocess.Popen(
                    full,
                    shell=False,
                    close_fds=True,
                    creationflags=getattr(subprocess, "DETACHED_PROCESS", 0)
                    | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
                )
        else:
            kwargs: dict = {
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
                "stdin": subprocess.DEVNULL,
                "start_new_session": True,
                "close_fds": True,
            }
            subprocess.Popen(full, **kwargs)
        return True, f"Opened **{app.name}**."
    except FileNotFoundError:
        return False, f"Could not find executable for **{app.name}** (`{full[0]}`)."
    except OSError as exc:
        return False, f"Failed to open **{app.name}**: {exc}"


def open_path(path: str) -> tuple[bool, str]:
    """Open a file or folder with the OS default handler."""
    p = Path(path).expanduser()
    if not p.exists():
        return False, f"Path not found: `{path}`"
    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.Popen(["open", str(p)], start_new_session=True)
        elif system == "Windows":
            os.startfile(str(p))  # type: ignore[attr-defined]
        else:
            subprocess.Popen(
                ["xdg-open", str(p)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        return True, f"Opened `{p}`."
    except OSError as exc:
        return False, f"Could not open `{p}`: {exc}"


def open_url(url: str) -> tuple[bool, str]:
    if not url.startswith(("http://", "https://", "file://")):
        url = "https://" + url
    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.Popen(["open", url], start_new_session=True)
        elif system == "Windows":
            os.startfile(url)  # type: ignore[attr-defined]
        else:
            opener = shutil.which("xdg-open") or shutil.which("firefox") or shutil.which("chromium")
            if not opener:
                return False, "No browser opener found (xdg-open / firefox)."
            subprocess.Popen(
                [opener, url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        return True, f"Opening {url}"
    except OSError as exc:
        return False, f"Could not open URL: {exc}"


# Singleton used by the brain
registry = AppRegistry()
