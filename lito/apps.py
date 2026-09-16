"""App registry - maps friendly names to launch commands.

Discovers *all* installed desktop apps (Linux .desktop, macOS /Applications,
Windows Start Menu) so `list apps` can show the full inventory.
Uses almost no RAM: plain lists + lazy one-shot scan.
"""

from __future__ import annotations

import os
import platform
import re
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
    source: str = ""  # catalogue | desktop | flatpak | snap | macos | windows

    def matches(self, query: str) -> bool:
        q = query.lower().strip()
        if not q:
            return False
        if q == self.name.lower() or q in self.name.lower():
            return True
        return any(q == a or q in a for a in self.aliases)


# Built-in catalogue - only entries whose binary exists are kept at runtime.
_CATALOGUE: tuple[AppEntry, ...] = (
    AppEntry("firefox", "firefox", ("browser", "web", "mozilla"), "Web browser", "catalogue"),
    AppEntry("chrome", "google-chrome", ("chromium", "google chrome"), "Chrome browser", "catalogue"),
    AppEntry("chromium", "chromium", ("chromium-browser",), "Chromium browser", "catalogue"),
    AppEntry("brave", "brave-browser", ("brave",), "Brave browser", "catalogue"),
    AppEntry("code", "code", ("vscode", "vs code", "visual studio code"), "VS Code", "catalogue"),
    AppEntry("cursor", "cursor", (), "Cursor editor", "catalogue"),
    AppEntry(
        "terminal",
        "x-terminal-emulator",
        ("term", "console", "shell", "kitty", "alacritty", "gnome-terminal"),
        "Terminal",
        "catalogue",
    ),
    AppEntry("kitty", "kitty", (), "Kitty terminal", "catalogue"),
    AppEntry("alacritty", "alacritty", (), "Alacritty terminal", "catalogue"),
    AppEntry("gnome-terminal", "gnome-terminal", (), "GNOME Terminal", "catalogue"),
    AppEntry(
        "files",
        "xdg-open",
        ("file manager", "nautilus", "dolphin", "thunar", "explorer", "finder"),
        "File manager",
        "catalogue",
    ),
    AppEntry("nautilus", "nautilus", ("files",), "GNOME Files", "catalogue"),
    AppEntry("dolphin", "dolphin", (), "KDE Dolphin", "catalogue"),
    AppEntry("thunar", "thunar", (), "Thunar", "catalogue"),
    AppEntry("spotify", "spotify", ("music",), "Spotify", "catalogue"),
    AppEntry("slack", "slack", (), "Slack", "catalogue"),
    AppEntry("discord", "discord", (), "Discord", "catalogue"),
    AppEntry("telegram", "telegram-desktop", ("tg",), "Telegram", "catalogue"),
    AppEntry("obsidian", "obsidian", ("notes app",), "Obsidian", "catalogue"),
    AppEntry("gimp", "gimp", (), "GIMP", "catalogue"),
    AppEntry("inkscape", "inkscape", (), "Inkscape", "catalogue"),
    AppEntry("vlc", "vlc", ("video", "media player"), "VLC", "catalogue"),
    AppEntry("calculator", "gnome-calculator", ("calc", "math"), "Calculator", "catalogue"),
    AppEntry("settings", "gnome-control-center", ("preferences", "control center"), "Settings", "catalogue"),
    AppEntry(
        "system monitor",
        "gnome-system-monitor",
        ("task manager", "htop", "monitor"),
        "System monitor",
        "catalogue",
    ),
    AppEntry(
        "text editor",
        "gedit",
        ("editor", "notepad", "gedit", "kate", "mousepad"),
        "Text editor",
        "catalogue",
    ),
    AppEntry("gedit", "gedit", (), "gedit", "catalogue"),
    AppEntry("kate", "kate", (), "Kate", "catalogue"),
    AppEntry("libreoffice", "libreoffice", ("writer", "office", "docs"), "LibreOffice", "catalogue"),
    AppEntry("writer", "libreoffice --writer", ("word", "document"), "LibreOffice Writer", "catalogue"),
    AppEntry("calc sheet", "libreoffice --calc", ("spreadsheet", "excel"), "LibreOffice Calc", "catalogue"),
    AppEntry("steam", "steam", ("games",), "Steam", "catalogue"),
    AppEntry("blender", "blender", (), "Blender", "catalogue"),
)


def _which(cmd: str) -> str | None:
    binary = cmd.split()[0]
    return shutil.which(binary)


def _platform_defaults() -> list[AppEntry]:
    system = platform.system()
    extras: list[AppEntry] = []
    if system == "Darwin":
        extras = [
            AppEntry("safari", "open -a Safari", ("browser",), "Safari", "catalogue"),
            AppEntry("finder", "open -a Finder", ("files", "file manager"), "Finder", "catalogue"),
            AppEntry("terminal", "open -a Terminal", ("term", "console"), "Terminal", "catalogue"),
            AppEntry("iterm", "open -a iTerm", ("iterm2",), "iTerm", "catalogue"),
            AppEntry("notes", "open -a Notes", (), "Notes", "catalogue"),
            AppEntry("mail", "open -a Mail", (), "Mail", "catalogue"),
            AppEntry("messages", "open -a Messages", (), "Messages", "catalogue"),
            AppEntry("music", "open -a Music", ("itunes",), "Music", "catalogue"),
            AppEntry("preview", "open -a Preview", (), "Preview", "catalogue"),
            AppEntry(
                "system settings",
                "open -a 'System Settings'",
                ("settings", "preferences"),
                "System Settings",
                "catalogue",
            ),
        ]
    elif system == "Windows":
        extras = [
            AppEntry("explorer", "explorer", ("files", "file manager"), "File Explorer", "catalogue"),
            AppEntry("notepad", "notepad", ("editor", "text editor"), "Notepad", "catalogue"),
            AppEntry("cmd", "cmd", ("terminal", "shell"), "Command Prompt", "catalogue"),
            AppEntry("powershell", "powershell", ("pwsh", "terminal"), "PowerShell", "catalogue"),
            AppEntry("calculator", "calc", ("calc",), "Calculator", "catalogue"),
            AppEntry("paint", "mspaint", (), "Paint", "catalogue"),
            AppEntry("edge", "start msedge", ("browser",), "Microsoft Edge", "catalogue"),
        ]
    return extras


def _parse_desktop_file(path: Path) -> AppEntry | None:
    """Parse one .desktop file into an AppEntry, or None if not a user-facing app."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None

    name = generic = cmd = comment = icon = ""
    nodisplay = hidden = false_type = False
    try_exec = ""
    terminal_only = False

    in_desktop_entry = False
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("[") and s.endswith("]"):
            in_desktop_entry = s.lower() == "[desktop entry]"
            continue
        if not in_desktop_entry:
            continue
        low = s.lower()
        if low.startswith("name=") and not name:
            name = s.split("=", 1)[1].strip()
        elif low.startswith("name[") and not name:
            # Prefer first localized name only if generic Name missing later
            pass
        elif low.startswith("genericname=") and not generic:
            generic = s.split("=", 1)[1].strip()
        elif low.startswith("exec=") and not cmd:
            raw = s.split("=", 1)[1].strip()
            # Strip field codes %f %u %c %k etc. and quoting
            parts = []
            for p in re.findall(r'"[^"]*"|\'[^\']*\'|\S+', raw):
                p = p.strip("\"'")
                if p.startswith("%"):
                    continue
                parts.append(p)
            cmd = " ".join(parts)
        elif low.startswith("tryexec=") and not try_exec:
            try_exec = s.split("=", 1)[1].strip()
        elif low.startswith("comment=") and not comment:
            comment = s.split("=", 1)[1].strip()
        elif low.startswith("icon=") and not icon:
            icon = s.split("=", 1)[1].strip()
        elif low == "nodisplay=true" or low == "hidden=true":
            nodisplay = True
        elif low.startswith("type=") and "application" not in low:
            false_type = True
        elif low == "terminal=true":
            terminal_only = True

    if nodisplay or hidden or false_type or not name or not cmd:
        return None

    binary = cmd.split()[0].strip("\"'")
    # Keep entries generously so "list apps" shows everything the desktop
    # knows about. Only drop when TryExec explicitly fails on an absolute path.
    if try_exec:
        te = try_exec.strip("\"'")
        if te.startswith("/") and not Path(te).exists():
            if not (binary.startswith("/") and Path(binary).exists()) and not shutil.which(binary):
                return None

    aliases = [name.lower()]
    # slug from desktop filename
    stem = path.stem.lower()
    if stem and stem not in aliases:
        aliases.append(stem)
        # org.gnome.TextEditor -> texteditor bits
        if "." in stem:
            aliases.append(stem.split(".")[-1])
    if generic:
        aliases.append(generic.lower())

    source = "desktop"
    pstr = str(path).lower()
    if "flatpak" in pstr or cmd.startswith("flatpak "):
        source = "flatpak"
    elif "/snap/" in pstr or cmd.startswith("snap "):
        source = "snap"

    desc = comment or generic or source
    return AppEntry(
        name=name,
        command=cmd,
        aliases=tuple(dict.fromkeys(aliases)),  # dedupe, keep order
        description=desc[:120],
        source=source,
    )


def _desktop_search_dirs() -> list[Path]:
    dirs: list[Path] = [
        Path("/usr/share/applications"),
        Path("/usr/local/share/applications"),
        Path("/var/lib/flatpak/exports/share/applications"),
        Path("/var/lib/snapd/desktop/applications"),
        Path.home() / ".local/share/applications",
        Path.home() / ".local/share/flatpak/exports/share/applications",
    ]
    # XDG_DATA_DIRS
    xdg = os.environ.get("XDG_DATA_DIRS", "")
    for part in xdg.split(":"):
        if not part:
            continue
        dirs.append(Path(part) / "applications")
    # de-dupe while preserving order
    seen: set[str] = set()
    out: list[Path] = []
    for d in dirs:
        key = str(d)
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def discover_desktop_apps(limit: int = 5000) -> list[AppEntry]:
    """Parse .desktop files for installed GUI apps. Uncapped by default."""
    found: list[AppEntry] = []
    seen_names: set[str] = set()

    for d in _desktop_search_dirs():
        if not d.is_dir():
            continue
        try:
            entries = sorted(d.glob("*.desktop"))
        except OSError:
            continue
        for desk in entries:
            if len(found) >= limit:
                return found
            app = _parse_desktop_file(desk)
            if not app:
                continue
            key = app.name.lower()
            if key in seen_names:
                continue
            seen_names.add(key)
            found.append(app)
    return found


def discover_macos_apps(limit: int = 2000) -> list[AppEntry]:
    """List .app bundles under /Applications and ~/Applications."""
    roots = [
        Path("/Applications"),
        Path("/System/Applications"),
        Path.home() / "Applications",
        Path("/System/Applications/Utilities"),
    ]
    found: list[AppEntry] = []
    seen: set[str] = set()
    for root in roots:
        if not root.is_dir():
            continue
        try:
            children = list(root.iterdir())
        except OSError:
            continue
        # Also one level of subfolders (e.g. /Applications/Utilities already covered)
        candidates: list[Path] = []
        for child in children:
            if child.suffix == ".app" and child.is_dir():
                candidates.append(child)
            elif child.is_dir():
                try:
                    for nested in child.iterdir():
                        if nested.suffix == ".app" and nested.is_dir():
                            candidates.append(nested)
                except OSError:
                    pass
        for app_path in sorted(candidates, key=lambda p: p.name.lower()):
            if len(found) >= limit:
                return found
            name = app_path.stem
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            # Prefer open -a "Name" which resolves by bundle name
            cmd = f'open -a "{name}"'
            found.append(
                AppEntry(
                    name=name,
                    command=cmd,
                    aliases=(key, name.replace(" ", "").lower()),
                    description=str(app_path),
                    source="macos",
                )
            )
    return found


def discover_windows_apps(limit: int = 2000) -> list[AppEntry]:
    """Discover Start Menu shortcuts (.lnk) via PowerShell when available."""
    found: list[AppEntry] = []
    seen: set[str] = set()

    # Built-in always-on Windows tools already come from catalogue defaults.
    # Scan Start Menu .lnk files with a short PowerShell script.
    ps = shutil.which("powershell") or shutil.which("pwsh")
    if not ps:
        return found

    script = r"""
$ErrorActionPreference = 'SilentlyContinue'
$roots = @(
  "$env:ProgramData\Microsoft\Windows\Start Menu\Programs",
  "$env:APPDATA\Microsoft\Windows\Start Menu\Programs"
)
$wsh = New-Object -ComObject WScript.Shell
foreach ($root in $roots) {
  if (-not (Test-Path $root)) { continue }
  Get-ChildItem -Path $root -Filter *.lnk -Recurse -ErrorAction SilentlyContinue | ForEach-Object {
    try {
      $s = $wsh.CreateShortcut($_.FullName)
      $name = [System.IO.Path]::GetFileNameWithoutExtension($_.Name)
      $target = $s.TargetPath
      if (-not $target) { return }
      if ($target -match '(?i)uninstall|help\.lnk|website') { return }
      Write-Output ($name + "`t" + $target + "`t" + $s.Arguments)
    } catch {}
  }
}
"""
    try:
        out = subprocess.check_output(
            [ps, "-NoProfile", "-NonInteractive", "-Command", script],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=20,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return found

    for line in out.splitlines():
        if len(found) >= limit:
            break
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        name, target = parts[0].strip(), parts[1].strip()
        args = parts[2].strip() if len(parts) > 2 else ""
        if not name or not target:
            continue
        key = name.lower()
        if key in seen:
            continue
        # Skip obvious uninstallers
        if re.search(r"uninstall|setup\.exe$|installer", target, re.I):
            continue
        seen.add(key)
        cmd = f'"{target}"'
        if args:
            cmd = f'"{target}" {args}'
        found.append(
            AppEntry(
                name=name,
                command=cmd,
                aliases=(key,),
                description=target,
                source="windows",
            )
        )
    return found


def discover_flatpak_cli(limit: int = 500) -> list[AppEntry]:
    """List flatpak apps via `flatpak list` when the CLI exists."""
    if not shutil.which("flatpak"):
        return []
    try:
        out = subprocess.check_output(
            ["flatpak", "list", "--app", "--columns=application,name"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=8,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return []
    found: list[AppEntry] = []
    for line in out.splitlines():
        if len(found) >= limit:
            break
        line = line.strip()
        if not line or line.lower().startswith("application"):
            continue
        # format can be "app.id    Name" tab or multi-space
        parts = re.split(r"\s{2,}|\t", line, maxsplit=1)
        if len(parts) == 1:
            app_id = parts[0].strip()
            name = app_id.split(".")[-1]
        else:
            app_id, name = parts[0].strip(), parts[1].strip()
        if not app_id:
            continue
        found.append(
            AppEntry(
                name=name or app_id,
                command=f"flatpak run {app_id}",
                aliases=(name.lower(), app_id.lower(), app_id.split(".")[-1].lower()),
                description=app_id,
                source="flatpak",
            )
        )
    return found


def discover_snap_cli(limit: int = 500) -> list[AppEntry]:
    if not shutil.which("snap"):
        return []
    try:
        out = subprocess.check_output(
            ["snap", "list"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=8,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return []
    found: list[AppEntry] = []
    for i, line in enumerate(out.splitlines()):
        if i == 0 or len(found) >= limit:
            continue  # header
        parts = line.split()
        if not parts:
            continue
        name = parts[0]
        if name in {"core", "core18", "core20", "core22", "core24", "snapd", "bare", "gtk-common-themes"}:
            continue
        found.append(
            AppEntry(
                name=name,
                command=f"snap run {name}",
                aliases=(name.lower(),),
                description="snap",
                source="snap",
            )
        )
    return found


class AppRegistry:
    """Lazy, cached list of launchable apps - full installed inventory."""

    __slots__ = ("_apps", "_loaded")

    def __init__(self) -> None:
        self._apps: list[AppEntry] = []
        self._loaded = False

    def reload(self) -> None:
        """Force a fresh scan of installed apps."""
        self._loaded = False
        self._apps = []
        self._load()

    def _load(self) -> None:
        if self._loaded:
            return
        apps: list[AppEntry] = []
        seen_keys: set[str] = set()

        def add(entry: AppEntry) -> None:
            key = entry.name.lower().strip()
            if not key or key in seen_keys:
                return
            # Also skip exact command dupes of same name family
            seen_keys.add(key)
            apps.append(entry)

        system = platform.system()
        for entry in (*_CATALOGUE, *_platform_defaults()):
            cmd0 = entry.command.split()[0]
            if cmd0 in ("open", "start", "explorer", "notepad", "calc", "mspaint", "cmd", "powershell"):
                add(entry)
                continue
            if _which(entry.command):
                add(entry)

        if shutil.which("xdg-open"):
            add(
                AppEntry(
                    "files",
                    f"xdg-open {Path.home()}",
                    ("file manager", "home folder", "downloads"),
                    "Open home folder",
                    "catalogue",
                )
            )

        skip_scan = os.environ.get("LITO_NO_DESKTOP_SCAN") == "1"
        if not skip_scan:
            if system == "Linux":
                for extra in discover_desktop_apps():
                    add(extra)
                for extra in discover_flatpak_cli():
                    add(extra)
                for extra in discover_snap_cli():
                    add(extra)
            elif system == "Darwin":
                for extra in discover_macos_apps():
                    add(extra)
            elif system == "Windows":
                for extra in discover_windows_apps():
                    add(extra)

        apps.sort(key=lambda a: a.name.lower())
        self._apps = apps
        self._loaded = True

    def all(self) -> list[AppEntry]:
        self._load()
        return list(self._apps)

    def count(self) -> int:
        self._load()
        return len(self._apps)

    def find(self, query: str) -> AppEntry | None:
        self._load()
        q = query.lower().strip()
        if not q:
            return None
        for app in self._apps:
            if q == app.name.lower() or q in app.aliases:
                return app
        for app in self._apps:
            if app.matches(q):
                return app
        tokens = q.split()
        for app in self._apps:
            blob = f"{app.name} {' '.join(app.aliases)}".lower()
            if all(t in blob for t in tokens):
                return app
        return None

    def search(self, query: str, limit: int | None = 12) -> list[AppEntry]:
        self._load()
        q = query.lower().strip()
        if not q:
            return list(self._apps) if limit is None else self._apps[:limit]
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
        if limit is None:
            return [a for _, a in scored]
        return [a for _, a in scored[:limit]]


def launch(app: AppEntry, extra_args: Iterable[str] | None = None) -> tuple[bool, str]:
    """Spawn an app detached from Lito. Returns (ok, message)."""
    cmd = app.command
    args = list(extra_args or ())

    if cmd.startswith("xdg-open ") or cmd == "xdg-open":
        full = cmd.split() + args
    elif cmd.startswith("open -a ") or cmd.startswith("flatpak ") or cmd.startswith("snap "):
        # Keep as shell-ish for quoted app names on macOS
        full = None  # type: ignore[assignment]
    else:
        full = cmd.split() + args

    system = platform.system()
    try:
        if system == "Windows":
            if full and full[0] == "start":
                subprocess.Popen(" ".join(full), shell=True, close_fds=True)
            elif full is None:
                subprocess.Popen(cmd, shell=True, close_fds=True)
            elif full[0].startswith('"') or (full and " " in full[0] and not full[0].startswith("/")):
                # Quoted Windows path
                subprocess.Popen(cmd + ((" " + " ".join(args)) if args else ""), shell=True, close_fds=True)
            else:
                subprocess.Popen(
                    full,
                    shell=False,
                    close_fds=True,
                    creationflags=getattr(subprocess, "DETACHED_PROCESS", 0)
                    | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
                )
        else:
            if full is None or (cmd.startswith("open -a ") and " " in cmd):
                subprocess.Popen(
                    cmd if full is None else cmd,
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                    close_fds=True,
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
        try:
            from .usage import record_launch

            record_launch(app.name, app.command)
        except Exception:
            pass
        return True, f"Opened **{app.name}**."
    except FileNotFoundError:
        bin0 = (full or [cmd])[0]
        return False, f"Could not find executable for **{app.name}** (`{bin0}`)."
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
