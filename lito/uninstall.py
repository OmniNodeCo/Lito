"""Uninstall installed apps safely by owner type.

Resolves the app via the registry, detects package manager / source, then
runs the matching uninstall path. Requires confirm=True for actual removal
unless LITO_UNINSTALL_YES=1 (for automation).

Windows notes
-------------
`winget uninstall --name "Kleopatra"` often fails because Kleopatra is a
component of **Gpg4win** (id `GnuPG.Gpg4win`), not a standalone winget name.
We therefore:
1. Resolve a winget **Id** via `winget list` / `winget search`
2. Fall back to the Uninstall registry (MSI / Inno / NSIS strings)
3. Last resort: open Windows Apps & Features
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .apps import AppEntry, registry


@dataclass
class UninstallPlan:
    app: AppEntry
    method: str  # flatpak | snap | apt | dnf | pacman | brew | macos | winget | windows-registry | windows-settings | unknown
    command: str
    detail: str = ""
    risky: bool = False

    def summary(self) -> str:
        risk = " (needs care)" if self.risky else ""
        return (
            f"**{self.app.name}** via `{self.method}`{risk}\n"
            f"- plan: `{self.command}`\n"
            f"- {self.detail or self.app.source or 'installed app'}"
        )


def _run(cmd: list[str] | str, timeout: float = 180.0) -> tuple[bool, str]:
    run_kw: dict = {"capture_output": True, "text": True, "timeout": timeout}
    if platform.system() == "Windows":
        cf = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        if cf:
            run_kw["creationflags"] = cf
    try:
        if isinstance(cmd, str):
            proc = subprocess.run(cmd, shell=True, **run_kw)
        else:
            proc = subprocess.run(cmd, **run_kw)
    except TypeError:
        run_kw.pop("creationflags", None)
        try:
            if isinstance(cmd, str):
                proc = subprocess.run(cmd, shell=True, **run_kw)
            else:
                proc = subprocess.run(cmd, **run_kw)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return False, str(exc)
    except subprocess.TimeoutExpired:
        return False, f"Timed out after {timeout}s"
    except OSError as exc:
        return False, str(exc)
    out = ((proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")).strip()
    # winget prints progress with CR; normalize
    out = re.sub(r"[^\S\n]*\r[^\S\n]*", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return proc.returncode == 0, out[:4000] or f"(exit {proc.returncode})"


def _flatpak_id(app: AppEntry) -> str | None:
    if app.command.startswith("flatpak run "):
        return app.command.split("flatpak run ", 1)[1].strip().split()[0]
    if app.description and re.match(r"^[\w.-]+\.[\w.-]+", app.description):
        return app.description.split()[0]
    for a in app.aliases:
        if a.count(".") >= 2 and " " not in a:
            return a
    return None


def _snap_name(app: AppEntry) -> str | None:
    if app.command.startswith("snap run "):
        return app.command.split("snap run ", 1)[1].strip().split()[0]
    if app.source == "snap":
        return app.name.lower().replace(" ", "-")
    return None


def _apt_package_for_binary(binary: str) -> str | None:
    if not shutil.which("dpkg") and not shutil.which("apt-get"):
        return None
    path = shutil.which(binary) or (binary if binary.startswith("/") else None)
    if not path:
        return None
    try:
        out = subprocess.check_output(
            ["dpkg", "-S", path], text=True, stderr=subprocess.DEVNULL, timeout=5
        )
        pkg = out.split(":", 1)[0].strip()
        return pkg or None
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


# ---------------------------------------------------------------------------
# Windows helpers
# ---------------------------------------------------------------------------

# Well-known component → parent package (when winget name match fails)
_WINDOWS_COMPONENT_HINTS: dict[str, tuple[str, ...]] = {
    "kleopatra": ("gpg4win", "gnupg.gpg4win", "gnupg"),
    "gpa": ("gpg4win", "gnupg.gpg4win"),
    "kleopatra.exe": ("gpg4win",),
    "gpgex": ("gpg4win",),
    "wkdlookup": ("gpg4win",),
}


def _winget_available() -> bool:
    return bool(shutil.which("winget"))


def _parse_winget_table(raw: str) -> list[dict[str, str]]:
    """Parse winget list/search text table into {name,id,version} rows."""
    lines = [re.sub(r".*\r", "", ln).rstrip() for ln in (raw or "").splitlines()]
    lines = [ln for ln in lines if ln.strip()]
    header_i = -1
    for i, ln in enumerate(lines):
        low = ln.lower()
        if "name" in low and re.search(r"\bid\b", low):
            header_i = i
            break
    if header_i < 0:
        return []
    header = lines[header_i]
    m_id = re.search(r"\bId\b", header, re.I)
    m_ver = re.search(r"\bVersion\b", header, re.I)
    if not m_id:
        return []
    id_pos = m_id.start()
    ver_pos = m_ver.start() if m_ver else -1
    rows: list[dict[str, str]] = []
    for ln in lines[header_i + 1 :]:
        if re.match(r"^-+$", ln.strip()) or set(ln.strip()) <= {"-"}:
            continue
        low = ln.lower()
        if "no installed package" in low or "no package found" in low:
            continue
        if len(ln) <= id_pos:
            continue
        name = ln[:id_pos].strip()
        if ver_pos > id_pos and len(ln) >= ver_pos:
            pkg_id = ln[id_pos:ver_pos].strip()
            version = ln[ver_pos:].strip().split()[0] if ln[ver_pos:].strip() else ""
        else:
            parts = ln[id_pos:].split()
            pkg_id = parts[0] if parts else ""
            version = parts[1] if len(parts) > 1 else ""
        if name and pkg_id and not pkg_id.startswith("-"):
            rows.append({"name": name, "id": pkg_id, "version": version})
    return rows


def _winget_list_matches(query: str) -> list[dict[str, str]]:
    if not _winget_available() or not (query or "").strip():
        return []
    q = query.strip()
    cmd = [
        "winget",
        "list",
        "--query",
        q,
        "--disable-interactivity",
        "--accept-source-agreements",
    ]
    ok, out = _run(cmd, timeout=60)
    rows = _parse_winget_table(out)
    if rows:
        return rows
    # Some winget builds return non-zero when 0 matches; still try search
    return []


def _winget_search_matches(query: str) -> list[dict[str, str]]:
    if not _winget_available() or not (query or "").strip():
        return []
    cmd = [
        "winget",
        "search",
        "--query",
        query.strip(),
        "--disable-interactivity",
        "--accept-source-agreements",
    ]
    _ok, out = _run(cmd, timeout=60)
    return _parse_winget_table(out)


def _windows_registry_uninstallers(query: str) -> list[dict[str, str]]:
    """Read Uninstall registry for DisplayName match -> UninstallString."""
    if platform.system() != "Windows":
        return []
    try:
        import winreg  # type: ignore
    except ImportError:
        return []

    roots = [
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
        (
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall",
        ),
        (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
    ]
    q = (query or "").strip().lower()
    tokens = [t for t in re.split(r"\W+", q) if len(t) >= 3]
    found: list[dict[str, str]] = []
    seen: set[str] = set()

    for hive, sub in roots:
        try:
            key = winreg.OpenKey(hive, sub)
        except OSError:
            continue
        try:
            i = 0
            while True:
                try:
                    sk_name = winreg.EnumKey(key, i)
                except OSError:
                    break
                i += 1
                try:
                    sk = winreg.OpenKey(key, sk_name)
                except OSError:
                    continue
                try:

                    def _val(name: str, _sk=sk) -> str:
                        try:
                            v, _ = winreg.QueryValueEx(_sk, name)
                            return str(v)
                        except OSError:
                            return ""

                    display = _val("DisplayName")
                    uninst = _val("UninstallString")
                    quiet = _val("QuietUninstallString")
                    if not display or not (uninst or quiet):
                        continue
                    dlow = display.lower()
                    if q and q not in dlow and not any(t in dlow for t in tokens):
                        continue
                    uid = f"{display}|{uninst}"
                    if uid in seen:
                        continue
                    seen.add(uid)
                    found.append(
                        {
                            "name": display,
                            "uninstall": quiet or uninst,
                            "quiet": quiet,
                            "normal": uninst,
                        }
                    )
                finally:
                    try:
                        winreg.CloseKey(sk)
                    except OSError:
                        pass
        finally:
            try:
                winreg.CloseKey(key)
            except OSError:
                pass

    def score(item: dict[str, str]) -> tuple:
        n = item["name"].lower()
        return (0 if n == q else 1, 0 if n.startswith(q) else 1, len(n))

    found.sort(key=score)
    return found


def _path_hints(app: AppEntry) -> list[str]:
    """Vendor / folder hints from the launch command path."""
    hints: list[str] = []
    cmd = app.command or ""
    m = re.search(r'([A-Za-z]:\\[^"\']+)', cmd)
    path = m.group(1) if m else cmd.strip().strip('"')
    skip = {
        "bin",
        "app",
        "application",
        "applications",
        "program files",
        "program files (x86)",
        "windows",
        "system32",
        "cmd",
        "users",
        "x86",
    }
    try:
        parts = Path(path).parts
    except Exception:
        parts = ()
    for part in parts:
        pl = part.lower().strip("\\/")
        if len(pl) < 3 or pl in skip:
            continue
        if pl.endswith(".exe"):
            hints.append(Path(pl).stem)
            continue
        hints.append(part)
    return hints


def _candidate_queries(app: AppEntry) -> list[str]:
    names: list[str] = []
    bare = re.sub(r"\s*(\(.*\)|-\s*shortcut|\.lnk)$", "", app.name, flags=re.I).strip()
    for n in (app.name, bare, *app.aliases):
        n = (n or "").strip()
        if n and n not in names:
            names.append(n)
    for h in _path_hints(app):
        if h not in names:
            names.append(h)
    # Component → parent package hints (Kleopatra → Gpg4win)
    extra: list[str] = []
    for n in list(names):
        key = n.lower()
        for hint in _WINDOWS_COMPONENT_HINTS.get(key, ()):
            if hint not in names and hint not in extra:
                extra.append(hint)
        # also stem of exe
        if key.endswith(".exe"):
            for hint in _WINDOWS_COMPONENT_HINTS.get(key[:-4], ()):
                if hint not in names and hint not in extra:
                    extra.append(hint)
    names.extend(extra)
    return names


def _plan_windows_registry_only(app: AppEntry) -> UninstallPlan | None:
    """Build a plan from the Windows Uninstall registry only."""
    reg: list[dict[str, str]] = []
    for q in _candidate_queries(app):
        reg = _windows_registry_uninstallers(q)
        if reg:
            break
    if not reg:
        return None

    best = reg[0]
    un = best.get("quiet") or best["uninstall"]
    msi = re.search(r"msiexec\.exe\s+(/[ix])\s*(\{[^}]+\}|[^\s]+)", un, re.I)
    if msi:
        command = f"msiexec /x {msi.group(2)} /qn /norestart"
    elif best.get("quiet"):
        command = best["quiet"]
    else:
        command = un
        # Inno Setup silent
        if re.search(r"unins(tall)?\d*\.exe", un, re.I) and "/SILENT" not in un.upper():
            if not un.startswith('"') and " " in un:
                command = f'"{un}" /SILENT'
            else:
                command = f"{un} /SILENT"
        elif re.search(r"Uninstall\.exe", un, re.I) and "/S" not in un:
            if not un.startswith('"') and " " in un:
                command = f'"{un}" /S'
            else:
                command = f"{un} /S"

    extras = ""
    if len(reg) > 1:
        extras = " · also matched: " + ", ".join(r["name"] for r in reg[1:4])
    return UninstallPlan(
        app=app,
        method="windows-registry",
        command=command,
        detail=f"Registry uninstall for **{best['name']}**{extras}",
        risky=True,
    )


def _pick_winget_row(rows: list[dict[str, str]], app: AppEntry) -> dict[str, str]:
    q = app.name.lower()
    queries = [x.lower() for x in _candidate_queries(app)]

    def score(r: dict[str, str]) -> tuple:
        nm = r["name"].lower()
        iid = r["id"].lower()
        exact = 0 if nm == q else 1
        name_hit = 0 if any(x and x in nm for x in queries) else 1
        id_hit = 0 if any(x and x in iid for x in queries) else 1
        return (exact, name_hit, id_hit, len(nm))

    return sorted(rows, key=score)[0]


def _plan_windows(app: AppEntry) -> UninstallPlan:
    """Best-effort Windows uninstall: winget id → registry → settings."""
    winget_rows: list[dict[str, str]] = []
    if _winget_available():
        for q in _candidate_queries(app):
            winget_rows = _winget_list_matches(q)
            if winget_rows:
                break
        if not winget_rows:
            for q in _candidate_queries(app):
                winget_rows = _winget_search_matches(q)
                # Prefer rows that look installed-ish; search may list available only
                if winget_rows:
                    break

    if winget_rows:
        best = _pick_winget_row(winget_rows, app)
        pkg_id = best["id"]
        extras = ""
        if len(winget_rows) > 1:
            extras = " · also: " + ", ".join(
                f"{r['name']} (`{r['id']}`)" for r in winget_rows[1:4]
            )
        return UninstallPlan(
            app=app,
            method="winget",
            command=(
                f'winget uninstall --id "{pkg_id}" '
                f"--exact --silent --disable-interactivity "
                f"--accept-source-agreements"
            ),
            detail=(
                f"winget package **{best['name']}** id `{pkg_id}`"
                + (f" v{best['version']}" if best.get("version") else "")
                + extras
            ),
            risky=True,
        )

    reg_plan = _plan_windows_registry_only(app)
    if reg_plan is not None:
        return reg_plan

    return UninstallPlan(
        app=app,
        method="windows-settings",
        command="start ms-settings:appsfeatures",
        detail=(
            f"Could not resolve a winget id or registry uninstaller for **{app.name}**. "
            f"Opening Windows Apps & Features — search for the app there. "
            f"Note: Kleopatra is often part of **Gpg4win** "
            f"(`winget uninstall --id GnuPG.Gpg4win`)."
        ),
        risky=False,
    )


def _run_windows_uninstall(plan: UninstallPlan) -> tuple[bool, str]:
    """Execute Windows uninstall with winget id/name/registry fallbacks."""
    if plan.method == "windows-settings" or plan.command.startswith("start "):
        ok, msg = _run(plan.command)
        return ok, msg or "Opened Windows Apps & Features."

    if plan.method == "windows-registry":
        return _run(plan.command)

    if plan.method != "winget":
        return _run(plan.command)

    # --- winget path with fallbacks ---
    ok, msg = _run(plan.command)
    low = (msg or "").lower()
    if ok and "no installed package" not in low:
        return True, msg

    app_name = plan.app.name
    m = re.search(r'--id\s+"([^"]+)"', plan.command)
    pkg_id = m.group(1) if m else ""
    attempts: list[list[str]] = []
    if pkg_id:
        attempts.append(
            [
                "winget",
                "uninstall",
                "--id",
                pkg_id,
                "--exact",
                "--disable-interactivity",
                "--accept-source-agreements",
            ]
        )
        attempts.append(
            [
                "winget",
                "uninstall",
                "--id",
                pkg_id,
                "--exact",
                "--force",
                "--disable-interactivity",
                "--accept-source-agreements",
            ]
        )
    # Try every resolved list match
    seen_ids = {pkg_id} if pkg_id else set()
    for q in _candidate_queries(plan.app):
        for row in _winget_list_matches(q):
            aid = row["id"]
            if aid in seen_ids:
                continue
            seen_ids.add(aid)
            attempts.append(
                [
                    "winget",
                    "uninstall",
                    "--id",
                    aid,
                    "--exact",
                    "--disable-interactivity",
                    "--accept-source-agreements",
                ]
            )
    attempts.append(
        [
            "winget",
            "uninstall",
            "--name",
            app_name,
            "--exact",
            "--disable-interactivity",
            "--accept-source-agreements",
        ]
    )

    logs = [f"$ {plan.command}\n{msg}"]
    for cmd in attempts:
        ok2, msg2 = _run(cmd)
        logs.append(f"$ {' '.join(cmd)}\n{msg2}")
        if ok2 and "no installed package" not in (msg2 or "").lower():
            return True, "\n\n".join(logs[-3:])

    # Registry fallback
    reg_plan = _plan_windows_registry_only(plan.app)
    if reg_plan is not None:
        ok3, msg3 = _run(reg_plan.command)
        logs.append(f"$ {reg_plan.command}\n{msg3}")
        if ok3:
            return True, "\n\n".join(logs[-3:])

    tip = (
        "Could not uninstall via winget (no matching package id).\n\n"
        + "\n\n".join(logs[-4:])
        + "\n\n**Tips**\n"
        + f"- Run `winget list {app_name}` and use the exact **Id**\n"
        + "- Kleopatra is often inside **Gpg4win**: "
        "`winget uninstall --id GnuPG.Gpg4win`\n"
        "- Or uninstall from **Settings → Apps → Installed apps**\n"
        "- Admin rights may be required"
    )
    return False, tip


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


def plan_uninstall(name: str) -> tuple[UninstallPlan | None, str]:
    app = registry.find(name)
    if not app:
        hits = registry.search(name, limit=5)
        if not hits:
            return None, f"No installed app matched **{name}**. Try `list apps`."
        if len(hits) > 1:
            opts = ", ".join(h.name for h in hits)
            return None, f"Ambiguous - did you mean: {opts}?"
        app = hits[0]

    system = platform.system()
    cmd = app.command

    # Flatpak
    fid = _flatpak_id(app)
    if fid and shutil.which("flatpak"):
        return (
            UninstallPlan(
                app=app,
                method="flatpak",
                command=f"flatpak uninstall -y {fid}",
                detail=f"Flatpak id `{fid}`",
            ),
            "",
        )

    # Snap
    sn = _snap_name(app)
    if (app.source == "snap" or sn) and shutil.which("snap"):
        snap = sn or app.name.lower()
        return (
            UninstallPlan(
                app=app,
                method="snap",
                command=f"snap remove {snap}",
                detail=f"Snap package `{snap}`",
                risky=True,
            ),
            "",
        )

    # macOS .app
    if system == "Darwin":
        m = re.search(r'open -a ["\']?([^"\']+)["\']?', cmd)
        app_name = m.group(1) if m else app.name
        for root in (Path("/Applications"), Path.home() / "Applications"):
            path = root / f"{app_name}.app"
            if path.is_dir():
                return (
                    UninstallPlan(
                        app=app,
                        method="macos",
                        command=f'rm -rf "{path}"',
                        detail=f"Move/remove bundle `{path}`",
                        risky=True,
                    ),
                    "",
                )
        if shutil.which("brew"):
            cask = app.name.lower().replace(" ", "-")
            return (
                UninstallPlan(
                    app=app,
                    method="brew",
                    command=f"brew uninstall --cask {cask}",
                    detail=f"Guessed Homebrew cask `{cask}` (verify before confirm)",
                    risky=True,
                ),
                "",
            )

    # Windows
    if system == "Windows":
        return _plan_windows(app), ""

    # Linux native packages
    binary = cmd.split()[0].strip("\"'") if cmd else ""
    if binary in {"xdg-open", "env", "sh", "bash"}:
        binary = ""
    pkg = _apt_package_for_binary(binary) if binary else None
    if pkg and shutil.which("apt-get"):
        return (
            UninstallPlan(
                app=app,
                method="apt",
                command=f"sudo apt-get remove -y {pkg}",
                detail=f"APT package `{pkg}` (from `{binary}`)",
                risky=True,
            ),
            "",
        )
    if binary and shutil.which("dnf"):
        return (
            UninstallPlan(
                app=app,
                method="dnf",
                command=f"sudo dnf remove -y {binary}",
                detail=f"DNF guess from binary `{binary}`",
                risky=True,
            ),
            "",
        )
    if binary and shutil.which("pacman"):
        return (
            UninstallPlan(
                app=app,
                method="pacman",
                command=f"sudo pacman -R --noconfirm {binary}",
                detail=f"pacman guess from binary `{binary}`",
                risky=True,
            ),
            "",
        )

    return (
        UninstallPlan(
            app=app,
            method="unknown",
            command="",
            detail=(
                f"Found **{app.name}** (`{app.command}`) but could not detect a "
                f"package manager uninstall path. Remove it from your software center, "
                f"or tell me the package name."
            ),
            risky=True,
        ),
        "",
    )


def _clear_owner_cache(app_name: str) -> str:
    """Best-effort: clear this app's user caches after uninstall (never blocks)."""
    try:
        from . import cache as cache_mod
    except Exception as exc:
        return f"(cache clean skipped: {exc})"
    try:
        # After uninstall the app is not running; include_recent so leftovers go too.
        result = cache_mod.clear_caches(
            owner=app_name,
            unused_only=True,
            include_recent=True,
            include_system=False,
            dry_run=False,
            idle_days=0,
        )
        if result.cleared:
            lines = [result.summary()]
            lines.extend(result.lines[:12])
            if len(result.lines) > 12:
                lines.append(f"...(+{len(result.lines) - 12} more)")
            return "\n".join(lines)
        return result.summary() + " (no matching user caches found)"
    except Exception as exc:
        return f"(cache clean failed: {exc})"


def uninstall_app(
    name: str,
    *,
    confirm: bool = False,
    clear_cache: bool | None = None,
) -> tuple[bool, str]:
    """Plan or execute uninstall. confirm=False -> dry description only.

    clear_cache:
      True  - also delete this app's user caches after a successful uninstall
      False - leave caches alone
      None  - default False, but the plan message offers the option
    Env: LITO_UNINSTALL_CLEAR_CACHE=1 forces clear_cache on confirm.
    """
    plan, err = plan_uninstall(name)
    if err:
        return False, err
    assert plan is not None

    if plan.method == "unknown" or not plan.command:
        return False, plan.summary()

    env_clear = os.environ.get("LITO_UNINSTALL_CLEAR_CACHE", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if clear_cache is None:
        want_cache = env_clear
    else:
        want_cache = bool(clear_cache) or env_clear

    auto = os.environ.get("LITO_UNINSTALL_YES", "").strip() in {"1", "true", "yes"}
    if not confirm and not auto:
        cache_line = (
            f"- cache: will **also clear** **{plan.app.name}** user caches after uninstall"
            if want_cache
            else (
                f"- cache: left in place (add **`and cache`** or say "
                f"**`confirm uninstall {plan.app.name} and cache`** to delete caches too)"
            )
        )
        return True, (
            f"Ready to uninstall:\n{plan.summary()}\n{cache_line}\n\n"
            f"Say **`confirm uninstall {plan.app.name}`** to proceed, or "
            f"**`confirm uninstall {plan.app.name} and cache`** to also wipe its caches "
            f"(or set `LITO_UNINSTALL_YES=1`)."
        )

    # Execute
    if plan.method in {
        "winget",
        "windows",
        "windows-registry",
        "windows-settings",
    }:
        ok, msg = _run_windows_uninstall(plan)
        if ok:
            try:
                registry.reload()
            except Exception:
                pass
            if plan.method == "windows-settings" or plan.command.startswith("start "):
                return True, (
                    f"Opened Windows Apps & Features for **{plan.app.name}**.\n"
                    f"{plan.detail}\n```\n{msg}\n```"
                )
            body = (
                f"Uninstalled **{plan.app.name}** via `{plan.method}`.\n"
                f"```\n{msg}\n```"
            )
            if want_cache:
                body += "\n\n**App cache cleanup**\n" + _clear_owner_cache(plan.app.name)
            elif clear_cache is not False:
                body += (
                    f"\n\nCaches kept. Say `clear cache for {plan.app.name}` "
                    f"if you want them gone."
                )
            return True, body
        return False, (
            f"Uninstall of **{plan.app.name}** failed (`{plan.method}`).\n"
            f"Command: `{plan.command}`\n```\n{msg}\n```\n"
            f"You may need admin rights, or the package may be a suite "
            f"(e.g. Kleopatra → **Gpg4win**)."
        )

    ok, msg = _run(plan.command)
    if ok:
        try:
            registry.reload()
        except Exception:
            pass
        body = f"Uninstalled **{plan.app.name}** via `{plan.method}`.\n```\n{msg}\n```"
        if want_cache:
            body += "\n\n**App cache cleanup**\n" + _clear_owner_cache(plan.app.name)
        elif clear_cache is not False:
            body += (
                f"\n\nCaches kept. Say `clear cache for {plan.app.name}` "
                f"if you want them gone."
            )
        return True, body
    return False, (
        f"Uninstall of **{plan.app.name}** failed (`{plan.method}`).\n"
        f"Command: `{plan.command}`\n```\n{msg}\n```\n"
        f"You may need sudo/admin rights."
    )
