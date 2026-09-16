"""Uninstall installed apps safely by owner type.

Resolves the app via the registry, detects package manager / source, then
runs the matching uninstall path. Requires confirm=True for actual removal
unless LITO_UNINSTALL_YES=1 (for automation).
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .apps import AppEntry, registry


@dataclass
class UninstallPlan:
    app: AppEntry
    method: str  # flatpak | snap | apt | dnf | pacman | brew | macos | windows | unknown
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


def _run(cmd: list[str] | str, timeout: float = 120.0) -> tuple[bool, str]:
    try:
        if isinstance(cmd, str):
            proc = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=timeout
            )
        else:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"Timed out after {timeout}s"
    except OSError as exc:
        return False, str(exc)
    out = ((proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")).strip()
    return proc.returncode == 0, out[:3000] or f"(exit {proc.returncode})"


def _flatpak_id(app: AppEntry) -> str | None:
    if app.command.startswith("flatpak run "):
        return app.command.split("flatpak run ", 1)[1].strip().split()[0]
    if app.description and re.match(r"^[\w.-]+\.[\w.-]+", app.description):
        return app.description.split()[0]
    # aliases may hold app id
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
        # "pkg: /usr/bin/foo"
        pkg = out.split(":", 1)[0].strip()
        return pkg or None
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


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
        # Homebrew cask guess
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
        # Prefer winget if present
        if shutil.which("winget"):
            return (
                UninstallPlan(
                    app=app,
                    method="winget",
                    command=f'winget uninstall --name "{app.name}" --silent',
                    detail="Windows Package Manager",
                    risky=True,
                ),
                "",
            )
        return (
            UninstallPlan(
                app=app,
                method="windows",
                command=f'start ms-settings:appsfeatures',
                detail="Open Windows Apps & Features to uninstall manually",
                risky=False,
            ),
            "",
        )

    # Linux native packages
    binary = cmd.split()[0].strip("\"'")
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


def uninstall_app(name: str, *, confirm: bool = False) -> tuple[bool, str]:
    """Plan or execute uninstall. confirm=False -> dry description only."""
    plan, err = plan_uninstall(name)
    if err:
        return False, err
    assert plan is not None

    if plan.method == "unknown" or not plan.command:
        return False, plan.summary()

    auto = os.environ.get("LITO_UNINSTALL_YES", "").strip() in {"1", "true", "yes"}
    if not confirm and not auto:
        return True, (
            f"Ready to uninstall:\n{plan.summary()}\n\n"
            f"Say **`confirm uninstall {plan.app.name}`** to proceed "
            f"(or set `LITO_UNINSTALL_YES=1`)."
        )

    # Execute
    if plan.method == "windows" and plan.command.startswith("start "):
        ok, msg = _run(plan.command)
        return ok, f"Opened uninstall UI for **{plan.app.name}**.\n{msg}"

    ok, msg = _run(plan.command)
    if ok:
        # Drop from registry cache so list apps updates
        try:
            registry.reload()
        except Exception:
            pass
        return True, f"Uninstalled **{plan.app.name}** via `{plan.method}`.\n```\n{msg}\n```"
    return False, (
        f"Uninstall of **{plan.app.name}** failed (`{plan.method}`).\n"
        f"Command: `{plan.command}`\n```\n{msg}\n```\n"
        f"You may need sudo/admin rights."
    )
