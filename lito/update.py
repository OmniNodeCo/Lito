"""Check GitHub Releases for newer Lito builds and install updates.

Designed for frozen (PyInstaller) executables and source installs.
Uses only the stdlib - no updater daemon, minimal RAM.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import __version__
from .config import DATA_DIR, ensure_data_dir

# Override for forks / enterprise
DEFAULT_REPO = os.environ.get("LITO_GITHUB_REPO", "OmniNodeCo/Lito")
API_LATEST = "https://api.github.com/repos/{repo}/releases/latest"
# Stable JSON pointer published on every release (also attached as asset)
LATEST_JSON_URL = "https://github.com/{repo}/releases/latest/download/latest.json"


@dataclass
class ReleaseInfo:
    version: str
    tag: str
    notes: str
    html_url: str
    asset_name: str | None
    asset_url: str | None
    asset_digest: str | None  # sha256 hex if known
    published_at: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "tag": self.tag,
            "notes": self.notes,
            "html_url": self.html_url,
            "asset_name": self.asset_name,
            "asset_url": self.asset_url,
            "asset_digest": self.asset_digest,
            "published_at": self.published_at,
            "current_version": __version__,
        }


def _parse_version(v: str) -> tuple[int, ...]:
    v = v.strip().lstrip("vV")
    parts = re.findall(r"\d+", v)
    return tuple(int(p) for p in parts) if parts else (0,)


def is_newer(remote: str, local: str | None = None) -> bool:
    local = local or __version__
    return _parse_version(remote) > _parse_version(local)


def current_platform_tag() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64"}:
        arch = "x86_64"
    elif machine in {"aarch64", "arm64"}:
        arch = "arm64"
    else:
        arch = machine
    if system == "darwin":
        return f"macos-{arch}"
    if system == "windows":
        return f"windows-{arch}"
    return f"linux-{arch}"


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False)) or hasattr(sys, "_MEIPASS")


def executable_path() -> Path:
    if is_frozen():
        return Path(sys.executable).resolve()
    return Path(sys.argv[0]).resolve()


def _http_json(url: str, timeout: float = 20.0) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": f"Lito/{__version__} (+https://github.com/{DEFAULT_REPO})",
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _pick_asset(assets: list[dict[str, Any]], plat: str | None = None) -> dict[str, Any] | None:
    plat = plat or current_platform_tag()
    # Prefer names containing platform tag
    scored: list[tuple[int, dict[str, Any]]] = []
    for a in assets:
        name = str(a.get("name") or "").lower()
        if name in {"latest.json", "checksums.txt", "sha256sums.txt"}:
            continue
        if not any(name.endswith(ext) for ext in (".zip", ".tar.gz", ".tgz", ".exe", ".appimage", "")):
            # allow extensionless unix binaries named lito-*
            if "lito" not in name:
                continue
        score = 0
        if plat.replace("-", "_") in name or plat in name:
            score += 100
        # loose matching
        sys_token = plat.split("-")[0]  # linux/macos/windows
        if sys_token in name:
            score += 40
        arch = plat.split("-")[-1]
        if arch in name or (arch == "x86_64" and "amd64" in name) or (arch == "arm64" and "aarch64" in name):
            score += 30
        if name.startswith("lito"):
            score += 10
        if score:
            scored.append((score, a))
    if not scored:
        return None
    scored.sort(key=lambda x: -x[0])
    return scored[0][1]


def check_for_update(
    *,
    repo: str | None = None,
    current: str | None = None,
) -> ReleaseInfo | None:
    """Return remote release if newer than current, else None.

    Tries latest.json asset first (fast, CDN), then GitHub API.
    """
    repo = repo or DEFAULT_REPO
    current = current or __version__

    info = _from_latest_json(repo)
    if info is None:
        info = _from_api(repo)
    if info is None:
        return None
    if not is_newer(info.version, current):
        return None
    return info


def _from_latest_json(repo: str) -> ReleaseInfo | None:
    url = LATEST_JSON_URL.format(repo=repo)
    try:
        data = _http_json(url)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    version = str(data.get("version") or data.get("tag_name") or "").lstrip("v")
    if not version:
        return None
    assets = data.get("assets") or {}
    plat = current_platform_tag()
    asset = None
    if isinstance(assets, dict):
        asset = assets.get(plat) or assets.get(plat.replace("x86_64", "amd64"))
    asset_name = asset_url = asset_digest = None
    if isinstance(asset, dict):
        asset_name = asset.get("name")
        asset_url = asset.get("url") or asset.get("browser_download_url")
        asset_digest = asset.get("sha256")
    elif isinstance(asset, str):
        asset_url = asset
        asset_name = asset.rsplit("/", 1)[-1]
    return ReleaseInfo(
        version=version,
        tag=str(data.get("tag_name") or f"v{version}"),
        notes=str(data.get("notes") or data.get("body") or "")[:2000],
        html_url=str(data.get("html_url") or f"https://github.com/{repo}/releases/tag/v{version}"),
        asset_name=asset_name,
        asset_url=asset_url,
        asset_digest=asset_digest,
        published_at=str(data.get("published_at") or ""),
    )


def _from_api(repo: str) -> ReleaseInfo | None:
    url = API_LATEST.format(repo=repo)
    try:
        data = _http_json(url)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, OSError):
        return None
    tag = str(data.get("tag_name") or "")
    version = tag.lstrip("vV")
    if not version:
        return None
    assets = data.get("assets") or []
    picked = _pick_asset(assets) if isinstance(assets, list) else None
    asset_name = asset_url = asset_digest = None
    if picked:
        asset_name = picked.get("name")
        asset_url = picked.get("browser_download_url")
        # GitHub may provide digest in API (newer)
        digest = picked.get("digest") or ""
        if isinstance(digest, str) and digest.startswith("sha256:"):
            asset_digest = digest.split(":", 1)[1]
    return ReleaseInfo(
        version=version,
        tag=tag,
        notes=str(data.get("body") or "")[:2000],
        html_url=str(data.get("html_url") or ""),
        asset_name=asset_name,
        asset_url=asset_url,
        asset_digest=asset_digest,
        published_at=str(data.get("published_at") or ""),
    )


def _download(url: str, dest: Path, expected_sha256: str | None = None) -> None:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": f"Lito/{__version__}", "Accept": "application/octet-stream"},
    )
    h = hashlib.sha256()
    with urllib.request.urlopen(req, timeout=120) as resp, dest.open("wb") as out:
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            h.update(chunk)
            out.write(chunk)
    if expected_sha256 and h.hexdigest().lower() != expected_sha256.lower():
        dest.unlink(missing_ok=True)
        raise ValueError(
            f"SHA256 mismatch for download (got {h.hexdigest()}, expected {expected_sha256})"
        )


def apply_update(info: ReleaseInfo, *, restart: bool = False) -> tuple[bool, str]:
    """Download release asset and replace the running frozen binary when possible."""
    if not info.asset_url:
        return (
            False,
            f"Update **{info.version}** is available but no binary asset matched "
            f"`{current_platform_tag()}`. Open {info.html_url}",
        )

    ensure_data_dir()
    updates_dir = DATA_DIR / "updates"
    updates_dir.mkdir(parents=True, exist_ok=True)

    name = info.asset_name or Path(info.asset_url).name or "lito-update"
    dest = updates_dir / name

    try:
        _download(info.asset_url, dest, expected_sha256=info.asset_digest)
    except Exception as exc:  # noqa: BLE001
        return False, f"Download failed: {exc}"

    # Unpack if archive
    payload = dest
    try:
        if name.endswith(".zip"):
            import zipfile

            extract_dir = updates_dir / f"extract-{info.version}"
            if extract_dir.exists():
                shutil.rmtree(extract_dir, ignore_errors=True)
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(dest, "r") as zf:
                zf.extractall(extract_dir)
            payload = _find_binary(extract_dir)
        elif name.endswith((".tar.gz", ".tgz")):
            import tarfile

            extract_dir = updates_dir / f"extract-{info.version}"
            if extract_dir.exists():
                shutil.rmtree(extract_dir, ignore_errors=True)
            extract_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(dest, "r:gz") as tf:
                tf.extractall(extract_dir)
            payload = _find_binary(extract_dir)
    except Exception as exc:  # noqa: BLE001
        return False, f"Unpack failed: {exc}"

    if payload is None or not payload.exists():
        return False, f"Downloaded to `{dest}` but couldn't locate a Lito binary inside."

    if not is_frozen():
        return (
            True,
            f"Downloaded **{info.version}** -> `{payload}`.\n"
            f"This install is source/pip (not a frozen exe). "
            f"Replace manually or: `pip install -U .` / grab the binary from {info.html_url}",
        )

    target = executable_path()
    backup = target.with_suffix(target.suffix + ".bak")
    try:
        # On Windows, can't overwrite running exe - stage a .new and a helper script
        if platform.system() == "Windows":
            staged = target.with_suffix(target.suffix + ".new")
            shutil.copy2(payload, staged)
            helper = DATA_DIR / "apply_update.bat"
            helper.write_text(
                "\r\n".join(
                    [
                        "@echo off",
                        "timeout /t 2 /nobreak >nul",
                        f'move /Y "{staged}" "{target}"',
                        f'start "" "{target}"',
                        "del %~f0",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            msg = (
                f"Staged **{info.version}**. Restart Lito to finish "
                f"(or run `{helper}`)."
            )
            if restart:
                os.spawnl(os.P_NOWAIT, os.environ.get("COMSPEC", "cmd.exe"), "cmd.exe", "/c", str(helper))
                msg += "\nRestarting..."
            return True, msg

        # POSIX: replace binary atomically when possible
        tmp = target.with_suffix(target.suffix + ".new")
        shutil.copy2(payload, tmp)
        os.chmod(tmp, 0o755)
        if target.exists():
            shutil.copy2(target, backup)
        os.replace(tmp, target)
        msg = f"Updated to **{info.version}**. Backup: `{backup}`."
        if restart:
            os.execv(str(target), [str(target)] + sys.argv[1:])
        return True, msg + " Restart Lito to run the new build."
    except OSError as exc:
        return False, f"Could not replace binary: {exc}. File saved at `{payload}`."


def _find_binary(root: Path) -> Path | None:
    candidates: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        n = p.name.lower()
        if n in {"lito", "lito.exe"} or (n.startswith("lito") and p.stat().st_mode & 0o111):
            candidates.append(p)
        elif n.endswith(".exe") and "lito" in n:
            candidates.append(p)
    if not candidates:
        # any executable
        for p in root.rglob("*"):
            if p.is_file() and os.access(p, os.X_OK) and "lito" in p.name.lower():
                candidates.append(p)
    if not candidates:
        return None
    candidates.sort(key=lambda p: (0 if p.name.lower() in {"lito", "lito.exe"} else 1, len(str(p))))
    return candidates[0]


def format_update_status(info: ReleaseInfo | None) -> str:
    if info is None:
        return (
            f"You're on **{__version__}** ({current_platform_tag()}) - up to date "
            f"(or GitHub unreachable)."
        )
    asset = info.asset_name or "(no matching binary)"
    notes = (info.notes or "").strip()
    if len(notes) > 400:
        notes = notes[:400] + "..."
    lines = [
        f"**Update available:** {__version__} -> **{info.version}**",
        f"Platform: `{current_platform_tag()}` - asset: `{asset}`",
        f"Release: {info.html_url}",
    ]
    if notes:
        lines.append(f"\n{notes}")
    lines.append("\nSay **`install update`** to download and apply, or **`open release page`**.")
    return "\n".join(lines)


def auto_check_enabled() -> bool:
    flag = os.environ.get("LITO_AUTO_UPDATE", "1").strip().lower()
    return flag not in {"0", "false", "no", "off"}


def write_last_check(info: ReleaseInfo | None) -> None:
    ensure_data_dir()
    path = DATA_DIR / "update_check.json"
    payload = {
        "checked_at": __import__("time").time(),
        "current": __version__,
        "available": info.as_dict() if info else None,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
