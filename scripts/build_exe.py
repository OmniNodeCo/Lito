#!/usr/bin/env python3
"""Build a standalone Lito executable with PyInstaller.

Usage:
  python scripts/build_exe.py
  python scripts/build_exe.py --onefile

Output lands in dist/ — e.g. dist/lito or dist/lito.exe
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Lito executable")
    parser.add_argument("--onefile", action="store_true", default=True, help="Single-file binary (default)")
    parser.add_argument("--onedir", action="store_true", help="Folder bundle instead of one-file")
    parser.add_argument("--name", default="lito", help="Output binary name")
    parser.add_argument("--clean", action="store_true", help="Wipe build/ and dist/ first")
    args = parser.parse_args()
    onefile = not args.onedir

    os.chdir(ROOT)
    if args.clean:
        for d in ("build", "dist"):
            p = ROOT / d
            if p.exists():
                shutil.rmtree(p)

    # Ensure pyinstaller
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("Installing pyinstaller…")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller>=6.0"])

    entry = ROOT / "run_lito.py"
    static = ROOT / "lito" / "static"
    sep = ";" if platform.system() == "Windows" else ":"
    add_data = f"{static}{sep}lito/static"

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--name",
        args.name,
        "--paths",
        str(ROOT),
        f"--add-data={add_data}",
        "--hidden-import=lito",
        "--hidden-import=lito.brain",
        "--hidden-import=lito.actions",
        "--hidden-import=lito.apps",
        "--hidden-import=lito.cache",
        "--hidden-import=lito.update",
        "--hidden-import=lito.ui",
        "--hidden-import=lito.cli",
        "--hidden-import=lito.memory",
        "--hidden-import=lito.config",
        "--collect-submodules=lito",
    ]
    if onefile:
        cmd.append("--onefile")
    else:
        cmd.append("--onedir")

    # Console app (CLI + server logs)
    cmd.append("--console")
    cmd.append(str(entry))

    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)

    dist = ROOT / "dist"
    built = list(dist.glob(f"{args.name}*"))
    print("Artifacts:")
    for b in built:
        size = b.stat().st_size if b.is_file() else sum(f.stat().st_size for f in b.rglob("*") if f.is_file())
        print(f"  {b}  ({size / 1024 / 1024:.1f} MB)")

    # Write a small platform stamp next to the binary
    tag = _platform_tag()
    stamp = dist / "build-info.txt"
    from lito import __version__

    stamp.write_text(
        f"name={args.name}\nversion={__version__}\nplatform={tag}\nsystem={platform.system()}\n"
        f"machine={platform.machine()}\n",
        encoding="utf-8",
    )
    print(f"Platform tag: {tag}")
    print("Done.")
    return 0


def _platform_tag() -> str:
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


if __name__ == "__main__":
    raise SystemExit(main())
