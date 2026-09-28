#!/usr/bin/env python3
"""Build a one-file Lito executable with PyInstaller."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    spec = ROOT / "lito.spec"
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", str(spec)]
    print(" ".join(cmd))
    return subprocess.call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
