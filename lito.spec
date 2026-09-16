# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for Lito — used by CI and local builds.
#   pyinstaller lito.spec

import platform
from pathlib import Path

block_cipher = None
root = Path(SPECPATH)
static = root / "lito" / "static"

system = platform.system().lower()
machine = platform.machine().lower()
if machine in ("x86_64", "amd64"):
    arch = "x86_64"
elif machine in ("aarch64", "arm64"):
    arch = "arm64"
else:
    arch = machine
if system == "darwin":
    plat = f"macos-{arch}"
elif system == "windows":
    plat = f"windows-{arch}"
else:
    plat = f"linux-{arch}"

exe_name = f"lito-{plat}"

a = Analysis(
    ["run_lito.py"],
    pathex=[str(root)],
    binaries=[],
    datas=[(str(static), "lito/static")],
    hiddenimports=[
        "lito",
        "lito.brain",
        "lito.actions",
        "lito.apps",
        "lito.cache",
        "lito.update",
        "lito.ui",
        "lito.cli",
        "lito.memory",
        "lito.config",
        "lito.__main__",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib",
        "numpy",
        "pandas",
        "torch",
        "tensorflow",
        "PIL",
        "test",
        "unittest",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name=exe_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
