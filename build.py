#!/usr/bin/env python3
"""
Build script to create executables using PyInstaller.
Reads configuration from build.yml.

Builds every target listed in build.yml (console chat app + GUI app),
falling back to the flat single-target keys for older configs.
"""

import subprocess
import sys
import os


def _parse_value(value: str):
    """Parse a scalar YAML-ish value: bool, list, quoted or plain string."""
    value = value.strip()
    if value.lower() == 'true':
        return True
    if value.lower() == 'false':
        return False
    if value.startswith('[') and value.endswith(']'):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [v.strip().strip("'\"") for v in inner.split(',')]
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]
    return value


def load_build_config():
    """Load build configuration from build.yml (simple parser, no pyyaml).

    Flat ``key: value`` lines work as before. A ``targets:`` block lists
    several executables to build:

        targets:
          - name: LitoEngine
            entry_point: server.py
            console: true
    """
    config = {
        'name': 'Lito',
        'entry_point': 'server.py',
        'icon': None,
        'onefile': True,
        'console': True,
        'extra_data': ['checkpoints'],
        'hidden_imports': ['numpy', 'json', 're'],
        'targets': [],
    }

    if not os.path.exists('build.yml'):
        return config

    with open('build.yml', 'r') as f:
        lines = f.read().splitlines()

    targets = []
    current_target = None
    in_targets = False

    for raw in lines:
        stripped = raw.strip()
        if not stripped or stripped.startswith('#'):
            continue
        indented = raw[:1] in (' ', '\t')

        if in_targets:
            if stripped.startswith('- '):
                if current_target is not None:
                    targets.append(current_target)
                current_target = {}
                item = stripped[2:].strip()
                if ':' in item:
                    key, value = item.split(':', 1)
                    current_target[key.strip()] = _parse_value(value)
                continue
            if indented:
                if current_target is not None and ':' in stripped:
                    key, value = stripped.split(':', 1)
                    current_target[key.strip()] = _parse_value(value)
                continue
            # First unindented line after the list ends the block.
            in_targets = False
            if current_target is not None:
                targets.append(current_target)
                current_target = None

        if ':' not in stripped:
            continue
        key, value = stripped.split(':', 1)
        key = key.strip()
        if key == 'targets':
            in_targets = True
            continue
        config[key] = _parse_value(value)

    if current_target is not None:
        targets.append(current_target)
    if targets:
        config['targets'] = [t for t in targets if t]
    return config


def _build_target(config: dict, target: dict):
    """Build one executable.

    Returns (status, exe_path) with status in {'built', 'skipped', 'failed'}.
    """
    name = str(target.get('name') or config.get('name', 'Lito'))
    entry = str(target.get('entry_point') or config.get('entry_point', 'server.py'))
    console = target.get('console', config.get('console', True))
    onefile = target.get('onefile', config.get('onefile', True))
    icon = target.get('icon', config.get('icon'))
    extra_data = target.get('extra_data', config.get('extra_data', ['checkpoints']))
    hidden_imports = target.get('hidden_imports',
                                config.get('hidden_imports', []))

    kind = 'console' if console else 'windowed'
    print(f"\nBuilding: {name} ({kind}) from {entry}")

    cmd = [
        sys.executable, '-m', 'PyInstaller',
        '--name', name,
        '--noconfirm',
        '--clean',
    ]

    if onefile:
        cmd.append('--onefile')

    cmd.append('--console' if console else '--windowed')

    if icon and os.path.exists(str(icon)):
        cmd.extend(['--icon', str(icon)])

    # Add data directories listed in extra_data (src, checkpoints, data,
    # extensions, ...) - each is bundled with the executable
    for data_dir in extra_data:
        data_dir = str(data_dir)
        if os.path.exists(data_dir) and os.path.isdir(data_dir):
            cmd.extend(['--add-data', f'{data_dir}{os.pathsep}{data_dir}'])

    # Hidden imports
    for imp in hidden_imports:
        cmd.extend(['--hidden-import', str(imp)])

    # Add the entry point
    cmd.append(entry)

    print(f"  Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        exe = os.path.join('dist', name + ('.exe' if sys.platform == 'win32' else ''))
        print(f"  OK: {exe}")
        return 'built', exe

    print(f"  FAILED: {name} (PyInstaller exit code {result.returncode})")
    return 'failed', None


def build():
    """Build every configured executable."""
    print("=" * 50)
    print("  Lito Build System")
    print("=" * 50)

    # Check PyInstaller
    try:
        import PyInstaller
        print(f"PyInstaller version: {PyInstaller.__version__}")
    except ImportError:
        print("Installing PyInstaller...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyinstaller'])

    config = load_build_config()

    targets = config.get('targets') or []
    if not targets:
        # Older flat config: a single target from the top-level keys.
        targets = [{
            'name': config.get('name', 'Lito'),
            'entry_point': config.get('entry_point', 'server.py'),
            'console': config.get('console', True),
        }]

    built, skipped, failed = [], [], []
    for target in targets:
        status, exe = _build_target(config, target)
        if status == 'built':
            built.append(exe)
        elif status == 'skipped':
            skipped.append(target.get('name'))
        else:
            failed.append(target.get('name'))

    print("\n" + "=" * 50)
    print("  BUILD SUMMARY")
    for exe in built:
        print(f"  Built:     {exe}")
    for name in skipped:
        print(f"  Skipped:   {name}")
    for name in failed:
        print(f"  FAILED:    {name}")
    if built:
        print("  Output directory: dist/")
        print("  Engine: dist/LitoEngine/ (onedir)")
        print("  GUI: build the Go desktop app instead (cd desktop && go build)")
    print("=" * 50)

    if failed or not built:
        sys.exit(1)


if __name__ == '__main__':
    build()
