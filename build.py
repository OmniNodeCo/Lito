#!/usr/bin/env python3
"""
Build script to create executable using PyInstaller.
Reads configuration from build.yml.
"""

import subprocess
import sys
import os
import shutil


def load_build_config():
    """Load build configuration from build.yml (simple parser, no pyyaml needed)."""
    config = {
        'name': 'SmartAI',
        'entry_point': 'main.py',
        'icon': None,
        'onefile': True,
        'console': True,
        'extra_data': ['checkpoints'],
        'hidden_imports': ['numpy', 'json', 're'],
    }

    if os.path.exists('build.yml'):
        with open('build.yml', 'r') as f:
            for line in f:
                line = line.strip()
                if ':' in line and not line.startswith('#'):
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    if value.lower() == 'true':
                        value = True
                    elif value.lower() == 'false':
                        value = False
                    elif value.startswith('['):
                        # Simple list parsing
                        value = [v.strip().strip("'\"")
                                for v in value.strip('[]').split(',')]
                    config[key] = value

    return config


def build():
    """Build the executable."""
    print("=" * 50)
    print("  SmartAI Build System")
    print("=" * 50)

    # Check PyInstaller
    try:
        import PyInstaller
        print(f"PyInstaller version: {PyInstaller.__version__}")
    except ImportError:
        print("Installing PyInstaller...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyinstaller'])

    config = load_build_config()
    print(f"Building: {config['name']}")
    print(f"Entry point: {config['entry_point']}")

    # Build PyInstaller command
    cmd = [
        sys.executable, '-m', 'PyInstaller',
        '--name', str(config['name']),
        '--noconfirm',
        '--clean',
    ]

    if config.get('onefile', True):
        cmd.append('--onefile')

    if config.get('console', True):
        cmd.append('--console')
    else:
        cmd.append('--windowed')

    if config.get('icon') and os.path.exists(str(config['icon'])):
        cmd.extend(['--icon', str(config['icon'])])

    # Add data directories listed in extra_data (src, checkpoints, dictionary
    # data, ...) - each is bundled next to the executable
    for data_dir in config.get('extra_data', ['checkpoints', 'src']):
        data_dir = str(data_dir)
        if os.path.exists(data_dir) and os.path.isdir(data_dir):
            cmd.extend(['--add-data', f'{data_dir}{os.pathsep}{data_dir}'])

    # Hidden imports
    for imp in config.get('hidden_imports', []):
        cmd.extend(['--hidden-import', imp])

    # Add the entry point
    cmd.append(config['entry_point'])

    print(f"\nRunning: {' '.join(cmd)}\n")

    # Execute build
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        print("\n" + "=" * 50)
        print("  BUILD SUCCESSFUL!")
        print(f"  Executable: dist/{config['name']}")
        if sys.platform == 'win32':
            print(f"  Run: dist\\{config['name']}.exe")
        else:
            print(f"  Run: dist/{config['name']}")
        print("=" * 50)
    else:
        print("\n  BUILD FAILED!")
        sys.exit(1)


if __name__ == '__main__':
    build()