"""
Version information for SmartAI.

The AI carries two version numbers:

- The application version (e.g. 1.1.0) - the software itself, shown at
  startup, in chat and via `python main.py --version`.
- The knowledge version - starts at the application version and bumps its
  patch number every time the AI learns a new word or term from the web
  (see src/dictionary.py), e.g. 1.1.0 -> 1.1.1 -> 1.1.2.
"""

__version__ = '1.2.1'
APP_NAME = 'SmartAI'


def parse_version(version: str):
    """Parse 'v1.2.3' into (1, 2, 3). Missing parts count as 0."""
    parts = str(version).lstrip('vV').split('.')
    numbers = []
    for part in parts[:3]:
        try:
            numbers.append(int(part))
        except (TypeError, ValueError):
            numbers.append(0)
    while len(numbers) < 3:
        numbers.append(0)
    return tuple(numbers)


def format_version(version: str = '') -> str:
    """Format a version string as 'v1.2.3' (defaults to the app version)."""
    return 'v' + '.'.join(str(p) for p in parse_version(version or __version__))


def bump_patch(version: str) -> str:
    """1.2.3 -> 1.2.4"""
    major, minor, patch = parse_version(version)
    return f'{major}.{minor}.{patch + 1}'


def bump_minor(version: str) -> str:
    """1.2.3 -> 1.3.0"""
    major, minor, _patch = parse_version(version)
    return f'{major}.{minor + 1}.0'
