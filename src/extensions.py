"""
Optional extensions (plugins) for Lito.

An extension is a small Python module that lives in the `extensions/`
directory. Each one may provide:

    NAME, VERSION, DESCRIPTION        required metadata
    COMMANDS = {'/cmd': 'help text'}  slash commands it handles
    can_handle(text) -> bool          passive trigger ("roll a dice")
    handle(text, brain) -> str        produce a reply for a trigger
    handle_command(cmd, args, brain)  reply for one of its slash commands

Extensions are optional at runtime: they can be enabled and disabled
(`/extensions off dice`, or via the GUI), the state persists to
`data/extensions.json`, and a broken extension is skipped instead of
crashing the chat.
"""

import importlib.util
import json
import os
import re
import sys
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ExtensionInfo:
    """A loaded extension module plus its metadata."""
    name: str
    version: str
    description: str
    module: object
    commands: Dict[str, str] = field(default_factory=dict)
    path: str = ''


class ExtensionManager:
    """Discovers, loads and toggles optional extensions."""

    def __init__(self, extensions_dir: Optional[str] = None,
                 config_file: Optional[str] = None):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.extensions_dir = extensions_dir or os.path.join(root, 'extensions')
        self.config_file = (config_file or
                            os.path.join(root, 'data', 'extensions.json'))
        self._extensions: List[ExtensionInfo] = []
        self._enabled: Dict[str, bool] = {}
        self._load_config()
        self.discover()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_config(self) -> None:
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._enabled = {str(k): bool(v)
                             for k, v in data.get('enabled', {}).items()}
        except (OSError, ValueError):
            self._enabled = {}

    def _save_config(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump({'enabled': self._enabled}, f, indent=2)
        except OSError:
            pass

    def _load_module(self, path: str):
        """Import a single extension file safely."""
        name = 'lito_ext_' + re.sub(r'\W+', '_', os.path.basename(path)[:-3])
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    def discover(self) -> List[ExtensionInfo]:
        """(Re)load every extension in the extensions directory."""
        self._extensions = []
        if not os.path.isdir(self.extensions_dir):
            return self._extensions
        for fname in sorted(os.listdir(self.extensions_dir)):
            if not fname.endswith('.py') or fname.startswith('_'):
                continue
            path = os.path.join(self.extensions_dir, fname)
            try:
                module = self._load_module(path)
                ext_name = str(getattr(module, 'NAME', '')).strip()
                if not ext_name:
                    continue
                self._extensions.append(ExtensionInfo(
                    name=ext_name,
                    version=str(getattr(module, 'VERSION', '1.0.0')),
                    description=str(getattr(module, 'DESCRIPTION', '')),
                    module=module,
                    commands=dict(getattr(module, 'COMMANDS', {}) or {}),
                    path=path,
                ))
            except Exception:
                print(f"[extensions] failed to load {fname}:")
                traceback.print_exc()
        return self._extensions

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def extensions(self) -> List[ExtensionInfo]:
        return list(self._extensions)

    def get(self, name: str) -> Optional[ExtensionInfo]:
        low = (name or '').lower().strip()
        for ext in self._extensions:
            if ext.name.lower() == low:
                return ext
        return None

    def is_enabled(self, name: str) -> bool:
        ext = self.get(name)
        if ext is None:
            return False
        return self._enabled.get(ext.name, True)

    def set_enabled(self, name: str, enabled: bool) -> bool:
        ext = self.get(name)
        if ext is None:
            return False
        self._enabled[ext.name] = bool(enabled)
        self._save_config()
        return True

    def command_help(self):
        """Yield (command, help_text, extension_name) for all commands."""
        for ext in self._extensions:
            for cmd, help_text in ext.commands.items():
                yield cmd, help_text, ext.name

    # ------------------------------------------------------------------
    # Answering
    # ------------------------------------------------------------------

    def try_handle(self, text: str, brain) -> str:
        """
        Give every enabled extension a chance to answer `text`.
        Returns the first reply, or '' when no extension wants it.
        """
        text = (text or '').strip()
        if not text or text.startswith('/'):
            return ''
        for ext in self._extensions:
            if not self.is_enabled(ext.name):
                continue
            can = getattr(ext.module, 'can_handle', None)
            handler = getattr(ext.module, 'handle', None)
            if not callable(can) or not callable(handler):
                continue
            try:
                if not can(text):
                    continue
                reply = handler(text, brain)
            except Exception:
                print(f"[extensions] {ext.name} errored:")
                traceback.print_exc()
                continue
            if reply and str(reply).strip():
                return str(reply).strip()
        return ''

    def handle_command(self, cmd: str, args: str, brain) -> str:
        """Route a slash command to the extension that owns it."""
        cmd = (cmd or '').lower().strip()
        for ext in self._extensions:
            if cmd not in ext.commands:
                continue
            handler = getattr(ext.module, 'handle_command', None)
            if not callable(handler):
                return f"The {ext.name} extension doesn't handle {cmd}."
            if not self.is_enabled(ext.name):
                return (f"The {ext.name} extension is disabled. "
                        f"Enable it with: /extensions on {ext.name}")
            try:
                return str(handler(cmd, args, brain) or
                           f"{cmd} returned nothing.")
            except Exception as e:
                return f"{cmd} failed: {e}"
        return ''
