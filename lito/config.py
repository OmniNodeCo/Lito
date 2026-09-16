"""Tiny config loader. Defaults keep RAM near zero."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

DEFAULTS: dict[str, Any] = {
    "host": "127.0.0.1",
    "port": 8765,
    "max_history": 200,
    "safe_shell": True,
    "theme": "dark",
    "name": "Lito",
    "confirm_dangerous": True,
}


def data_dir() -> Path:
    """Resolve data dir each call so LITO_DATA env changes (tests) apply."""
    return Path(os.environ.get("LITO_DATA", Path.home() / ".lito"))


# Back-compat aliases (evaluated lazily via __getattr__)
def config_path() -> Path:
    return data_dir() / "config.json"


def memory_path() -> Path:
    return data_dir() / "memory.json"


def notes_path() -> Path:
    return data_dir() / "notes.json"


def history_path() -> Path:
    return data_dir() / "history.jsonl"


def ensure_data_dir() -> Path:
    d = data_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_config() -> dict[str, Any]:
    ensure_data_dir()
    cfg = dict(DEFAULTS)
    path = config_path()
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                user = json.load(f)
            if isinstance(user, dict):
                cfg.update(user)
        except (OSError, json.JSONDecodeError):
            pass
    return cfg


def save_config(cfg: dict[str, Any]) -> None:
    ensure_data_dir()
    with config_path().open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


# Module-level names some callers still read; keep as properties via __getattr__
def __getattr__(name: str) -> Any:
    mapping = {
        "DATA_DIR": data_dir,
        "CONFIG_PATH": config_path,
        "MEMORY_PATH": memory_path,
        "NOTES_PATH": notes_path,
        "HISTORY_PATH": history_path,
    }
    if name in mapping:
        return mapping[name]()
    raise AttributeError(name)
