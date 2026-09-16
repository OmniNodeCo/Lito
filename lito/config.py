"""Tiny config loader. Defaults keep RAM near zero."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# Home for user data (notes, memory, custom apps)
DATA_DIR = Path(os.environ.get("LITO_DATA", Path.home() / ".lito"))
CONFIG_PATH = DATA_DIR / "config.json"
MEMORY_PATH = DATA_DIR / "memory.json"
NOTES_PATH = DATA_DIR / "notes.json"
HISTORY_PATH = DATA_DIR / "history.jsonl"

DEFAULTS: dict[str, Any] = {
    "host": "127.0.0.1",
    "port": 8765,
    "max_history": 200,
    "safe_shell": True,
    "theme": "dark",
    "name": "Lito",
    "confirm_dangerous": True,
}


def ensure_data_dir() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def load_config() -> dict[str, Any]:
    ensure_data_dir()
    cfg = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        try:
            with CONFIG_PATH.open("r", encoding="utf-8") as f:
                user = json.load(f)
            if isinstance(user, dict):
                cfg.update(user)
        except (OSError, json.JSONDecodeError):
            pass
    return cfg


def save_config(cfg: dict[str, Any]) -> None:
    ensure_data_dir()
    with CONFIG_PATH.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
