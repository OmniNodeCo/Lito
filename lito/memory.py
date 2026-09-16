"""Ultra-light persistent memory (JSON). Avoids any DB dependency."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .config import (
    ensure_data_dir,
    history_path,
    memory_path,
    notes_path,
)


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def _write_json(path: Path, data: Any) -> None:
    ensure_data_dir()
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    # Windows-friendly replace
    try:
        tmp.replace(path)
    except OSError:
        if path.exists():
            path.unlink(missing_ok=True)
        tmp.rename(path)


# --- key/value memory -------------------------------------------------------

def remember(key: str, value: str) -> None:
    data = _read_json(memory_path(), {})
    if not isinstance(data, dict):
        data = {}
    data[key.strip().lower()] = {"value": value, "ts": time.time()}
    _write_json(memory_path(), data)


def recall(key: str) -> str | None:
    data = _read_json(memory_path(), {})
    if not isinstance(data, dict):
        return None
    item = data.get(key.strip().lower())
    if isinstance(item, dict):
        return str(item.get("value", ""))
    if isinstance(item, str):
        return item
    return None


def forget(key: str) -> bool:
    data = _read_json(memory_path(), {})
    if not isinstance(data, dict):
        return False
    k = key.strip().lower()
    if k in data:
        del data[k]
        _write_json(memory_path(), data)
        return True
    return False


def list_memory() -> dict[str, str]:
    data = _read_json(memory_path(), {})
    if not isinstance(data, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in data.items():
        if isinstance(v, dict):
            out[k] = str(v.get("value", ""))
        else:
            out[k] = str(v)
    return out


# --- notes ------------------------------------------------------------------

def add_note(text: str) -> dict[str, Any]:
    notes = _read_json(notes_path(), [])
    if not isinstance(notes, list):
        notes = []
    note = {"id": int(time.time() * 1000) % 10_000_000, "text": text.strip(), "ts": time.time()}
    notes.insert(0, note)
    notes = notes[:200]
    _write_json(notes_path(), notes)
    return note


def list_notes(limit: int = 20) -> list[dict[str, Any]]:
    notes = _read_json(notes_path(), [])
    if not isinstance(notes, list):
        return []
    return notes[:limit]


def clear_notes() -> int:
    notes = _read_json(notes_path(), [])
    n = len(notes) if isinstance(notes, list) else 0
    _write_json(notes_path(), [])
    return n


# --- chat history (append-only, capped) -------------------------------------

def append_history(role: str, text: str, max_lines: int = 200) -> None:
    ensure_data_dir()
    path = history_path()
    line = json.dumps({"ts": time.time(), "role": role, "text": text}, ensure_ascii=False)
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        return
    try:
        if path.stat().st_size > 200_000:
            with path.open("r", encoding="utf-8") as f:
                lines = f.readlines()[-max_lines:]
            with path.open("w", encoding="utf-8") as f:
                f.writelines(lines)
    except OSError:
        pass


def recent_history(limit: int = 40) -> list[dict[str, Any]]:
    path = history_path()
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            lines = f.readlines()[-limit:]
    except OSError:
        return []
    out: list[dict[str, Any]] = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out
