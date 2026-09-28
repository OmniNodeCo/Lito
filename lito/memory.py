"""Tiny persistent memory (notes + key/value). File-backed, few KB."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .config import data_dir, ensure_data_dir

_MEM = "memory.json"
_MAX_NOTES = 200
_MAX_KV = 500


def _path() -> Path:
    return data_dir() / _MEM


def _load() -> dict[str, Any]:
    p = _path()
    if not p.exists():
        return {"kv": {}, "notes": []}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"kv": {}, "notes": []}
        data.setdefault("kv", {})
        data.setdefault("notes", [])
        return data
    except (OSError, json.JSONDecodeError):
        return {"kv": {}, "notes": []}


def _save(data: dict[str, Any]) -> None:
    ensure_data_dir()
    p = _path()
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def remember(key: str, value: str) -> None:
    data = _load()
    kv = data.setdefault("kv", {})
    kv[key.strip().lower()] = {"value": value, "ts": time.time()}
    while len(kv) > _MAX_KV:
        oldest = min(kv.items(), key=lambda kv_i: float(kv_i[1].get("ts") or 0))
        kv.pop(oldest[0], None)
    _save(data)


def recall(key: str) -> str | None:
    data = _load()
    item = data.get("kv", {}).get(key.strip().lower())
    if isinstance(item, dict):
        return str(item.get("value", ""))
    if isinstance(item, str):
        return item
    return None


def forget(key: str) -> bool:
    data = _load()
    kv = data.get("kv", {})
    if key.strip().lower() in kv:
        del kv[key.strip().lower()]
        _save(data)
        return True
    return False


def note(text: str) -> None:
    data = _load()
    notes = data.setdefault("notes", [])
    notes.append({"text": text.strip(), "ts": time.time()})
    if len(notes) > _MAX_NOTES:
        data["notes"] = notes[-_MAX_NOTES:]
    _save(data)


def list_notes(limit: int = 20) -> list[dict[str, Any]]:
    notes = _load().get("notes") or []
    return list(reversed(notes[-limit:]))


def all_kv() -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in (_load().get("kv") or {}).items():
        if isinstance(v, dict):
            out[k] = str(v.get("value", ""))
        else:
            out[k] = str(v)
    return out


def search_memory(query: str, limit: int = 8) -> list[str]:
    q = query.lower().strip()
    hits: list[str] = []
    for k, v in all_kv().items():
        if q in k or q in v.lower():
            hits.append(f"{k} = {v}")
        if len(hits) >= limit:
            return hits
    for n in list_notes(50):
        t = n.get("text") or ""
        if q in t.lower():
            hits.append(f"note: {t}")
        if len(hits) >= limit:
            break
    return hits
