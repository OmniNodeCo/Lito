"""Tiny agent tokenizer — small vocab for a micro LM."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

SPECIAL = [
    "<pad>",
    "<bos>",
    "<eos>",
    "<unk>",
    "<think>",
    "</think>",
    "<tool>",
    "</tool>",
    "<obs>",
    "</obs>",
    "<ans>",
    "</ans>",
    "<user>",
    "<bot>",
]

SEED_WORDS = """
help hello hi hey thanks yes no ok
what who when where why how is are be
the a an to of in on for with from as by or and not
i you we my your
can will would should
calculate calc compute plus minus times
search look up find explain define tell about
remember recall note notes memory
open run shell read time date sysinfo status ram
answer think use tool result
wifi project city color port name value key
mqtt photosynthesis walrus gravity recursion quantum tcp udp
ada lovelace plant energy light protocol messaging
lito nano agent ai model
true false error done blocked
echo hello world example https http
file report pdf
""".split()

DIGITS = list("0123456789")
OPS = list("+-*/^=.,:?!")


class Tokenizer:
    def __init__(self, stoi: dict[str, int] | None = None):
        if stoi is None:
            stoi = self._build_default()
        self.stoi = dict(stoi)
        self.itos = {i: s for s, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)
        self.pad_id = self.stoi["<pad>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]
        self.unk_id = self.stoi["<unk>"]

    @staticmethod
    def _build_default() -> dict[str, int]:
        words: list[str] = []
        seen = set()
        for w in SPECIAL + SEED_WORDS + DIGITS + OPS:
            w = w if w.startswith("<") else w.lower()
            if w and w not in seen:
                seen.add(w)
                words.append(w)
        # limited byte fallback (printable ascii)
        for i in range(32, 127):
            tok = f"<b{i}>"
            if tok not in seen:
                seen.add(tok)
                words.append(tok)
        return {w: i for i, w in enumerate(words)}

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"stoi": self.stoi}, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Tokenizer":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(data["stoi"])

    _token_re = re.compile(r"(</?[a-z]+>|https?://\S+|[\w']+|[^\s\w])", re.I)

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        for raw in self._token_re.findall(text):
            low = raw if raw.startswith("<") else raw.lower()
            if low in self.stoi:
                ids.append(self.stoi[low])
                continue
            if raw in self.stoi:
                ids.append(self.stoi[raw])
                continue
            for ch in raw:
                b = ord(ch) if 32 <= ord(ch) < 127 else 63
                ids.append(self.stoi.get(f"<b{b}>", self.unk_id))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        parts: list[str] = []
        byte_buf = bytearray()

        def flush() -> None:
            nonlocal byte_buf
            if byte_buf:
                parts.append(byte_buf.decode("utf-8", errors="replace"))
                byte_buf = bytearray()

        for i in ids:
            tok = self.itos.get(int(i), "<unk>")
            if tok.startswith("<b") and tok.endswith(">") and tok[2:-1].isdigit():
                byte_buf.append(int(tok[2:-1]))
                continue
            flush()
            if tok in {"<pad>", "<bos>", "<eos>", "<unk>"}:
                continue
            if tok.startswith("<") and tok.endswith(">"):
                parts.append(tok)
                continue
            if parts and not parts[-1].endswith(">") and re.match(r"^[^\w\s]+$", tok):
                parts.append(tok)
            elif parts and parts[-1].endswith(">"):
                parts.append(("" if tok.startswith("<") else " ") + tok)
            else:
                parts.append(("" if not parts else " ") + tok if not re.match(r"^[^\w\s]+$", tok) else tok)
        flush()
        s = "".join(parts)
        s = re.sub(r"\s+", " ", s).strip()
        for tag in ("think", "tool", "obs", "ans"):
            s = s.replace(f"<{tag}>", f"\n<{tag}>").replace(f"</{tag}>", f"</{tag}>\n")
        return s.strip()

    def extend_from_corpus(self, texts: Iterable[str], max_extra: int = 120) -> None:
        c: Counter[str] = Counter()
        for t in texts:
            for raw in self._token_re.findall(t):
                low = raw if raw.startswith("<") else raw.lower()
                if low not in self.stoi and not low.startswith("<b"):
                    c[low] += 1
        added = 0
        for w, _n in c.most_common():
            if added >= max_extra:
                break
            if w not in self.stoi and 1 <= len(w) <= 18:
                self.stoi[w] = len(self.stoi)
                added += 1
        self.itos = {i: s for s, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)
