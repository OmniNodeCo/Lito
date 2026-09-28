"""Lito-Nano: ultra-small neural LM (pure Python, no torch/numpy).

Architecture (chosen for speed + agent skill, not parameter flexing)
-------------------------------------------------------------------
* Token embedding Wte [V, D]
* Positional embedding Wpe [T, D]
* Context = concat of last K token states (or pad) → R^{K·D}
* 2-layer MLP:  (K·D) → H → H → V   with GELU
* Causal next-token training

Default ~80–150k weights, f32 blob < 1 MB, inference < 20 ms/token on CPU.
Still a real neural language model — not rules, not n-gram tables.
"""

from __future__ import annotations

import json
import math
import random
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, List, Sequence

from . import tensor as T


@dataclass
class ModelConfig:
    vocab_size: int = 512
    n_embd: int = 48
    n_hidden: int = 96
    n_ctx: int = 4  # last K tokens fed to MLP
    block_size: int = 64
    # kept for API compat with older docs
    n_head: int = 1
    n_layer: int = 2


class NanoLM:
    def __init__(self, config: ModelConfig):
        self.config = config
        V, D, H, K = config.vocab_size, config.n_embd, config.n_hidden, config.n_ctx
        scale = 0.02
        self.wte = T.randn((V, D), scale=scale)
        self.wpe = T.randn((config.block_size, D), scale=0.01)
        in_dim = K * D
        self.fc1 = T.xavier(in_dim, H)  # [in, H]
        self.b1 = [0.0] * H
        self.fc2 = T.xavier(H, H)
        self.b2 = [0.0] * H
        self.head = T.xavier(H, V)  # [H, V]
        self.hb = [0.0] * V

    def param_count(self) -> int:
        def n(obj) -> int:
            if isinstance(obj, float):
                return 1
            if isinstance(obj, list):
                if not obj:
                    return 0
                if isinstance(obj[0], list):
                    return sum(len(r) for r in obj)
                return len(obj)
            return 0

        return (
            n(self.wte)
            + n(self.wpe)
            + n(self.fc1)
            + n(self.b1)
            + n(self.fc2)
            + n(self.b2)
            + n(self.head)
            + n(self.hb)
        )

    def _tok_vec(self, tok: int, pos: int) -> List[float]:
        V, B = self.config.vocab_size, self.config.block_size
        if tok < 0 or tok >= V:
            tok = 0
        if pos < 0:
            pos = 0
        if pos >= B:
            pos = B - 1
        te, pe = self.wte[tok], self.wpe[pos]
        return [a + b for a, b in zip(te, pe)]

    def _context_vec(self, idx: Sequence[int], end: int) -> List[float]:
        """Build concat of last K token vectors ending at position end-1 inclusive."""
        K, D = self.config.n_ctx, self.config.n_embd
        start = max(0, end - K)
        vecs = []
        for pos in range(start, end):
            vecs.extend(self._tok_vec(idx[pos], pos))
        # left-pad
        need = K * D - len(vecs)
        if need > 0:
            vecs = [0.0] * need + vecs
        return vecs

    def _mlp(self, x: Sequence[float]) -> tuple[List[float], List[float], List[float]]:
        """Return logits, h1, h2 for backprop."""
        # h1 = gelu(x @ fc1 + b1)
        h1_pre = [
            sum(x[i] * self.fc1[i][j] for i in range(len(x))) + self.b1[j]
            for j in range(len(self.b1))
        ]
        h1 = [T.gelu(v) for v in h1_pre]
        h2_pre = [
            sum(h1[i] * self.fc2[i][j] for i in range(len(h1))) + self.b2[j]
            for j in range(len(self.b2))
        ]
        h2 = [T.gelu(v) for v in h2_pre]
        logits = [
            sum(h2[i] * self.head[i][j] for i in range(len(h2))) + self.hb[j]
            for j in range(len(self.hb))
        ]
        return logits, h1, h2

    def logits_at(self, idx: Sequence[int], end: int) -> List[float]:
        x = self._context_vec(idx, end)
        logits, _, _ = self._mlp(x)
        return logits

    def forward(self, idx: Sequence[int]) -> List[List[float]]:
        """Logits for predicting tokens at positions 1..T given 0..T-1 context ends."""
        # returns length T logits where logits[t] predicts idx[t] from idx[:t]
        # (caller usually pairs with targets idx[1:])
        out = []
        for end in range(1, len(idx) + 1):
            out.append(self.logits_at(idx, end))
        return out

    def logits_last(self, idx: Sequence[int]) -> List[float]:
        return self.logits_at(idx, len(idx))

    # ---- serialization ----
    def _walk_params(self, fn) -> None:
        fn(self.wte)
        fn(self.wpe)
        fn(self.fc1)
        fn(self.b1)
        fn(self.fc2)
        fn(self.b2)
        fn(self.head)
        fn(self.hb)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        floats: list[float] = []

        def collect(obj) -> None:
            if isinstance(obj, list):
                if obj and isinstance(obj[0], list):
                    for r in obj:
                        floats.extend(float(x) for x in r)
                else:
                    floats.extend(float(x) for x in obj)

        self._walk_params(collect)
        meta = {"config": asdict(self.config), "format": "lito-nano-mlp-f32-v2"}
        blob = struct.pack(f"<{len(floats)}f", *floats)
        path.write_bytes(json.dumps(meta).encode("utf-8") + b"\n\n" + blob)

    @classmethod
    def load(cls, path: Path) -> "NanoLM":
        raw = path.read_bytes()
        sep = raw.find(b"\n\n")
        meta = json.loads(raw[:sep].decode("utf-8"))
        cfg = ModelConfig(**{k: v for k, v in meta["config"].items() if k in ModelConfig.__dataclass_fields__})
        model = cls(cfg)
        blob = raw[sep + 2 :]
        n = len(blob) // 4
        floats = list(struct.unpack(f"<{n}f", blob[: n * 4]))
        it = iter(floats)

        def fill(obj) -> None:
            if isinstance(obj, list):
                if obj and isinstance(obj[0], list):
                    for row in obj:
                        for i in range(len(row)):
                            row[i] = next(it)
                else:
                    for i in range(len(obj)):
                        obj[i] = next(it)

        model._walk_params(fill)
        return model


def generate(
    model: NanoLM,
    idx: List[int],
    *,
    max_new: int = 48,
    temperature: float = 0.8,
    top_k: int = 20,
    eos_id: int | None = None,
) -> List[int]:
    out = list(idx)
    for _ in range(max_new):
        ctx = out[-model.config.block_size :]
        logits = model.logits_last(ctx)
        if temperature <= 0:
            next_id = max(range(len(logits)), key=lambda i: logits[i])
        else:
            scaled = [l / max(temperature, 1e-6) for l in logits]
            if top_k and top_k < len(scaled):
                top = sorted(range(len(scaled)), key=lambda i: scaled[i], reverse=True)[:top_k]
                keep = set(top)
                scaled = [scaled[i] if i in keep else -1e9 for i in range(len(scaled))]
            probs = T.softmax(scaled)
            r = random.random()
            cum = 0.0
            next_id = len(probs) - 1
            for i, p in enumerate(probs):
                cum += p
                if r <= cum:
                    next_id = i
                    break
        out.append(next_id)
        if eos_id is not None and next_id == eos_id:
            break
    return out
