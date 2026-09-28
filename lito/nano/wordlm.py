"""Conditional word-level LM: P(answer_word | user_ctx, prev_words).

Trained only on natural dialogue — generates free-form replies.
"""

from __future__ import annotations

import json
import math
import random
import re
import struct
from dataclasses import dataclass
from pathlib import Path

from .dialogue import chat_pairs


def tok(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+|[.!?]", text.lower())


@dataclass
class WConfig:
    n_hash: int = 192
    n_emb: int = 32
    n_hidden: int = 64
    ctx: int = 3  # previous answer words
    max_vocab: int = 900


class WordLM:
    """Tiny conditional generator.

    user_bag (hashed) + sum(prev word embeds) → MLP → vocab logits
    """

    def __init__(self, cfg: WConfig, stoi: dict[str, int] | None = None):
        self.cfg = cfg
        self.stoi = stoi or {"<pad>": 0, "<unk>": 1, "<eos>": 2}
        self.itos = {i: s for s, i in self.stoi.items()}
        V = max(len(self.stoi), 8)
        H, E, B = cfg.n_hidden, cfg.n_emb, cfg.n_hash
        self.V = V
        # user bag projection B→E
        self.Uw = [[random.gauss(0, 0.08) for _ in range(E)] for _ in range(B)]
        # word embeds
        self.We = [[random.gauss(0, 0.08) for _ in range(E)] for _ in range(V)]
        # MLP: concat(user_e, ctx_e) = 2E → H → V
        self.W1 = [[random.gauss(0, 0.05) for _ in range(H)] for _ in range(2 * E)]
        self.b1 = [0.0] * H
        self.W2 = [[random.gauss(0, 0.05) for _ in range(V)] for _ in range(H)]
        self.b2 = [0.0] * V

    def ensure_vocab(self, words: list[str]) -> None:
        changed = False
        for w in words:
            if w not in self.stoi and len(self.stoi) < self.cfg.max_vocab:
                self.stoi[w] = len(self.stoi)
                changed = True
        if changed:
            self.itos = {i: s for s, i in self.stoi.items()}
            self._resize()

    def _resize(self) -> None:
        V = len(self.stoi)
        E, H = self.cfg.n_emb, self.cfg.n_hidden
        while len(self.We) < V:
            self.We.append([random.gauss(0, 0.08) for _ in range(E)])
        # W2 out dim
        for row in self.W2:
            while len(row) < V:
                row.append(random.gauss(0, 0.05))
        while len(self.b2) < V:
            self.b2.append(0.0)
        self.V = V

    def param_count(self) -> int:
        return (
            sum(len(r) for r in self.Uw)
            + sum(len(r) for r in self.We)
            + sum(len(r) for r in self.W1)
            + len(self.b1)
            + sum(len(r) for r in self.W2)
            + len(self.b2)
        )

    def _hash(self, s: str) -> int:
        h = 2166136261
        for c in s.encode():
            h ^= c
            h = (h * 16777619) & 0xFFFFFFFF
        return h % self.cfg.n_hash

    def user_vec(self, user: str) -> list[float]:
        B, E = self.cfg.n_hash, self.cfg.n_emb
        bag = [0.0] * B
        for w in tok(user):
            bag[self._hash(w)] += 1.0
        if re.search(r"\d", user):
            bag[self._hash("#NUM")] += 1.5
        n = math.sqrt(sum(x * x for x in bag)) or 1.0
        bag = [x / n for x in bag]
        e = [0.0] * E
        for i, v in enumerate(bag):
            if not v:
                continue
            for j in range(E):
                e[j] += v * self.Uw[i][j]
        return e

    def wid(self, w: str) -> int:
        return self.stoi.get(w, self.stoi["<unk>"])

    def ctx_vec(self, prev: list[int]) -> list[float]:
        E = self.cfg.n_emb
        e = [0.0] * E
        use = prev[-self.cfg.ctx :]
        if not use:
            return e
        for wid in use:
            we = self.We[wid if wid < len(self.We) else 1]
            for j in range(E):
                e[j] += we[j]
        inv = 1.0 / len(use)
        return [x * inv for x in e]

    def logits(self, user_e: list[float], ctx_e: list[float]) -> list[float]:
        E, H, V = self.cfg.n_emb, self.cfg.n_hidden, self.V
        x = user_e + ctx_e
        h = [0.0] * H
        for j in range(H):
            s = self.b1[j]
            for i in range(2 * E):
                s += x[i] * self.W1[i][j]
            h[j] = math.tanh(s)
        out = [0.0] * V
        for j in range(V):
            s = self.b2[j]
            for i in range(H):
                s += h[i] * self.W2[i][j]
            out[j] = s
        return out, h, x

    def train_pair(self, user: str, answer: str, lr: float = 0.12) -> float:
        words = tok(answer) + ["<eos>"]
        self.ensure_vocab(tok(user) + words)
        self._resize()
        user_e = self.user_vec(user)
        prev: list[int] = []
        total = 0.0
        n = 0
        E, H, V = self.cfg.n_emb, self.cfg.n_hidden, self.V
        for w in words:
            target = self.wid(w)
            ctx_e = self.ctx_vec(prev)
            logits, h, x = self.logits(user_e, ctx_e)
            # softmax
            m = max(logits)
            ex = [math.exp(v - m) for v in logits]
            z = sum(ex)
            probs = [e / z for e in ex]
            total += -math.log(max(probs[target], 1e-12))
            n += 1
            dlog = probs
            dlog[target] -= 1.0
            # backprop W2/b2
            dh = [0.0] * H
            for j in range(V):
                g = dlog[j]
                self.b2[j] -= lr * g
                for i in range(H):
                    self.W2[i][j] -= lr * h[i] * g
                    dh[i] += self.W2[i][j] * g  # approx after update
            # fix dh with pre-update: recompute quickly
            dh = [0.0] * H
            for j in range(V):
                g = dlog[j]
                for i in range(H):
                    dh[i] += (self.W2[i][j] + lr * h[i] * g) * g  # messy
            # cleaner: store W2 copy — skip, use current
            dh = [0.0] * H
            for j in range(V):
                g = dlog[j]
                for i in range(H):
                    dh[i] += self.W2[i][j] * g
            dh_pre = [dh[i] * (1 - h[i] * h[i]) for i in range(H)]
            dx = [0.0] * (2 * E)
            for j in range(H):
                g = dh_pre[j]
                self.b1[j] -= lr * g
                for i in range(2 * E):
                    self.W1[i][j] -= lr * x[i] * g
                    dx[i] += self.W1[i][j] * g
            # ctx embeds (second half of x)
            if prev:
                use = prev[-self.cfg.ctx :]
                inv = lr / len(use)
                for wid in use:
                    for j in range(E):
                        self.We[wid][j] -= inv * dx[E + j]
            prev.append(target)
            if len(prev) > 12:
                prev = prev[-12:]
        return total / max(n, 1)

    def generate(self, user: str, *, max_words: int = 40, temperature: float = 0.85) -> str:
        self._resize()
        user_e = self.user_vec(user)
        prev: list[int] = []
        out: list[str] = []
        eos = self.stoi["<eos>"]
        for _ in range(max_words):
            ctx_e = self.ctx_vec(prev)
            logits, _, _ = self.logits(user_e, ctx_e)
            # temperature sample
            if temperature < 0.05:
                wid = max(range(len(logits)), key=lambda i: logits[i])
            else:
                scaled = [v / temperature for v in logits]
                # top-k
                k = min(40, len(scaled))
                top = sorted(range(len(scaled)), key=lambda i: scaled[i], reverse=True)[:k]
                keep = set(top)
                scaled = [scaled[i] if i in keep else -1e9 for i in range(len(scaled))]
                m = max(scaled)
                ex = [math.exp(v - m) for v in scaled]
                z = sum(ex)
                r = random.random()
                cum = 0.0
                wid = top[-1]
                for i in top:
                    cum += ex[i] / z
                    if r <= cum:
                        wid = i
                        break
            if wid == eos:
                break
            w = self.itos.get(wid, "")
            if not w or w.startswith("<"):
                continue
            out.append(w)
            prev.append(wid)
            # stop at sentence end after a few words
            if w in ".!?" and len(out) >= 6:
                break
        # detokenize
        s = " ".join(out)
        s = re.sub(r"\s+([.!?])", r"\1", s)
        s = s[:1].upper() + s[1:] if s else s
        return s

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"format": "lito-wordlm-v1", "stoi": self.stoi, "cfg": self.cfg.__dict__}
        floats: list[float] = []
        for row in self.Uw:
            floats.extend(row)
        for row in self.We:
            floats.extend(row)
        for row in self.W1:
            floats.extend(row)
        floats.extend(self.b1)
        for row in self.W2:
            floats.extend(row)
        floats.extend(self.b2)
        path.write_bytes(json.dumps(meta).encode() + b"\n\n" + struct.pack(f"<{len(floats)}f", *floats))

    @classmethod
    def load(cls, path: Path) -> "WordLM":
        raw = path.read_bytes()
        sep = raw.find(b"\n\n")
        meta = json.loads(raw[:sep].decode())
        cfg = WConfig(**{k: v for k, v in meta["cfg"].items() if k in WConfig.__dataclass_fields__})
        lm = cls(cfg, stoi=meta["stoi"])
        lm._resize()
        blob = raw[sep + 2 :]
        n = len(blob) // 4
        floats = list(struct.unpack(f"<{n}f", blob[: n * 4]))
        it = iter(floats)
        E, H, B, V = cfg.n_emb, cfg.n_hidden, cfg.n_hash, lm.V
        for i in range(B):
            for j in range(E):
                lm.Uw[i][j] = next(it)
        for i in range(V):
            for j in range(E):
                lm.We[i][j] = next(it)
        for i in range(2 * E):
            for j in range(H):
                lm.W1[i][j] = next(it)
        for j in range(H):
            lm.b1[j] = next(it)
        for i in range(H):
            for j in range(V):
                lm.W2[i][j] = next(it)
        for j in range(V):
            lm.b2[j] = next(it)
        return lm


def train_wordlm(
    steps: int = 4000,
    lr: float = 0.15,
    out: Path | None = None,
) -> WordLM:
    out = out or Path(__file__).resolve().parent / "weights" / "lito-wordlm.bin"
    plain, grounded = chat_pairs()
    pairs = list(plain)
    for u, obs, a in grounded:
        pairs.append((u, a))
        # also train with obs folded into user context
        pairs.append((f"{u} [info {obs}]", a))
    # expand
    pairs = pairs + pairs
    random.shuffle(pairs)
    print(f"wordlm pairs {len(pairs)}", flush=True)

    lm = WordLM(WConfig())
    # seed vocab
    bag = []
    for u, a in pairs:
        bag.extend(tok(u))
        bag.extend(tok(a))
    lm.ensure_vocab(bag)
    lm._resize()
    print(f"vocab {len(lm.stoi)} params {lm.param_count():,}", flush=True)

    avg = 0.0
    for step in range(1, steps + 1):
        u, a = random.choice(pairs)
        cur = lr * (1.0 - 0.75 * step / steps)
        loss = lm.train_pair(u, a, lr=max(cur, lr * 0.05))
        avg = 0.9 * avg + 0.1 * loss if step > 1 else loss
        if step % 500 == 0 or step == 1:
            print(f"step {step}/{steps} loss={loss:.3f} avg={avg:.3f}", flush=True)
            for q in ("hello", "what is mqtt", "thanks", "who are you"):
                print("  ", q, "→", lm.generate(q, temperature=0.7), flush=True)

    lm.save(out)
    print(f"saved {out}", flush=True)
    return lm
