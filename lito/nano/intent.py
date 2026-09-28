"""Neural intent classifier — tiny supervised MLP (the smart router).

Hashes tokens into a fixed bag + character n-grams → MLP → tool logits.
Trained on labeled curriculum. ~5–15k weights. Pure Python.
"""

from __future__ import annotations

import json
import math
import random
import re
import struct
from dataclasses import dataclass
from pathlib import Path

TOOLS = [
    "chat",
    "help",
    "calc",
    "search",
    "remember",
    "recall",
    "time",
    "sysinfo",
    "open",
    "shell",
    "note",
    "find",
]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+|https?|[\+\-\*\/\^]", text.lower())


def _hash(s: str, buckets: int) -> int:
    # FNV-1a style
    h = 2166136261
    for c in s.encode("utf-8", errors="ignore"):
        h ^= c
        h = (h * 16777619) & 0xFFFFFFFF
    return h % buckets


@dataclass
class IntentConfig:
    n_buckets: int = 256
    n_hidden: int = 48
    n_classes: int = len(TOOLS)


class IntentNet:
    def __init__(self, cfg: IntentConfig | None = None):
        self.cfg = cfg or IntentConfig()
        B, H, C = self.cfg.n_buckets, self.cfg.n_hidden, self.cfg.n_classes
        scale = 0.05
        self.W1 = [[random.gauss(0, scale) for _ in range(H)] for _ in range(B)]
        self.b1 = [0.0] * H
        self.W2 = [[random.gauss(0, scale) for _ in range(C)] for _ in range(H)]
        self.b2 = [0.0] * C

    def param_count(self) -> int:
        return (
            self.cfg.n_buckets * self.cfg.n_hidden
            + self.cfg.n_hidden
            + self.cfg.n_hidden * self.cfg.n_classes
            + self.cfg.n_classes
        )

    def embed(self, text: str) -> list[float]:
        B = self.cfg.n_buckets
        v = [0.0] * B
        toks = _tokens(text)
        if not toks:
            toks = ["empty"]
        for t in toks:
            v[_hash(t, B)] += 1.0
            if len(t) >= 3:
                v[_hash("#" + t[:3], B)] += 0.5
            if len(t) >= 4:
                v[_hash(t[-3:], B)] += 0.35
        # digit / op signals
        if re.search(r"\d", text):
            v[_hash("HASNUM", B)] += 2.0
        if re.search(r"[\+\-\*\/\^]", text):
            v[_hash("HASOP", B)] += 2.0
        if "?" in text:
            v[_hash("HASQ", B)] += 1.5
        if text.lower().startswith(("what", "who", "why", "how", "where", "when")):
            v[_hash("WH", B)] += 1.5
        # L2 normalize
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def forward(self, text: str) -> list[float]:
        x = self.embed(text)
        H = self.cfg.n_hidden
        C = self.cfg.n_classes
        h = [0.0] * H
        for j in range(H):
            s = self.b1[j]
            for i, xv in enumerate(x):
                s += xv * self.W1[i][j]
            h[j] = math.tanh(s)
        logits = [0.0] * C
        for j in range(C):
            s = self.b2[j]
            for i in range(H):
                s += h[i] * self.W2[i][j]
            logits[j] = s
        return logits

    def predict(self, text: str) -> tuple[str, list[float]]:
        logits = self.forward(text)
        m = max(logits)
        ex = [math.exp(v - m) for v in logits]
        z = sum(ex)
        probs = [e / z for e in ex]
        i = max(range(len(probs)), key=lambda k: probs[k])
        return TOOLS[i], probs

    def train_step(self, text: str, label: str, lr: float = 0.2) -> float:
        if label not in TOOLS:
            return 0.0
        y = TOOLS.index(label)
        x = self.embed(text)
        H, C, B = self.cfg.n_hidden, self.cfg.n_classes, self.cfg.n_buckets
        # forward with cache
        h_pre = [0.0] * H
        for j in range(H):
            s = self.b1[j]
            for i in range(B):
                s += x[i] * self.W1[i][j]
            h_pre[j] = s
        h = [math.tanh(v) for v in h_pre]
        logits = [0.0] * C
        for j in range(C):
            s = self.b2[j]
            for i in range(H):
                s += h[i] * self.W2[i][j]
            logits[j] = s
        m = max(logits)
        ex = [math.exp(v - m) for v in logits]
        z = sum(ex)
        probs = [e / z for e in ex]
        loss = -math.log(max(probs[y], 1e-12))
        # dlogits
        dlogits = list(probs)
        dlogits[y] -= 1.0
        # grads for W2/b2 and dh using *current* weights
        dh = [0.0] * H
        gW2 = [[0.0] * C for _ in range(H)]
        gb2 = [0.0] * C
        for j in range(C):
            g = dlogits[j]
            gb2[j] = g
            for i in range(H):
                gW2[i][j] = h[i] * g
                dh[i] += self.W2[i][j] * g
        dh_pre = [dh[i] * (1.0 - h[i] * h[i]) for i in range(H)]
        gW1 = [[0.0] * H for _ in range(B)]
        gb1 = [0.0] * H
        for j in range(H):
            g = dh_pre[j]
            gb1[j] = g
            for i in range(B):
                gW1[i][j] = x[i] * g
        # apply
        for j in range(C):
            self.b2[j] -= lr * gb2[j]
            for i in range(H):
                self.W2[i][j] -= lr * gW2[i][j]
        for j in range(H):
            self.b1[j] -= lr * gb1[j]
            for i in range(B):
                self.W1[i][j] -= lr * gW1[i][j]
        return loss

    def save(self, path: Path) -> None:
        floats: list[float] = []
        for row in self.W1:
            floats.extend(row)
        floats.extend(self.b1)
        for row in self.W2:
            floats.extend(row)
        floats.extend(self.b2)
        meta = {
            "format": "lito-intent-v1",
            "config": {
                "n_buckets": self.cfg.n_buckets,
                "n_hidden": self.cfg.n_hidden,
                "n_classes": self.cfg.n_classes,
            },
            "tools": TOOLS,
        }
        path.write_bytes(json.dumps(meta).encode() + b"\n\n" + struct.pack(f"<{len(floats)}f", *floats))

    @classmethod
    def load(cls, path: Path) -> "IntentNet":
        raw = path.read_bytes()
        sep = raw.find(b"\n\n")
        meta = json.loads(raw[:sep].decode())
        cfg = IntentConfig(**meta["config"])
        net = cls(cfg)
        floats = list(struct.unpack(f"<{len(raw[sep+2:])//4}f", raw[sep + 2 : sep + 2 + 4 * (len(raw[sep+2:]) // 4)]))
        it = iter(floats)
        B, H, C = cfg.n_buckets, cfg.n_hidden, cfg.n_classes
        for i in range(B):
            for j in range(H):
                net.W1[i][j] = next(it)
        for j in range(H):
            net.b1[j] = next(it)
        for i in range(H):
            for j in range(C):
                net.W2[i][j] = next(it)
        for j in range(C):
            net.b2[j] = next(it)
        return net


def build_labeled_data() -> list[tuple[str, str]]:
    data: list[tuple[str, str]] = []

    def add(label: str, phrases: list[str], n: int = 1) -> None:
        for p in phrases:
            for _ in range(n):
                data.append((p, label))

    add(
        "chat",
        [
            "hello", "hi", "hey", "hello there", "hi lito", "yo", "thanks", "thank you",
            "good morning", "hey there", "good evening", "howdy", "hiya", "sup",
            "hello friend", "hey lito", "hi there", "greetings",
        ],
        8,
    )
    add(
        "help",
        [
            "help", "what can you do", "commands", "how do i use you", "help me",
            "list tools", "show help", "assist me", "usage", "capabilities",
            "what do you support", "help please",
        ],
        8,
    )
    add("time", ["time", "what time is it", "what's the time", "date", "current time", "tell me the time"], 3)
    add("sysinfo", ["sysinfo", "system info", "status", "how much ram", "free ram", "memory usage", "whoami system"], 3)
    # calc
    calcs = []
    for a in range(0, 15):
        for b in range(0, 12):
            calcs.append(f"calculate {a}+{b}")
            calcs.append(f"calc {a}*{b}")
            calcs.append(f"what is {a} plus {b}")
            calcs.append(f"{a}+{b}")
            calcs.append(f"compute {a}-{b}")
    calcs += ["calculate 2^10", "calc 6*7", "2^8", "solve 12*12", "what is 100/4"]
    add("calc", calcs, 1)
    # search
    searches = [
        "what is mqtt",
        "what is photosynthesis",
        "what is a walrus",
        "who is ada lovelace",
        "explain gravity",
        "define recursion",
        "search web quantum computing",
        "look up tcp vs udp",
        "tell me about black holes",
        "what is python",
        "how does wifi work",
        "why is the sky blue",
        "what is photosynthesis really",
        "search mqtt qos",
        "google latest news",
    ]
    add("search", searches, 3)
    # memory
    for k, v in [("wifi", "home"), ("project", "lito"), ("city", "miami"), ("port", "8765"), ("color", "green")]:
        add("remember", [f"remember {k} is {v}", f"remember that {k} is {v}", f"remember {k} = {v}"], 3)
        add("recall", [f"recall {k}", f"what is my {k}", f"remind me of {k}", f"what's my {k}"], 3)
    add(
        "open",
        [
            "open https://example.com",
            "open firefox",
            "launch code",
            "start terminal",
            "open ~/Documents",
            "open the browser",
            "launch spotify",
            "start calculator",
        ],
        4,
    )
    # "tell me about X" must be search, never open
    add(
        "search",
        [
            "tell me about the ocean",
            "tell me about music",
            "tell me about ai",
            "tell me about love",
            "tell me about python",
            "tell me about gravity",
        ],
        5,
    )
    add("shell", ["run echo hello", "shell ls", "run date", "exec uname -a"], 3)
    add("note", ["note buy milk", "note call mom", "jot meeting at 3", "take a note ship it"], 3)
    add("find", ["find file report.pdf", "find notes.txt", "locate readme", "find file budget"], 3)
    # keep soft chat from looking like recall
    add(
        "chat",
        [
            "i failed today",
            "i am stressed",
            "motivate me",
            "i am bored",
            "i am sad",
            "i am happy",
            "tell me a story",
            "give me advice",
            "how do i stay productive",
        ],
        6,
    )
    random.shuffle(data)
    return data


def train_intent(
    steps: int = 2500,
    lr: float = 0.25,
    out: Path | None = None,
) -> IntentNet:
    out = out or Path(__file__).resolve().parent / "weights" / "lito-intent.bin"
    data = build_labeled_data()
    net = IntentNet(IntentConfig(n_buckets=256, n_hidden=48))
    print(f"intent data {len(data)}  params {net.param_count():,}", flush=True)
    avg = 0.0
    for step in range(1, steps + 1):
        text, label = random.choice(data)
        # mild LR decay
        cur_lr = lr * (1.0 - 0.7 * step / steps)
        loss = net.train_step(text, label, lr=cur_lr)
        avg = 0.95 * avg + 0.05 * loss if step > 1 else loss
        if step % 500 == 0 or step == 1:
            # accuracy probe
            ok = 0
            probe = data[:80]
            for t, y in probe:
                pred, _ = net.predict(t)
                ok += int(pred == y)
            acc = ok / len(probe)
            print(f"step {step}/{steps} loss={loss:.3f} avg={avg:.3f} acc@80={acc:.2f}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    net.save(out)
    print(f"saved {out}", flush=True)
    # show confusions
    for t in ["hello", "calculate 3+4", "what is mqtt", "remember wifi is x", "recall wifi", "help", "time"]:
        p, probs = net.predict(t)
        print(f"  {t!r:30s} → {p:10s} p={max(probs):.2f}", flush=True)
    return net
