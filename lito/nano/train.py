"""Train Lito-Nano — balanced agent curriculum, pure Python."""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

from .curriculum import build_corpus
from .model import ModelConfig, NanoLM, generate
from .tokenizer import Tokenizer


def _default_out() -> Path:
    return Path(__file__).resolve().parent / "weights" / "lito-nano.bin"


def gelu(x: float) -> float:
    return 0.5 * x * (1.0 + math.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))


def gelu_grad(x: float) -> float:
    c0 = 0.7978845608
    x2 = x * x
    inner = c0 * (x + 0.044715 * x * x2)
    th = math.tanh(inner)
    dinner = c0 * (1.0 + 0.134145 * x2)
    return 0.5 * (1.0 + th) + 0.5 * x * (1.0 - th * th) * dinner


def softmax(xs: list[float]) -> list[float]:
    m = max(xs)
    ex = [math.exp(v - m) for v in xs]
    s = sum(ex)
    inv = 1.0 / s
    return [e * inv for e in ex]


def train_on_sequence(model: NanoLM, ids: list[int], lr: float) -> float:
    if len(ids) < 2:
        return 0.0
    C = model.config
    K, D, H, V = C.n_ctx, C.n_embd, C.n_hidden, C.vocab_size
    in_dim = K * D
    fc1, b1, fc2, b2 = model.fc1, model.b1, model.fc2, model.b2
    head, hb, wte = model.head, model.hb, model.wte

    g_fc1 = [[0.0] * H for _ in range(in_dim)]
    g_b1 = [0.0] * H
    g_fc2 = [[0.0] * H for _ in range(H)]
    g_b2 = [0.0] * H
    g_head = [[0.0] * V for _ in range(H)]
    g_hb = [0.0] * V
    g_wte = [[0.0] * D for _ in range(V)]

    total_loss = 0.0
    positions = list(range(1, len(ids)))
    if len(positions) > 28:
        # bias toward the end (tool/answer tokens)
        tail = positions[-18:]
        head_p = positions[:-18]
        positions = tail + random.sample(head_p, min(10, len(head_p)))

    n_pos = 0
    for end in positions:
        target = ids[end]
        start = max(0, end - K)
        x = [0.0] * in_dim
        got = end - start
        pad = K - got
        for local_i, pos in enumerate(range(start, end)):
            off = (pad + local_i) * D
            te = wte[ids[pos]]
            pe = model.wpe[min(pos, C.block_size - 1)]
            for u in range(D):
                x[off + u] = te[u] + pe[u]

        h1_pre = [0.0] * H
        for j in range(H):
            s = b1[j]
            col = [fc1[i][j] for i in range(in_dim)]  # slow — avoid
            s = b1[j]
            for i in range(in_dim):
                s += x[i] * fc1[i][j]
            h1_pre[j] = s
        h1 = [gelu(v) for v in h1_pre]
        h2_pre = [0.0] * H
        for j in range(H):
            s = b2[j]
            for i in range(H):
                s += h1[i] * fc2[i][j]
            h2_pre[j] = s
        h2 = [gelu(v) for v in h2_pre]
        logits = [0.0] * V
        for j in range(V):
            s = hb[j]
            for i in range(H):
                s += h2[i] * head[i][j]
            logits[j] = s

        probs = softmax(logits)
        total_loss += -math.log(max(probs[target], 1e-12))
        n_pos += 1
        dlogits = probs
        dlogits[target] -= 1.0

        dh2 = [0.0] * H
        for j in range(V):
            g = dlogits[j]
            g_hb[j] += g
            hj = g
            for i in range(H):
                g_head[i][j] += h2[i] * hj
                dh2[i] += head[i][j] * hj

        dh2_pre = [dh2[i] * gelu_grad(h2_pre[i]) for i in range(H)]
        dh1 = [0.0] * H
        for j in range(H):
            g = dh2_pre[j]
            g_b2[j] += g
            for i in range(H):
                g_fc2[i][j] += h1[i] * g
                dh1[i] += fc2[i][j] * g
        dh1_pre = [dh1[i] * gelu_grad(h1_pre[i]) for i in range(H)]
        dx = [0.0] * in_dim
        for j in range(H):
            g = dh1_pre[j]
            g_b1[j] += g
            for i in range(in_dim):
                g_fc1[i][j] += x[i] * g
                dx[i] += fc1[i][j] * g
        for local_i, pos in enumerate(range(start, end)):
            off = (pad + local_i) * D
            tok = ids[pos]
            row = g_wte[tok]
            for u in range(D):
                row[u] += dx[off + u]

    if n_pos == 0:
        return 0.0
    inv = lr / n_pos

    def sgd_mat(W, G):
        for i, row in enumerate(G):
            Wi = W[i]
            for j, g in enumerate(row):
                if g:
                    Wi[j] -= inv * g

    def sgd_vec(W, G):
        for i, g in enumerate(G):
            if g:
                W[i] -= inv * g

    sgd_mat(fc1, g_fc1)
    sgd_vec(b1, g_b1)
    sgd_mat(fc2, g_fc2)
    sgd_vec(b2, g_b2)
    sgd_mat(head, g_head)
    sgd_vec(hb, g_hb)
    sgd_mat(wte, g_wte)
    return total_loss / n_pos


def bucketize(corpus: list[str]) -> dict[str, list[str]]:
    buckets = {"math": [], "memory": [], "search": [], "chat": [], "sys": [], "other": []}
    for s in corpus:
        low = s.lower()
        if "calc" in low or "calculate" in low or "plus" in low or "*" in low:
            buckets["math"].append(s)
        elif "remember" in low or "recall" in low:
            buckets["memory"].append(s)
        elif "search" in low or "what is" in low or "explain" in low or "look up" in low or "define" in low or "who is" in low:
            buckets["search"].append(s)
        elif "sysinfo" in low or "<tool> time" in low or "how much ram" in low:
            buckets["sys"].append(s)
        elif "hello" in low or "help" in low or "thanks" in low or "who are you" in low:
            buckets["chat"].append(s)
        else:
            buckets["other"].append(s)
    return buckets


def sample_balanced(buckets: dict[str, list[str]]) -> str:
    # weights: keep math present but not dominant
    weights = {
        "math": 0.18,
        "memory": 0.18,
        "search": 0.28,
        "chat": 0.14,
        "sys": 0.10,
        "other": 0.12,
    }
    keys = [k for k, v in buckets.items() if v]
    ws = [weights.get(k, 0.1) for k in keys]
    s = sum(ws)
    r = random.random() * s
    acc = 0.0
    pick = keys[-1]
    for k, w in zip(keys, ws):
        acc += w
        if r <= acc:
            pick = k
            break
    return random.choice(buckets[pick])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--dim", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=80)
    ap.add_argument("--ctx", type=int, default=5)
    ap.add_argument("--block", type=int, default=56)
    ap.add_argument("--lr", type=float, default=0.28)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", type=str, default=str(_default_out()))
    args = ap.parse_args(argv)

    random.seed(args.seed)
    corpus = build_corpus()
    # modest math only
    for a in range(0, 12):
        for b in range(0, 10):
            corpus.append(
                f"<user> calculate {a}+{b} <bot><think> add numbers </think>"
                f"<tool> calc {a}+{b} </tool><obs> {a+b} </obs>"
                f"<ans> {a}+{b} = {a+b} </ans>"
            )
    buckets = bucketize(corpus)
    print("buckets:", {k: len(v) for k, v in buckets.items()}, flush=True)

    tok = Tokenizer()
    tok.extend_from_corpus(corpus, max_extra=100)
    print(f"vocab {tok.vocab_size}", flush=True)

    cfg = ModelConfig(
        vocab_size=tok.vocab_size,
        n_embd=args.dim,
        n_hidden=args.hidden,
        n_ctx=args.ctx,
        block_size=args.block,
    )
    model = NanoLM(cfg)
    print(f"params {model.param_count():,} ({model.param_count()*4/1024:.0f} KB)", flush=True)

    t0 = time.time()
    avg = 0.0
    for step in range(1, args.steps + 1):
        text = sample_balanced(buckets)
        ids = tok.encode(text, add_bos=True, add_eos=True)
        if len(ids) > args.block:
            ids = ids[-args.block :]
        prog = step / args.steps
        lr = args.lr * 0.5 * (1 + math.cos(math.pi * prog))
        lr = max(lr, args.lr * 0.05)
        loss = train_on_sequence(model, ids, lr)
        avg = 0.9 * avg + 0.1 * loss if step > 1 else loss
        if step % 250 == 0 or step == 1:
            print(
                f"step {step}/{args.steps} loss={loss:.3f} avg={avg:.3f} "
                f"lr={lr:.3f} t={time.time()-t0:.0f}s",
                flush=True,
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    tok.save(out.with_suffix(".tok.json"))
    out.with_suffix(".meta.json").write_text(
        f'{{"params":{model.param_count()},"vocab":{tok.vocab_size},'
        f'"dim":{args.dim},"hidden":{args.hidden},"steps":{args.steps},'
        f'"name":"Lito-Nano","arch":"ctx-MLP","desc":"custom micro LLM agent brain"}}\n',
        encoding="utf-8",
    )
    print(f"saved {out} in {time.time()-t0:.1f}s", flush=True)

    for prompt in (
        "<user> calculate 3+4 <bot>",
        "<user> hello <bot>",
        "<user> what is mqtt <bot>",
        "<user> remember wifi is orchard <bot>",
        "<user> help <bot>",
    ):
        ids = tok.encode(prompt, add_bos=True)
        gen = generate(model, ids, max_new=40, temperature=0.2, top_k=6, eos_id=tok.eos_id)
        print(">", tok.decode(gen)[:220], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
