"""Train NanoLM as a free-form response generator (not tool-script presets)."""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

from .dialogue import build_generative_corpus
from .model import ModelConfig, NanoLM
from .tokenizer import Tokenizer
from .train import train_on_sequence  # reuse fast backprop


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2200)
    ap.add_argument("--dim", type=int, default=48)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--ctx", type=int, default=6)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.22)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).resolve().parent / "weights" / "lito-nano.bin"),
    )
    args = ap.parse_args(argv)

    random.seed(args.seed)
    corpus = build_generative_corpus()
    # densify short chat
    extra = []
    for s in corpus:
        if s.count(" ") < 25:
            extra.append(s)
    corpus.extend(extra)
    print(f"gen corpus {len(corpus)}", flush=True)

    tok = Tokenizer()
    tok.extend_from_corpus(corpus, max_extra=220)
    # ensure common response words
    for w in (
        "hey", "hello", "glad", "happy", "help", "today", "mind", "ready", "welcome",
        "equals", "plus", "times", "result", "found", "remember", "local", "system",
        "welcome", "question", "together", "neural", "assistant", "anything",
        "wonderful", "sorry", "listen", "curious", "plants", "light", "energy",
        "protocol", "devices", "function", "problem", "language", "people",
    ):
        if w not in tok.stoi:
            tok.stoi[w] = len(tok.stoi)
    tok.itos = {i: s for s, i in tok.stoi.items()}
    tok.vocab_size = len(tok.stoi)
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
        text = random.choice(corpus)
        ids = tok.encode(text, add_bos=True, add_eos=True)
        if len(ids) > args.block:
            # prefer keeping the answer tail
            ids = ids[-args.block :]
        prog = step / args.steps
        lr = args.lr * 0.5 * (1 + math.cos(math.pi * prog))
        lr = max(lr, args.lr * 0.05)
        loss = train_on_sequence(model, ids, lr)
        avg = 0.9 * avg + 0.1 * loss if step > 1 else loss
        if step % 200 == 0 or step == 1:
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
        f'"name":"Lito-Nano-Gen","goal":"free-form response generation"}}\n',
        encoding="utf-8",
    )
    print(f"saved {out} in {time.time()-t0:.1f}s", flush=True)

    from .generate import generate_reply

    for u in ("hello", "thanks", "what is mqtt", "calculate 3+4"):
        obs = "7" if "3+4" in u else ""
        if "mqtt" in u:
            obs = "MQTT is a lightweight pub/sub protocol"
        print(">", u, "=>", generate_reply(model, tok, u, obs=obs), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
