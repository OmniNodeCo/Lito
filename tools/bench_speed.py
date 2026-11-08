#!/usr/bin/env python3
"""
Measure training throughput for the same model under different
execution layouts. Parameter count is identical in every variant - only
how the maths is executed changes.

    python tools/bench_speed.py                 # all variants
    python tools/bench_speed.py --steps 10

Variants
    split+manual : one Linear per projection, hand-written attention
                   (the original implementation)
    fused        : Q/K/V in one GEMM, gate/up in one GEMM
    sdpa         : torch's fused scaled-dot-product attention
    fused+sdpa   : both (what train_big.py now uses by default)
    +compile     : fused+sdpa run through torch.compile
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from train_big import (build_model, split_state_dict, fuse_state_dict,  # noqa: E402
                       Muon)


def run_variant(name, cfg, vocab, batch, steps, warmup, fused, sdpa,
                compile_model=False):
    torch.manual_seed(0)
    model = build_model(cfg, vocab, fused=fused, sdpa=sdpa)
    n_params = sum(p.numel() for p in model.parameters())
    if compile_model:
        model = torch.compile(model)
    hidden = [p for n, p in model.named_parameters()
              if p.ndim == 2 and 'embed' not in n]
    others = [p for n, p in model.named_parameters()
              if not (p.ndim == 2 and 'embed' not in n)]
    opts = [Muon(hidden, lr=0.04),
            torch.optim.AdamW(others, lr=3e-3, betas=(0.9, 0.95),
                              weight_decay=0.1)]
    x = torch.randint(0, vocab, (batch, cfg['ctx']))
    y = torch.randint(0, vocab, (batch, cfg['ctx']))

    t_total = 0.0
    for i in range(steps + warmup):
        if i == warmup:
            t_total = time.time()
        for o in opts:
            o.zero_grad(set_to_none=True)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, vocab), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for o in opts:
            o.step()
    elapsed = time.time() - t_total
    tok = steps * batch * cfg['ctx']
    print(f'{name:<22} {n_params:>11,} params  '
          f'{elapsed / steps:6.2f} s/step  {tok / elapsed:7.0f} tok/s',
          flush=True)
    return tok / elapsed, n_params


def check_equivalence(cfg, vocab):
    """Fused and split layouts must compute the same function."""
    torch.manual_seed(1)
    fused = build_model(cfg, vocab, fused=True, sdpa=True)
    split = build_model(cfg, vocab, fused=False, sdpa=False)
    split.load_state_dict(split_state_dict(fused.state_dict(), cfg))
    ids = torch.randint(0, vocab, (2, cfg['ctx']))
    with torch.no_grad():
        a, b = fused(ids), split(ids)
    diff = (a - b).abs().max().item()
    # and the round trip back into the fused layout
    refused = build_model(cfg, vocab, fused=True, sdpa=True)
    refused.load_state_dict(fuse_state_dict(split_state_dict(
        fused.state_dict(), cfg)))
    with torch.no_grad():
        c = refused(ids)
    diff2 = (a - c).abs().max().item()
    print(f'fused vs split max|diff| {diff:.2e} | round trip {diff2:.2e}')
    assert diff < 1e-4 and diff2 == 0.0, 'layouts disagree'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=8)
    ap.add_argument('--warmup', type=int, default=2)
    ap.add_argument('--batch', type=int, default=12)
    ap.add_argument('--ctx', type=int, default=512)
    ap.add_argument('--d_model', type=int, default=256)
    ap.add_argument('--n_layers', type=int, default=6)
    ap.add_argument('--n_ff', type=int, default=672)
    ap.add_argument('--vocab', type=int, default=4096)
    ap.add_argument('--compile', action='store_true')
    args = ap.parse_args()

    torch.set_num_threads(max(1, os.cpu_count() or 2))
    cfg = {'d_model': args.d_model, 'n_layers': args.n_layers, 'n_heads': 8,
           'n_ff': args.n_ff, 'ctx': args.ctx, 'vocab_size': args.vocab}
    print(f'{os.cpu_count()} cores | batch {args.batch} x ctx {args.ctx} | '
          f'd{args.d_model} L{args.n_layers} ff{args.n_ff}\n')
    check_equivalence({**cfg, 'ctx': 64}, 128)
    print()

    base = None
    for name, fused, sdpa in (('split+manual (old)', False, False),
                              ('fused', True, False),
                              ('sdpa', False, True),
                              ('fused+sdpa (new)', True, True)):
        rate, _ = run_variant(name, cfg, args.vocab, args.batch, args.steps,
                              args.warmup, fused, sdpa)
        if base is None:
            base = rate
        else:
            print(f'{"":22} {rate / base:.2f}x vs old')
    if args.compile:
        rate, _ = run_variant('fused+sdpa+compile', cfg, args.vocab,
                              args.batch, args.steps, args.warmup,
                              True, True, compile_model=True)
        print(f'{"":22} {rate / base:.2f}x vs old')


if __name__ == '__main__':
    main()
