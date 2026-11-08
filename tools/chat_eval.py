#!/usr/bin/env python3
"""
Chat the model and grade the answers.

This is the gate that stops a release where "hello" comes back as word
salad. It loads a GGUF with the repo's own numpy engine, sends the
messages a person actually sends first, and checks each reply against
what a sane answer must contain.

    python tools/chat_eval.py                      # lito-chat.gguf
    python tools/chat_eval.py --gguf path.gguf -v  # show every reply
    python tools/chat_eval.py --min-score 0.8      # CI gate

Exit code is 0 when the score reaches --min-score, 1 otherwise.
"""

import argparse
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.llama_np import LlamaEngine  # noqa: E402

# (prompt, [accepted patterns], weight) - a reply passes when any
# pattern matches (case insensitive).
PROBES = [
    ('hello', [r'\b(hello|hi|hey|good (morning|afternoon|evening))\b'], 2),
    ('hi', [r'\b(hello|hi|hey)\b'], 2),
    ('hey there', [r'\b(hello|hi|hey)\b'], 1),
    ('good morning', [r'\b(good morning|morning|hello|hi)\b'], 1),
    ('who are you', [r'\blito\b', r'\b(assistant|ai|model|program)\b'], 2),
    ('what is your name', [r'\blito\b'], 2),
    ('what can you do', [r'\b(help|answer|story|stories|question|chat|'
                         r'talk)\b'], 1),
    ('are you human', [r'\b(no|not|ai|program|model|computer)\b'], 1),
    ('how are you', [r"\b(i'm|i am|good|well|fine|great|doing)\b"], 1),
    ('thanks', [r"\b(welcome|happy to help|any time|pleasure|glad)\b"], 1),
    ('bye', [r'\b(bye|goodbye|see you|take care|good night)\b'], 1),
    ('what is 2 plus 3?', [r'\b5\b|\bfive\b'], 1),
    ('what is 7 plus 5?', [r'\b12\b|\btwelve\b'], 1),
    ('What is the capital of France?', [r'\bparis\b'], 1),
    ('How many days are in a week?', [r'\bseven\b|\b7\b'], 1),
    ('tell me a story', [r'\b(once|there (was|were)|one day|a little)\b'], 2),
]

# A reply must also look like English prose, not a token soup - the
# same judge the app uses before it shows neural text to a user.
from src.quality import looks_like_english  # noqa: E402


def run(gguf, temperature=0.7, top_p=0.9, seed=1234, max_tokens=80,
        verbose=False):
    engine = LlamaEngine(gguf)
    total = sum(p[2] for p in PROBES)
    got = 0.0
    rows = []
    t0 = time.time()
    for prompt, patterns, weight in PROBES:
        reply = engine.chat([{'role': 'user', 'content': prompt}],
                            max_new_tokens=max_tokens,
                            temperature=temperature, top_p=top_p,
                            repeat_penalty=1.1, seed=seed)
        ok_english, why = looks_like_english(reply)
        matched = any(re.search(p, reply, re.I) for p in patterns)
        score = weight if (ok_english and matched) else 0
        got += score
        rows.append((prompt, reply, bool(score),
                     '' if ok_english else why))
        mark = 'PASS' if score else 'FAIL'
        note = f' [{why}]' if why else ''
        if verbose:
            print(f'{mark}  {prompt!r}{note}\n      -> {reply}\n')
        else:
            short = reply.replace('\n', ' ')[:70]
            print(f'{mark}  {prompt:<28} -> {short}{note}')
    elapsed = time.time() - t0
    print(f'\nscore {got:.0f}/{total} = {got / total:.0%}  '
          f'({len(PROBES)} probes in {elapsed:.0f}s)')
    return got / total, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gguf', default=os.path.join(ROOT, 'lito-chat.gguf'))
    ap.add_argument('--temperature', type=float, default=0.7)
    ap.add_argument('--top-p', type=float, default=0.9)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--max-tokens', type=int, default=80)
    ap.add_argument('--min-score', type=float, default=0.0,
                    help='exit 1 when the score is below this (0-1)')
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args()

    if not os.path.exists(args.gguf):
        print(f'no model at {args.gguf}')
        return 1
    print(f'model: {args.gguf} '
          f'({os.path.getsize(args.gguf) / 1e6:.1f} MB)\n')
    score, _ = run(args.gguf, args.temperature, args.top_p, args.seed,
                   args.max_tokens, args.verbose)
    if args.min_score and score < args.min_score:
        print(f'FAILED: score {score:.0%} below the required '
              f'{args.min_score:.0%}')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
