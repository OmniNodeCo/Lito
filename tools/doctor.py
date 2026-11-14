#!/usr/bin/env python3
"""
Lito doctor - "which model is actually answering me, and is it any good?"

Word salad from a chat model is almost never a mystery once you know
*which file* produced it. This walks every GGUF the app could load, says
where each one came from, picks the one that wins, and makes it answer
the six things a person types first - then grades the answers with the
same judge the release gate uses.

    python tools/doctor.py              # check the model the app will load
    python tools/doctor.py -m other.gguf
    python tools/doctor.py --quiet      # exit code only (0 healthy)

Exit codes: 0 healthy, 1 degraded (it answers, but badly), 2 no model.
Standard library + numpy only - the same thing the app needs.
"""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# The first minute of any conversation. If these are wrong, nothing else
# matters; if these are right, the model is doing its job.
FIRST_CONTACT = [
    'hello',
    'how are you',
    'who are you',
    'what can you do',
    'thanks',
    'bye',
]

HEALTHY = 0
DEGRADED = 1
NO_MODEL = 2


def find_candidates(explicit=None):
    """Every GGUF the app could pick, best first."""
    if explicit:
        return [os.path.abspath(explicit)]
    from src.brain import AIBrain
    return AIBrain.find_gguf_models()


def check(path, verbose=True):
    """Load one model, interview it, grade the answers."""
    from src.llama_np import LlamaEngine
    from src.quality import looks_like_english

    engine = LlamaEngine(path)
    summary = engine.build_summary()
    stale = not engine.train_info

    rows, good = [], 0
    for prompt in FIRST_CONTACT:
        try:
            reply = engine.chat([{'role': 'user', 'content': prompt}],
                                temperature=0.0, max_new_tokens=60).strip()
        except Exception as exc:                      # pragma: no cover
            reply, ok, why = f'<error: {exc}>', False, 'crashed'
        else:
            ok, why = looks_like_english(reply)
            # A greeting that answers a different question is still
            # English, so require it to look like a reply, not a ramble.
            if ok and len(reply) < 3:
                ok, why = False, 'empty'
        good += bool(ok)
        rows.append((prompt, reply, ok, why))

    if verbose:
        print(f'  build : {summary}')
        if stale:
            print('  WARNING: no provenance in this file. It was exported '
                  'before v1.2.6,')
            print('           which is almost certainly the old '
                  'word-salad build.')
        print()
        for prompt, reply, ok, why in rows:
            mark = 'ok  ' if ok else 'BAD '
            print(f'  {mark} {prompt:<14} -> {reply[:96]}')
            if not ok:
                print(f'       {"":<14}    ({why})')
        print()

    return summary, stale, good, len(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-m', '--model', help='check this GGUF instead')
    ap.add_argument('-q', '--quiet', action='store_true',
                    help='no output, just the exit code')
    args = ap.parse_args(argv)
    talk = not args.quiet

    candidates = find_candidates(args.model)
    if not candidates:
        if talk:
            print('no GGUF model found.\n')
            print('  The trained model ships with this repository:')
            print('      git -C . checkout lito-chat.gguf')
            print('  or train one:  python train_big.py && '
                  'python export_gguf.py')
        return NO_MODEL

    if talk:
        print()
        print(f'models found ({len(candidates)}):')
        for i, path in enumerate(candidates):
            size = os.path.getsize(path) / 1e6
            mark = '-> ' if i == 0 else '   '
            print(f'  {mark}{path}  ({size:.1f} MB)')
        if len(candidates) > 1:
            print('     (the arrow is the one the app loads; the others '
                  'are shadowed)')
        print()
        print(f'interviewing {os.path.basename(candidates[0])}:')

    summary, stale, good, total = check(candidates[0], verbose=talk)
    healthy = good == total and not stale

    if talk:
        verdict = ('healthy' if healthy else
                   'DEGRADED' if good else 'BROKEN')
        print(f'verdict: {verdict} - {good}/{total} first-contact answers '
              'are clean English')
        if not healthy:
            print()
            if stale:
                print('  This file predates v1.2.6 and does not say how it '
                      'was built.')
                print('  The current model is committed to the repo - '
                      'replace yours with it:')
                print('      git -C <repo> checkout lito-chat.gguf')
                print('  and if a stale copy is shadowing it, delete that '
                      'copy (see the list above).')
            else:
                print('  The file is current but answers badly. Retrain or '
                      're-export:')
                print('      python train_big.py --resume && '
                      'python export_gguf.py')
                print('      python tools/chat_eval.py -v   # full grade')
        print()

    return HEALTHY if healthy else DEGRADED


if __name__ == '__main__':
    raise SystemExit(main())
