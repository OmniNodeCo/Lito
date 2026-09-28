"""Terminal REPL — lightest interface."""

from __future__ import annotations

import sys

from . import __version__
from .agent import Agent


def run_cli(one_shot: str | None = None) -> int:
    agent = Agent()
    if one_shot is not None:
        r = agent.handle(one_shot)
        print(r.text)
        return 0 if r.ok else 1

    print(f"Lito {__version__} — thinking agent (tiny RAM). Type help · quit")
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.lower() in {"quit", "exit", "q"}:
            break
        r = agent.handle(line)
        print()
        print(r.text)
        print()
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    one = None
    if "-c" in argv:
        i = argv.index("-c")
        one = " ".join(argv[i + 1 :])
    elif "--cli" in argv or "-i" in argv:
        one = None
    return run_cli(one)
