"""python -m lito"""

from __future__ import annotations

import argparse
import os
import sys

from . import __version__


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="lito", description="Lito — lightest thinking AI")
    p.add_argument("--version", action="version", version=f"lito {__version__}")
    p.add_argument("--cli", "-i", action="store_true", help="terminal REPL")
    p.add_argument("-c", "--command", help="one-shot prompt")
    p.add_argument("--host", default=os.environ.get("LITO_HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(os.environ.get("LITO_PORT", "8765")))
    p.add_argument("--no-browser", action="store_true")
    p.add_argument(
        "--hide-thoughts",
        action="store_true",
        help="don't print thinking trace",
    )
    args = p.parse_args(argv)

    if args.hide_thoughts:
        os.environ["LITO_SHOW_THOUGHTS"] = "0"

    if args.command is not None:
        from .agent import Agent

        r = Agent().handle(args.command)
        print(r.text)
        return 0 if r.ok else 1

    if args.cli or not sys.stdin.isatty() and args.command is None and False:
        from .cli import run_cli

        return run_cli()

    if args.cli:
        from .cli import run_cli

        return run_cli()

    # default: UI
    from .ui import run_server

    run_server(host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
