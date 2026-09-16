"""python -m lito  → start the assistant."""

from __future__ import annotations

import argparse
import os
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="lito",
        description="Lito — lightweight desktop AI (open apps, do tasks, tiny RAM)",
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Terminal chat only (lowest RAM)",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("LITO_HOST", "0.0.0.0"),
        help="UI bind host (default 0.0.0.0 for LAN/preview)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("LITO_PORT", "8765")),
        help="UI port (default 8765)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not auto-open a browser tab",
    )
    parser.add_argument(
        "-c",
        "--command",
        metavar="TEXT",
        help="Run one command and exit (headless)",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print version and exit",
    )
    args = parser.parse_args(argv)

    if args.version:
        from . import __version__

        print(f"lito {__version__}")
        return 0

    if args.command:
        from .brain import Brain
        from . import memory

        reply = Brain().handle(args.command)
        memory.append_history("user", args.command)
        memory.append_history("assistant", reply.text)
        print(reply.text.replace("**", ""))
        return 0 if reply.ok else 1

    if args.cli:
        from .cli import run_cli

        return run_cli()

    from .ui import LitoServer

    server = LitoServer(host=args.host, port=args.port)
    server.start(open_browser=not args.no_browser, block=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
