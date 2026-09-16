"""python -m lito  -> start the assistant."""

from __future__ import annotations

import argparse
import os
import sys
import threading


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="lito",
        description="Lito - lightweight desktop AI (open apps, do tasks, tiny RAM)",
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
    parser.add_argument(
        "--check-update",
        action="store_true",
        help="Check GitHub Releases for a newer build and exit",
    )
    parser.add_argument(
        "--install-update",
        action="store_true",
        help="Download and apply the latest GitHub Release binary",
    )
    parser.add_argument(
        "--no-auto-update-check",
        action="store_true",
        help="Skip background update check on startup",
    )
    args = parser.parse_args(argv)

    if args.version:
        from . import __version__
        from .update import current_platform_tag, is_frozen

        frozen = ", frozen" if is_frozen() else ""
        print(f"lito {__version__} ({current_platform_tag()}{frozen})")
        return 0

    if args.check_update:
        from .actions import check_update

        ok, msg = check_update()
        print(msg.replace("**", ""))
        return 0 if ok else 1

    if args.install_update:
        from .actions import install_update

        ok, msg = install_update(restart=False)
        print(msg.replace("**", ""))
        return 0 if ok else 1

    if args.command:
        from . import memory
        from .brain import Brain

        reply = Brain().handle(args.command)
        memory.append_history("user", args.command)
        memory.append_history("assistant", reply.text)
        print(reply.text.replace("**", ""))
        return 0 if reply.ok else 1

    # Optional non-blocking GitHub update probe (stdlib HTTP, tiny)
    if not args.no_auto_update_check:
        _spawn_update_check()

    if args.cli:
        from .cli import run_cli

        return run_cli()

    from .ui import LitoServer

    server = LitoServer(host=args.host, port=args.port)
    server.start(open_browser=not args.no_browser, block=True)
    return 0


def _spawn_update_check() -> None:
    """Fire-and-forget latest-release probe so startup stays instant."""

    def _job() -> None:
        try:
            from . import __version__
            from . import update as update_mod

            if not update_mod.auto_check_enabled():
                return
            info = update_mod.check_for_update()
            update_mod.write_last_check(info)
            if info is not None:
                print(
                    f"[lito] update available: {__version__} -> {info.version} "
                    f"(check update / install update)",
                    flush=True,
                )
        except Exception:
            return

    threading.Thread(target=_job, name="lito-update-check", daemon=True).start()


if __name__ == "__main__":
    raise SystemExit(main())
