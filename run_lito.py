#!/usr/bin/env python3
"""Convenience launcher: python run_lito.py [--cli]"""

from lito.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
