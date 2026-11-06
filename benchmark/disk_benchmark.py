"""Convenience entry point for the bounded read-only disk benchmark."""

from __future__ import annotations

import sys

from main import main


if __name__ == "__main__":
    sys.argv[1:1] = ["disk-read"]
    raise SystemExit(main())
