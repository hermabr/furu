from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from furu.worker.loop import worker_loop


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m furu.worker.cli")
    parser.add_argument(
        "--server-url",
        required=True,
        help="URL of the furu manager server (e.g. http://head.example:8080)",
    )
    parser.add_argument(
        "--auth-token",
        required=True,
        help="Bearer token to authenticate with the manager.",
    )
    args = parser.parse_args(argv)

    worker_loop(server_url=args.server_url, auth_token=args.auth_token)
    return 0


if __name__ == "__main__":
    sys.exit(main())
