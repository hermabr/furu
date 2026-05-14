from __future__ import annotations

import argparse
from collections.abc import Sequence

from furu.worker.loop import worker_loop


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server-url",
        required=True,
        help="manager API URL",
    )
    parser.add_argument(
        "--auth-token",
        required=True,
        help="manager auth token",
    )
    args = parser.parse_args(argv)

    worker_loop(server_url=args.server_url, auth_token=args.auth_token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
