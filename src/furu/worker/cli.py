from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

from furu.worker.loop import worker_loop


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server-url",
        default=os.environ.get("FURU_MANAGER_SERVER_URL"),
        help="manager API URL; defaults to FURU_MANAGER_SERVER_URL",
    )
    parser.add_argument(
        "--auth-token",
        default=os.environ.get("FURU_MANAGER_AUTH_TOKEN"),
        help="manager auth token; defaults to FURU_MANAGER_AUTH_TOKEN",
    )
    args = parser.parse_args(argv)

    if args.server_url is None:
        parser.error("--server-url or FURU_MANAGER_SERVER_URL is required")
    if args.auth_token is None:
        parser.error("--auth-token or FURU_MANAGER_AUTH_TOKEN is required")

    worker_loop(server_url=args.server_url, auth_token=args.auth_token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
