from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

from furu.worker.loop import worker_loop

AUTH_TOKEN_ENV_VAR = "FURU_AUTH_TOKEN"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m furu.worker.cli")
    parser.add_argument(
        "--server-url",
        required=True,
        help="URL of the furu manager server (e.g. http://head.example:8080)",
    )
    parser.add_argument(
        "--auth-token",
        default=None,
        help=(
            f"Bearer token to authenticate with the manager. Defaults to the "
            f"{AUTH_TOKEN_ENV_VAR} environment variable."
        ),
    )
    args = parser.parse_args(argv)

    auth_token = args.auth_token or os.environ.get(AUTH_TOKEN_ENV_VAR)
    if not auth_token:
        parser.error(
            f"--auth-token is required (or set the {AUTH_TOKEN_ENV_VAR} environment variable)"
        )

    worker_loop(server_url=args.server_url, auth_token=auth_token)
    return 0


if __name__ == "__main__":
    sys.exit(main())
