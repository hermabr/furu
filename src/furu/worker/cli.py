from __future__ import annotations

import argparse

from furu.worker.loop import worker_loop


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--auth-token", required=True)
    args = parser.parse_args()

    worker_loop(server_url=args.server_url, auth_token=args.auth_token)


if __name__ == "__main__":
    main()
