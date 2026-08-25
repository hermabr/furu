import argparse
from collections.abc import Sequence
from pathlib import Path

from furu.resources import resource_request_adapter
from furu.worker.loop import worker_loop


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--server-url",
        required=True,
        help="execution coordinator WebSocket URL (ws://host:port)",
    )
    parser.add_argument(
        "--auth-token-file",
        required=True,
        type=Path,
        help="path to a file containing the execution coordinator auth token",
    )
    parser.add_argument(
        "--resources",
        required=True,
        type=resource_request_adapter.validate_json,
        help="JSON ResourceRequest this worker presents when leasing jobs",
    )
    parser.add_argument(
        "--idle-timeout",
        required=True,
        type=float,
        help="seconds to wait without a lease before this worker exits",
    )
    parser.add_argument(
        "--component",
        required=True,
        help="component label shown in this worker's logs",
    )
    parser.add_argument(
        "--backend",
        required=True,
        help="worker backend name recorded in provenance (e.g. slurm)",
    )
    args = parser.parse_args(argv)

    worker_loop(
        server_url=args.server_url,
        auth_token=args.auth_token_file.read_text(encoding="utf-8").rstrip(),
        resource_request=args.resources,
        idle_timeout=args.idle_timeout,
        component=args.component,
        backend=args.backend,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
