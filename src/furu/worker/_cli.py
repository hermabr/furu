import argparse
from collections.abc import Sequence
from pathlib import Path

from furu.resources import ResourceRequest
from furu.worker.loop import worker_loop


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--server-url",
        required=True,
        help="execution coordinator API URL",
    )
    parser.add_argument(
        "--auth-token-file",
        required=True,
        type=Path,
        help="path to a file containing the execution coordinator auth token",
    )
    parser.add_argument(
        "--resource-cpus",
        required=True,
        type=int,
        help="CPU count available to this worker",
    )
    parser.add_argument(
        "--resource-gpus",
        required=True,
        type=int,
        help="GPU count available to this worker",
    )
    parser.add_argument(
        "--idle-timeout",
        required=True,
        type=float,
        help="seconds to wait without a lease before this worker exits",
    )
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=None,
        help="consecutive failed jobs after which this worker exits",
    )
    parser.add_argument(
        "--component",
        default="wkr",
        help="component label shown in this worker's logs",
    )
    args = parser.parse_args(argv)

    worker_loop(
        server_url=args.server_url,
        auth_token=args.auth_token_file.read_text(encoding="utf-8").rstrip(),
        resource_request=ResourceRequest(
            cpus=args.resource_cpus,
            gpus=args.resource_gpus,
        ),
        idle_timeout=args.idle_timeout,
        max_consecutive_failures=args.max_consecutive_failures,
        component=args.component,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
