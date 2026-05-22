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
        help="manager API URL",
    )
    parser.add_argument(
        "--auth-token-file",
        required=True,
        type=Path,
        help="path to a file containing the manager auth token",
    )
    parser.add_argument(
        "--resource-cpus",
        type=int,
        help="CPU count available to this worker",
    )
    parser.add_argument(
        "--resource-gpus",
        type=int,
        help="GPU count available to this worker",
    )
    args = parser.parse_args(argv)

    worker_loop(
        server_url=args.server_url,
        auth_token=args.auth_token_file.read_text(encoding="utf-8").rstrip(),
        resource_request=ResourceRequest(
            cpus=args.resource_cpus,
            gpus=args.resource_gpus,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
