import argparse
from collections.abc import Sequence
from pathlib import Path

from furu.resources import resource_request_adapter
from furu.worker.loop import worker_loop


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--coordinator-file",
        required=True,
        type=Path,
        help="worker config file holding the execution coordinator URL",
    )
    parser.add_argument(
        "--resources",
        required=True,
        type=resource_request_adapter.validate_json,
        help="this worker's ResourceRequest as JSON",
    )
    parser.add_argument(
        "--idle-timeout",
        required=True,
        type=float,
        help="seconds to wait without a lease before this worker exits",
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        help="consecutive failed jobs after which this worker exits to be replaced",
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
        coordinator=args.coordinator_file,
        resource_request=args.resources,
        idle_timeout=args.idle_timeout,
        max_failures=args.max_failures,
        component=args.component,
        backend=args.backend,
        materialize_snapshot=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
