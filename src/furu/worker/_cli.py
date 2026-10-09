import argparse
from collections.abc import Sequence
from pathlib import Path

from furu.storage._layout import slurm_worker_log_path_in
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
        "--pool",
        required=True,
        help="key of the worker pool this worker belongs to",
    )
    parser.add_argument(
        "--idle-timeout",
        required=True,
        type=float,
        help="seconds to wait without a lease before this worker exits",
    )
    parser.add_argument(
        "--max-failures",
        required=True,
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

    try:
        worker_loop(
            coordinator=args.coordinator_file,
            pool=args.pool,
            idle_timeout=args.idle_timeout,
            max_failures=args.max_failures,
            component=args.component,
            backend=args.backend,
            materialize_snapshot=True,
            # This process's own output, which sbatch sends to this file.
            worker_log=slurm_worker_log_path_in(
                args.coordinator_file.parent, args.component
            ),
        )
    except SystemExit:  # gave up after too many failures; already logged
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
