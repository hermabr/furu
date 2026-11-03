import argparse
import os
from collections.abc import Sequence
from pathlib import Path

from furu.worker.loop import worker_loop

# The URL carries the manager's token, so it travels in the environment
# (readable only by this user) rather than on the command line.
MANAGER_URL_ENV_VAR = "_FURU_MANAGER_URL"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--pool", required=True, help="key of this worker's pool")
    parser.add_argument(
        "--component", required=True, help="this worker's name, e.g. its Slurm job id"
    )
    parser.add_argument("--log", required=True, type=Path, help="this worker's log")
    parser.add_argument(
        "--idle-timeout",
        required=True,
        type=float,
        help="seconds to wait without a job before this worker exits",
    )
    args = parser.parse_args(argv)
    worker_loop(
        url=os.environ[MANAGER_URL_ENV_VAR],
        pool=args.pool,
        idle_timeout=args.idle_timeout,
        component=args.component,
        backend="slurm",
        worker_log=args.log,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
