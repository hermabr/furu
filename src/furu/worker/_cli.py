import argparse
import os
import signal
import sys
from collections.abc import Sequence
from pathlib import Path
from types import FrameType

from websockets.exceptions import WebSocketException

from furu.config import _WORKER_JSON_CONFIG_FILE_ENV_VAR
from furu.logging import get_logger
from furu.resources import ResourceRequest
from furu.worker.endpoint import read_worker_endpoint
from furu.worker.loop import WorkerPreempted, worker_loop

logger = get_logger("worker.cli")


def _raise_preempted(signum: int, frame: FrameType | None) -> None:
    raise WorkerPreempted


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--endpoint-file",
        required=True,
        type=Path,
        help=(
            "JSON file naming the execution coordinator URL, auth token, and "
            "project; re-read after a disconnect so a successor coordinator "
            "can redirect this worker"
        ),
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
        "--resource-memory-gib",
        required=True,
        type=int,
        help="memory in GiB available to this worker",
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

    endpoint = read_worker_endpoint(args.endpoint_file)
    try:
        signal.signal(signal.SIGUSR1, _raise_preempted)
    except ValueError:
        # Not the main thread (e.g. a test runner); preemption then only
        # arrives via the connection-closed path.
        pass

    error: BaseException | None = None
    try:
        exit_reason = worker_loop(
            server_url=endpoint.server_url,
            auth_token=endpoint.auth_token,
            resource_request=ResourceRequest(
                cpus=args.resource_cpus,
                gpus=args.resource_gpus,
                memory_gib=args.resource_memory_gib,
            ),
            idle_timeout=args.idle_timeout,
            component=args.component,
            backend=args.backend,
        )
    except WorkerPreempted:
        logger.info("preempted; abandoning in-flight work")
        exit_reason = "disconnected"
    except (OSError, WebSocketException) as exc:
        error = exc
        exit_reason = "disconnected"

    if exit_reason == "disconnected":
        current = read_worker_endpoint(args.endpoint_file)
        if current.generation > endpoint.generation:
            logger.info(
                "endpoint generation %d -> %d; re-exec against %s",
                endpoint.generation,
                current.generation,
                current.server_url,
            )
            os.environ[_WORKER_JSON_CONFIG_FILE_ENV_VAR] = current.config_file
            # The submit-side venv belongs to the old project; uv would warn
            # before selecting the new --project venv.
            os.environ.pop("VIRTUAL_ENV", None)
            os.execvp(
                "uv",
                [
                    "uv",
                    "run",
                    "--frozen",
                    "--project",
                    current.project_root,
                    "python",
                    "-m",
                    "furu.worker._cli",
                    *(sys.argv[1:] if argv is None else argv),
                ],
            )
    if error is not None:
        raise error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
