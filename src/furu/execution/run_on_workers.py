from __future__ import annotations

import json
import os
import secrets
import subprocess
import time
from collections.abc import Sequence
from contextlib import nullcontext, suppress
from importlib.metadata import version

from websockets.exceptions import ConnectionClosed, InvalidHandshake
from websockets.sync.client import ClientConnection, connect

from furu import record
from furu.config import get_config
from furu.core import Spec
from furu.execution.launch import ask, stage_table
from furu.execution.run import Run
from furu.logging import _execution_log, get_logger
from furu.provenance import SubmitProvenance, capture_submit_provenance
from furu.snapshot import CodeLocation
from furu.storage._layout import execution_log_path_in
from furu.storage.stored import Stored
from furu.worker.backends.protocol import WorkerBackend
from furu.worker.messages import CancelMessage, LocalPool

logger = get_logger("run")

_MANAGER_RETRY_INTERVAL_S = 3.0


def run_on_workers(objs: Sequence[Spec], backends: Sequence[WorkerBackend]) -> None:
    """Build ``objs`` on ``backends``' workers, through the furu manager.

    When the connection to the manager drops, plan again from disk, exactly
    as a requeued script does.
    """
    if not backends:
        raise ValueError("on= needs at least one worker backend")
    by_key = {backend.pool_key: backend for backend in backends}
    if len(by_key) != len(backends):
        raise ValueError(
            "worker backends with identical configuration; "
            "use one backend with a larger max_workers instead"
        )
    local = {isinstance(backend.pool(), LocalPool) for backend in backends}
    if len(local) > 1:
        raise ValueError(
            "cannot mix local and Slurm worker backends: local threads run your "
            "live worktree, Slurm workers run the pushed commit"
        )
    look = Stored()
    if all(look.row(obj) is not None for obj in objs):
        logger.info("all objects already exist; nothing to run")
        return

    provenance = capture_submit_provenance()
    code = CodeLocation.here() if local == {True} else _prepare_snapshot(provenance)
    directories = get_config().directories.anchored()
    launch = secrets.token_hex(16)
    execution_log = execution_log_path_in(directories.executions / launch)
    launch_plan = directories.plans / f"{launch}.plan"
    from furu.execution.manager import Manager

    in_process = (
        Manager.serve(
            listen_host="127.0.0.1", connect_host="127.0.0.1", manager_dir=None
        )
        if local == {True}
        else nullcontext(None)
    )
    # Asked and written once: planning again after losing the manager is the
    # same launch.
    asked: frozenset[str] = frozenset()
    with _execution_log(execution_log), in_process as manager:
        while True:
            run = Run(
                backends=by_key,
                code=code,
                config=get_config().anchored(),
                execution_log=execution_log,
            )
            run.add(objs, launch_plan)
            if not run.nodes_by_id:
                logger.info("run finished ok")
                return
            missing = [node.obj for node in run.nodes_by_id.values()]
            logger.info(
                "planned exec %s · %d ready · %d blocked\n%s",
                execution_log.parent.name[:5],
                len(run.ready),
                len(run.blocked),
                stage_table(missing),
                extra={"path": execution_log},
            )
            if not asked:
                asked = ask(missing)
                record.launch(launch_plan, objs, missing, provenance)
            run.asked = asked
            with (
                connect(manager.url, max_size=None)
                if manager is not None
                else _connect_to_manager()
            ) as connection:
                try:
                    run.drive(connection)
                    return
                except ConnectionClosed:
                    logger.warning("lost the manager; planning again from disk")
                except BaseException:  # Ctrl-C or a failed run; never SIGTERM
                    with suppress(ConnectionClosed):
                        connection.send(CancelMessage().model_dump_json())
                    raise


def _prepare_snapshot(provenance: SubmitProvenance) -> CodeLocation:
    """Extract the pushed commit and build its venv once, before workers race."""
    code = CodeLocation.from_snapshot(provenance)
    subprocess.run(
        ["uv", "sync", "--frozen", "--project", str(code.project_root)],
        env={k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"},
        check=True,
    )
    return code


def _connect_to_manager() -> ClientConnection:
    """Connect to the running manager, waiting for one to start."""
    address = get_config().manager_version_dir / "address.json"
    waiting = False
    while True:
        try:
            published = json.loads(address.read_text(encoding="utf-8"))
            return connect(published["url"], max_size=None)
        except (OSError, ValueError, InvalidHandshake) as exc:  # not up yet
            if not waiting:
                waiting = True
                logger.warning(
                    "waiting for the furu manager (start one with "
                    "uvx furu@%s manager) · %s",
                    version("furu"),
                    type(exc).__name__,
                    extra={"path": address},
                )
            time.sleep(_MANAGER_RETRY_INTERVAL_S)
