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

from furu import plan, record, remember
from furu.config import get_config
from furu.core import Spec
from furu.execution.launch import ask, missing, stage_table
from furu.execution.run import Run
from furu.logging import _execution_log, get_logger
from furu.provenance import SubmitProvenance, capture_submit_provenance
from furu.snapshot import CodeLocation
from furu.storage._layout import execution_log_path_in
from furu.worker.backends.protocol import WorkerBackend
from furu.worker.messages import CancelMessage, Launch, LocalPool

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
            "use one backend with more workers instead"
        )
    local = {isinstance(backend.pool(), LocalPool) for backend in backends}
    if len(local) > 1:
        raise ValueError(
            "cannot mix local and Slurm worker backends: local threads run your "
            "live worktree, Slurm workers run the pushed commit"
        )
    launch = secrets.token_hex(16)
    running = missing(objs)
    remember.save(launch)
    if not running:
        logger.info("all objects already exist; nothing to run")
        return

    provenance = capture_submit_provenance()
    code = CodeLocation.here() if local == {True} else _prepare_snapshot(provenance)
    directories = get_config().directories.anchored()
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
    first = True
    with _execution_log(execution_log), in_process as manager:
        while running:
            logger.info(
                "planned exec %s · %d missing\n%s",
                execution_log.parent.name[:5],
                len(running),
                stage_table(running),
                extra={"path": execution_log},
            )
            if first:
                asked = ask(running)
            run = Run(
                backends=by_key,
                launch=Launch(
                    plan=launch_plan,
                    execution_log=execution_log,
                    config=get_config().anchored(),
                    code=code,
                    asked=asked,
                ),
            )
            # Placed before the plan is written: a launch that no pool can
            # run leaves no files.
            groups = run.place(running)
            if first:
                lines = record.launch(launch_plan, objs, running, provenance)
            else:
                ids = {obj.object_id for obj in running}
                lines = list(map(plan.dump, plan.lines(running, ids)))
            first = False
            submit = run.submit(groups, lines, launch_plan)
            with (
                connect(manager.url, max_size=None)
                if manager is not None
                else _connect_to_manager()
            ) as connection:
                try:
                    run.drive(connection, submit)
                    return
                except ConnectionClosed:
                    logger.warning("lost the manager; planning again from disk")
                except BaseException:  # Ctrl-C or a failed run; never SIGTERM
                    with suppress(ConnectionClosed):
                        connection.send(CancelMessage().model_dump_json())
                    raise
            running = missing(objs)
        logger.info("run finished ok")


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
