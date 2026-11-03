"""The ``furu`` command: ``uvx furu@<version> manager``."""

import argparse
import json
import os
import socket
import threading
from collections.abc import Sequence
from importlib.metadata import version

from furu.config import get_config
from furu.execution.manager import Manager
from furu.logging import _execution_log, get_logger

logger = get_logger("manager")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="furu")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "manager", help="run the queue and the worker pools for every furu run"
    )
    parser.parse_args(argv)

    manager_dir = get_config().manager_dir.resolve()
    manager_dir.mkdir(parents=True, exist_ok=True)
    with (
        _execution_log(manager_dir / "manager.log"),
        Manager.serve(
            listen_host="0.0.0.0",
            connect_host=socket.getfqdn(),
            manager_dir=manager_dir,
        ) as manager,
    ):
        # Scripts find the manager here. In place, not by rename: on NFS a
        # file replaced by rename stays invisible to nodes that had it open.
        address = manager_dir / "address.json"
        fd = os.open(address, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump({"url": manager.url, "furu_version": version("furu")}, file)
        logger.info(
            "manager listening on %s · furu %s",
            manager.url.rpartition("@")[2],
            version("furu"),
            extra={"path": address},
        )
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            logger.info("stopping; pending workers are cancelled, running ones finish")
    return 0
