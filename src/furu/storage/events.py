"""events.jsonl: which specs a run holds and where each attempt ran.

One JSON object per line, each with ``event`` and ``at`` (UTC), of three kinds:
``opened`` once, with the hashes of the run's specs in row order; then per
attempt a ``start`` before create(), with the machine and the plan file that
lists the specs, and an ``end`` after it. A start with no end was killed or is
still running. Only the worker that holds the run's lock appends, and nothing
else is in the file: what create() wrote is in ``log``.
"""

from __future__ import annotations

import json
import os
import socket
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from furu.storage._layout import events_path_in


def read_events(run_dir: Path) -> list[dict[str, Any]]:
    try:
        return parse_events(events_path_in(run_dir).read_bytes())
    except FileNotFoundError:
        return []


def parse_events(text: bytes) -> list[dict[str, Any]]:
    events = []
    for line in text.splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            continue  # blank, or cut short when its attempt was killed
    return events


def _append(run_dir: Path, text: str) -> None:
    with events_path_in(run_dir).open("a", encoding="utf-8") as file:
        file.write(text)
        file.flush()
        os.fsync(file.fileno())  # other machines read it once the run is published


def _line(event: str, **fields: object) -> str:
    at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return json.dumps({"event": event, "at": at, **fields}, separators=(",", ":"))


def start(run_dir: Path, specs: Sequence[str], plan: Path) -> None:
    """Begin an attempt in a working folder, opening the folder if it is new.

    ``plan`` is the plan file that lists the specs: their fields and what they
    depend on.
    """
    earlier = [event.get("event") for event in read_events(run_dir)]
    lines = [] if "opened" in earlier else [_line("opened", specs=list(specs))]
    lines.append(
        _line(
            "start",
            attempt=earlier.count("start") + 1,
            host=socket.gethostname(),
            plan=str(plan),
        )
    )
    # On a new line, so an attempt killed in the middle of a line costs only
    # that line and not this attempt's first.
    _append(run_dir, "".join(f"\n{line}" for line in lines) + "\n")


def end(
    run_dir: Path, outcome: Literal["ok", "failed", "blocked"], **details: object
) -> None:
    """End an attempt: with ``error`` when it failed, ``waiting_on`` when blocked."""
    _append(run_dir, _line("end", outcome=outcome, **details) + "\n")
