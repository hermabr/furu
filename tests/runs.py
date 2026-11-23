"""Where a spec's folders are, for tests that look inside them."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from furu import Spec
from furu.core import attempt_parent
from furu.storage._layout import FIXED, log_path_in, run_dir_in
from furu.storage.attempt_lock import current
from furu.storage.events import read_events
from furu.storage.stored import stored_row


def run_of(spec: Spec) -> Path:
    """The published run that holds spec's result."""
    row = stored_row(spec)
    assert row is not None, f"{spec.object_id} has no stored result"
    return row.run


def fixed_run(spec: Spec) -> Path:
    """Where spec's run is published when no traced code is involved."""
    return run_dir_in(spec._class_dir / FIXED, spec._artifact_hash)


def working(specs: Spec | Sequence[Spec]) -> Path | None:
    """The working folder of the run of a spec, or of a batch, if it has one."""
    batch = [specs] if isinstance(specs, Spec) else specs
    return current(attempt_parent(batch))


def folder_of(specs: Spec | Sequence[Spec]) -> Path:
    """The run of a spec, or of a batch: published, or else its working folder."""
    batch = [specs] if isinstance(specs, Spec) else specs
    row = stored_row(batch[0])
    folder = row.run if row is not None else working(batch)
    assert folder is not None, f"{batch[0].object_id} has no run"
    return folder


def log_of(specs: Spec | Sequence[Spec]) -> str:
    """What the create() of a spec or batch wrote."""
    return log_path_in(folder_of(specs)).read_text(encoding="utf-8")


def events_of(specs: Spec | Sequence[Spec]) -> list[dict[str, Any]]:
    return read_events(folder_of(specs))


def ends_of(specs: Spec | Sequence[Spec]) -> list[dict[str, Any]]:
    """Each attempt's end event, without its time."""
    return [
        {key: value for key, value in event.items() if key not in ("event", "at")}
        for event in events_of(specs)
        if event["event"] == "end"
    ]
