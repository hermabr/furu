"""Where a spec's folders are, for tests that look inside them."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from furu import Spec
from furu.config import get_config
from furu.core import attempt_parent
from furu.storage import attempt_lock
from furu.storage._layout import FIXED, log_path_in, run_dir_in
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


def working_folder(specs: Spec | Sequence[Spec]) -> Path:
    """The working folder of a spec's run, or a batch's, under the name it has now."""
    folder = attempt_lock.current(attempt_parent(_batch(specs)))
    assert folder is not None, f"{_batch(specs)[0].object_id} has no working folder"
    return folder


def _batch(specs: Spec | Sequence[Spec]) -> Sequence[Spec]:
    return [specs] if isinstance(specs, Spec) else specs


def _folder(specs: Spec | Sequence[Spec]) -> Path:
    """The run of a spec, or of a batch: published, or else its working folder."""
    row = stored_row(_batch(specs)[0])
    return row.run if row is not None else working_folder(specs)


def log_of(specs: Spec | Sequence[Spec]) -> str:
    """What the create() of a spec or batch wrote."""
    return log_path_in(_folder(specs)).read_text(encoding="utf-8")


def events_of(specs: Spec | Sequence[Spec]) -> list[dict[str, Any]]:
    return read_events(_folder(specs))


def ends_of(specs: Spec | Sequence[Spec]) -> list[dict[str, Any]]:
    """Each attempt's end event, without its time."""
    return [
        {key: value for key, value in event.items() if key not in ("event", "at")}
        for event in events_of(specs)
        if event["event"] == "end"
    ]


def stored_files() -> dict[Path, bytes | None]:
    """Every path in furu's directories, with what each file holds."""
    return {
        path: path.read_bytes() if path.is_file() else None
        for _, directory in get_config().directories.anchored()
        for path in sorted(directory.rglob("*"))
    }
