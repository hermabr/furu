"""Where a spec's folders are, for tests that look inside them."""

from pathlib import Path
from typing import Any

from furu import Spec
from furu.storage._layout import FIXED, log_path_in, run_dir_in
from furu.storage.events import read_events
from furu.storage.stored import stored_run


def run_of(spec: Spec) -> Path:
    """The published run that holds spec's result."""
    run = stored_run(spec)
    assert run is not None, f"{spec._log_label} has no stored result"
    return run


def fixed_run(spec: Spec) -> Path:
    """Where spec's run is published when no traced code is involved."""
    return run_dir_in(spec._class_dir / FIXED, spec._run_name)


def _folder(spec: Spec) -> Path:
    return stored_run(spec) or spec._attempt_dir


def log_of(spec: Spec) -> str:
    """What spec's create() wrote, published or still in its working folder."""
    return log_path_in(_folder(spec)).read_text(encoding="utf-8")


def events_of(spec: Spec) -> list[dict[str, Any]]:
    return read_events(_folder(spec))


def ends_of(spec: Spec) -> list[dict[str, Any]]:
    """Each attempt's end event, without its time."""
    return [
        {key: value for key, value in event.items() if key not in ("event", "at")}
        for event in events_of(spec)
        if event["event"] == "end"
    ]
