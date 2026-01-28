import sys
from pathlib import Path

import pytest


# Make `import furu` work in a src-layout checkout without requiring an install.
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


@pytest.fixture()
def furu_tmp_root(tmp_path, monkeypatch):
    import furu

    monkeypatch.setattr(furu.FURU_CONFIG, "base_root", tmp_path)
    monkeypatch.setattr(
        furu.FURU_CONFIG,
        "version_controlled_root_override",
        tmp_path / "furu-data" / "artifacts",
    )
    monkeypatch.setattr(furu.FURU_CONFIG, "record_git", "ignore")
    monkeypatch.setattr(furu.FURU_CONFIG, "allow_no_git_origin", False)
    monkeypatch.setattr(furu.FURU_CONFIG, "poll_interval", 0.01)
    monkeypatch.setattr(furu.FURU_CONFIG, "stale_timeout", 0.1)
    monkeypatch.setattr(furu.FURU_CONFIG, "max_wait_time_sec", None)
    monkeypatch.setattr(furu.FURU_CONFIG, "lease_duration_sec", 0.05)
    monkeypatch.setattr(furu.FURU_CONFIG, "heartbeat_interval_sec", 0.01)
    monkeypatch.setattr(furu.FURU_CONFIG, "cancelled_is_preempted", True)
    monkeypatch.setattr(furu.FURU_CONFIG, "retry_failed", True)
    return tmp_path
