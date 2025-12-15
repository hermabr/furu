import sys
from pathlib import Path

import pytest


# Make `import huldra` work in a src-layout checkout without requiring an install.
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


@pytest.fixture()
def huldra_tmp_root(tmp_path, monkeypatch):
    import huldra

    monkeypatch.setattr(huldra.HULDRA_CONFIG, "base_root", tmp_path)
    monkeypatch.setattr(huldra.HULDRA_CONFIG, "ignore_git_diff", True)
    monkeypatch.setattr(huldra.HULDRA_CONFIG, "poll_interval", 0.01)
    monkeypatch.setattr(huldra.HULDRA_CONFIG, "stale_timeout", 0.1)
    monkeypatch.setattr(huldra.HULDRA_CONFIG, "lease_duration_sec", 0.05)
    monkeypatch.setattr(huldra.HULDRA_CONFIG, "heartbeat_interval_sec", 0.01)
    monkeypatch.setattr(huldra.HULDRA_CONFIG, "cancelled_is_preempted", True)
    return tmp_path
