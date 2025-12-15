import json

import pytest

import huldra


class Fails(huldra.Huldra[int]):
    def _create(self) -> int:
        raise RuntimeError("boom")

    def _load(self) -> int:
        return json.loads((self.huldra_dir / "never.json").read_text())


def test_failed_create_raises_compute_error_and_records_state(huldra_tmp_root) -> None:
    obj = Fails()
    with pytest.raises(RuntimeError, match="boom"):
        obj.load_or_create()

    log_text = (obj.huldra_dir / "huldra.log").read_text()
    assert "[ERROR]" in log_text
    assert "_create failed" in log_text
    assert "Traceback (most recent call last)" in log_text
    assert "RuntimeError: boom" in log_text

    state = huldra.StateManager.read_state(obj.huldra_dir)
    assert state["result"]["status"] == "failed"
    attempt = state["attempt"]
    assert attempt["status"] == "failed"
    assert "boom" in attempt["error"]["message"]
    assert "traceback" in attempt["error"]


class InvalidValidate(huldra.Huldra[int]):
    def _create(self) -> int:
        (self.huldra_dir / "value.json").write_text(json.dumps(1))
        return 1

    def _load(self) -> int:
        return 1

    def _validate(self) -> bool:
        raise RuntimeError("validate error")


def test_exists_returns_false_if_validate_throws(huldra_tmp_root) -> None:
    obj = InvalidValidate()
    obj.load_or_create()
    assert obj.exists() is False
