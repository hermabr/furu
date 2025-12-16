import json

import pytest

import huldra
from huldra.storage.state import _StateResultFailed, _StateResultSuccess


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
    assert isinstance(state.result, _StateResultFailed)
    attempt = state.attempt
    assert attempt is not None
    assert attempt.status == "failed"
    error = getattr(attempt, "error", None)
    assert error is not None
    assert "boom" in error.message
    assert error.traceback is not None


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


class ValidateReturnsFalse(huldra.Huldra[int]):
    def _create(self) -> int:
        (self.huldra_dir / "value.json").write_text(json.dumps(1))
        return 1

    def _load(self) -> int:
        raise AssertionError("_load should not be called when _validate returns False")

    def _validate(self) -> bool:
        (self.huldra_dir / "validated.txt").write_text("1")
        return False


def test_load_or_create_recomputes_if_validate_returns_false(huldra_tmp_root) -> None:
    obj = ValidateReturnsFalse()
    obj.huldra_dir.mkdir(parents=True, exist_ok=True)

    def mutate(state) -> None:
        state.result = _StateResultSuccess(status="success", created_at="x")
        state.attempt = None

    huldra.StateManager.update_state(obj.huldra_dir, mutate)
    huldra.StateManager.write_success_marker(obj.huldra_dir, attempt_id="test")

    assert obj.load_or_create() == 1
    assert (obj.huldra_dir / "validated.txt").exists()
