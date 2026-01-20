import json

import pytest

import furu
from furu.storage.state import _StateResultFailed, _StateResultSuccess


class Fails(furu.Furu[int]):
    def _create(self) -> int:
        raise RuntimeError("boom")

    def _load(self) -> int:
        return json.loads((self.furu_dir / "never.json").read_text())


def test_failed_create_raises_compute_error_and_records_state(furu_tmp_root) -> None:
    obj = Fails()
    with pytest.raises(RuntimeError, match="boom"):
        obj.load_or_create()

    log_text = (obj.furu_dir / ".furu" / "furu.log").read_text()
    assert "[ERROR]" in log_text
    assert "_create failed" in log_text
    assert "Traceback (most recent call last)" in log_text
    assert "RuntimeError: boom" in log_text

    state = furu.StateManager.read_state(obj.furu_dir)
    assert isinstance(state.result, _StateResultFailed)
    attempt = state.attempt
    assert attempt is not None
    assert attempt.status == "failed"
    error = getattr(attempt, "error", None)
    assert error is not None
    assert "boom" in error.message
    assert error.traceback is not None


class InvalidValidate(furu.Furu[int]):
    def _create(self) -> int:
        (self.furu_dir / "value.json").write_text(json.dumps(1))
        return 1

    def _load(self) -> int:
        return 1

    def _validate(self) -> bool:
        raise RuntimeError("validate error")


def test_exists_raises_if_validate_throws(furu_tmp_root) -> None:
    obj = InvalidValidate()
    obj.load_or_create()
    with pytest.raises(RuntimeError, match="validate error"):
        obj.exists()


class ValidateReturnsFalse(furu.Furu[int]):
    def _create(self) -> int:
        (self.furu_dir / "value.json").write_text(json.dumps(1))
        return 1

    def _load(self) -> int:
        raise AssertionError("_load should not be called when _validate returns False")

    def _validate(self) -> bool:
        (self.furu_dir / "validated.txt").write_text("1")
        return False


def test_load_or_create_recomputes_if_validate_returns_false(furu_tmp_root) -> None:
    obj = ValidateReturnsFalse()
    obj.furu_dir.mkdir(parents=True, exist_ok=True)

    def mutate(state) -> None:
        state.result = _StateResultSuccess(status="success", created_at="x")
        state.attempt = None

    furu.StateManager.update_state(obj.furu_dir, mutate)
    furu.StateManager.write_success_marker(obj.furu_dir, attempt_id="test")

    assert obj.load_or_create() == 1
    assert (obj.furu_dir / "validated.txt").exists()
