import json
from typing import ClassVar

import pytest

import furu
from furu.storage.state import (
    _StateAttemptFailed,
    _StateResultFailed,
    _StateResultSuccess,
)


class Fails(furu.Furu[int]):
    def _create(self) -> int:
        raise RuntimeError("boom")

    def _load(self) -> int:
        return json.loads((self.furu_dir / "never.json").read_text())


class Flaky(furu.Furu[int]):
    _create_calls: ClassVar[int] = 0

    def _create(self) -> int:
        type(self)._create_calls += 1
        if type(self)._create_calls == 1:
            raise RuntimeError("boom")
        value = 7
        (self.furu_dir / "value.json").write_text(json.dumps(value))
        return value

    def _load(self) -> int:
        return json.loads((self.furu_dir / "value.json").read_text())


class MetadataFails(furu.Furu[int]):
    def _create(self) -> int:
        value = 5
        (self.furu_dir / "value.json").write_text(json.dumps(value))
        return value

    def _load(self) -> int:
        return json.loads((self.furu_dir / "value.json").read_text())


def test_failed_create_raises_compute_error_and_records_state(furu_tmp_root) -> None:
    obj = Fails()
    with pytest.raises(RuntimeError, match="boom"):
        obj.get()

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


def test_failed_state_raises_compute_error_with_recorded_traceback(
    furu_tmp_root,
    monkeypatch,
) -> None:
    obj = Fails()
    with pytest.raises(RuntimeError, match="boom"):
        obj.get()

    monkeypatch.setattr(furu.FURU_CONFIG, "retry_failed", False)
    with pytest.raises(furu.FuruComputeError) as exc:
        obj.get()

    assert exc.value.recorded_traceback is not None
    assert "RuntimeError: boom" in exc.value.recorded_traceback
    message = str(exc.value)
    assert "Recorded traceback" in message
    assert "FURU_RETRY_FAILED" in message


def test_retry_failed_allows_recompute(furu_tmp_root, monkeypatch) -> None:
    Flaky._create_calls = 0
    obj = Flaky()

    with pytest.raises(RuntimeError, match="boom"):
        obj.get()

    monkeypatch.setattr(furu.FURU_CONFIG, "retry_failed", True)
    result = obj.get()
    assert result == 7
    assert Flaky._create_calls == 2

    state = furu.StateManager.read_state(obj.furu_dir)
    assert isinstance(state.result, _StateResultSuccess)


def test_retry_failed_from_config(furu_tmp_root, monkeypatch) -> None:
    Flaky._create_calls = 0
    obj = Flaky()

    monkeypatch.setattr(furu.FURU_CONFIG, "retry_failed", False)
    with pytest.raises(RuntimeError, match="boom"):
        obj.get()

    monkeypatch.setattr(furu.FURU_CONFIG, "retry_failed", True)
    result = obj.get()
    assert result == 7
    assert Flaky._create_calls == 2


def test_metadata_failure_marks_attempt_failed(furu_tmp_root, monkeypatch) -> None:
    obj = MetadataFails()

    def boom(*_args, **_kwargs) -> None:
        raise RuntimeError("metadata boom")

    monkeypatch.setattr(furu.MetadataManager, "create_metadata", boom)
    with pytest.raises(furu.FuruComputeError, match="Failed to create metadata") as exc:
        obj.get()
    assert isinstance(exc.value.original_error, RuntimeError)

    state = furu.StateManager.read_state(obj.furu_dir)
    assert isinstance(state.result, _StateResultFailed)
    attempt = state.attempt
    assert attempt is not None
    assert attempt.status == "failed"
    assert isinstance(attempt, _StateAttemptFailed)
    assert attempt.error.message == "metadata boom"
    assert attempt.error.traceback is not None


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
    obj.get()
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


def test_get_recomputes_if_validate_returns_false(furu_tmp_root) -> None:
    obj = ValidateReturnsFalse()
    furu.StateManager.ensure_internal_dir(obj.furu_dir)

    def mutate(state) -> None:
        state.result = _StateResultSuccess(status="success", created_at="x")
        state.attempt = None

    furu.StateManager.update_state(obj.furu_dir, mutate)
    furu.StateManager.write_success_marker(obj.furu_dir, attempt_id="test")

    assert obj.get() == 1
    assert (obj.furu_dir / "validated.txt").exists()
