import json

import pytest

import huldra


class Fails(huldra.Huldra[int], slug="test-errors"):
    def _create(self) -> int:
        raise RuntimeError("boom")

    def _load(self) -> int:
        return json.loads((self.huldra_dir / "never.json").read_text())


def test_failed_create_raises_compute_error_and_records_state(huldra_tmp_root) -> None:
    obj = Fails()
    with pytest.raises(huldra.HuldraComputeError):
        obj.load_or_create()

    state = huldra.StateManager.read_state(obj.huldra_dir)
    assert state["status"] == "failed"
    assert "boom" in state.get("reason", "")
    assert "traceback" in state


class InvalidValidate(huldra.Huldra[int], slug="test-validate"):
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
