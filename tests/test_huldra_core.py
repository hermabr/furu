import json

import huldra


@huldra.huldra(slug="test-huldra-core")
class Dummy(huldra.Huldra[int]):
    _create_calls: int = 0
    _load_calls: int = 0

    def _create(self) -> int:
        object.__setattr__(self, "_create_calls", self._create_calls + 1)
        value = 123
        (self.huldra_dir / "value.json").write_text(json.dumps(value))
        return value

    def _load(self) -> int:
        object.__setattr__(self, "_load_calls", self._load_calls + 1)
        return json.loads((self.huldra_dir / "value.json").read_text())


def test_load_or_create_returns_create_result_without_load(huldra_tmp_root) -> None:
    obj = Dummy()
    result = obj.load_or_create()

    assert result == 123
    assert obj._create_calls == 1
    assert obj._load_calls == 0

    result2 = obj.load_or_create()
    assert result2 == 123
    assert obj._create_calls == 1
    assert obj._load_calls == 1


def test_exists_reflects_success_state(huldra_tmp_root) -> None:
    obj = Dummy()
    assert obj.exists() is False
    obj.load_or_create()
    assert obj.exists() is True
