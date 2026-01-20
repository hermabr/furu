import json

import furu


class Manifest(furu.Furu[list[str]]):
    items: list[str] = furu.chz.field()
    _create_calls: int = 0
    _load_calls: int = 0

    def _create(self) -> list[str]:
        object.__setattr__(self, "_create_calls", self._create_calls + 1)
        (self.furu_dir / "items.json").write_text(json.dumps(self.items))
        return list(self.items)

    def _load(self) -> list[str]:
        object.__setattr__(self, "_load_calls", self._load_calls + 1)
        return json.loads((self.furu_dir / "items.json").read_text())


class SortedManifest(Manifest):
    def _create(self) -> list[str]:
        object.__setattr__(self, "_create_calls", self._create_calls + 1)
        sorted_items = sorted(self.items)
        (self.furu_dir / "items.json").write_text(json.dumps(sorted_items))
        return sorted_items


def test_furu_subclass_can_override_create_and_inherit_load(furu_tmp_root) -> None:
    base = Manifest(items=["b", "a"])
    assert base.load_or_create() == ["b", "a"]
    assert base._create_calls == 1
    assert base._load_calls == 0
    assert base.load_or_create() == ["b", "a"]
    assert base._create_calls == 1
    assert base._load_calls == 1

    derived = SortedManifest(items=["b", "a"])
    assert derived.load_or_create() == ["a", "b"]
    assert derived._create_calls == 1
    assert derived._load_calls == 0
    assert derived.load_or_create() == ["a", "b"]
    assert derived._create_calls == 1
    assert derived._load_calls == 1
