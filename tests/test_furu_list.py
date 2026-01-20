import json

import furu


class Exp(furu.Furu[int]):
    value: int = furu.chz.field()

    def _create(self) -> int:
        (self.furu_dir / "value.json").write_text(json.dumps(self.value))
        return self.value

    def _load(self) -> int:
        return json.loads((self.furu_dir / "value.json").read_text())


class Experiments(furu.FuruList[Exp]):
    a = Exp(value=1)
    b = Exp(value=1)  # duplicate digest
    nested = {"x": Exp(value=2)}
    also_nested = [Exp(value=2)]  # duplicate digest


def test_collection_dedup_and_lookup(furu_tmp_root) -> None:
    all_exps = Experiments.all()
    assert sorted(e.value for e in all_exps) == [1, 2]

    assert Experiments.by_name("a").value == 1
    assert Experiments.by_name("x").value == 2
    assert Experiments.by_name("missing", strict=False) is None
