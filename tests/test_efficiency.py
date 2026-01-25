import inspect
from pathlib import Path

import furu


class MkdirTask(furu.Furu[int]):
    value: int = furu.chz.field()

    def _create(self) -> int:
        (self.furu_dir / "value.txt").write_text(str(self.value))
        return self.value

    def _load(self) -> int:
        return int((self.furu_dir / "value.txt").read_text())


def test_furu_mkdir_calls_new_vs_existing(furu_tmp_root, monkeypatch) -> None:
    all_calls: list[Path] = []
    furu_calls: list[Path] = []
    original = Path.mkdir
    furu_file = furu.__file__
    if furu_file is None:  # pragma: no cover - import should always have __file__
        raise RuntimeError("furu.__file__ was None")
    furu_base = Path(furu_file).resolve().parent

    def wrapped(
        self: Path,
        mode: int = 0o777,
        parents: bool = False,
        exist_ok: bool = False,
    ) -> None:
        all_calls.append(self)
        caller = Path(inspect.stack()[1].filename).resolve()
        if furu_base in caller.parents:
            furu_calls.append(self)
        return original(self, mode=mode, parents=parents, exist_ok=exist_ok)

    task = MkdirTask(value=1)
    task.furu_dir.parent.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(Path, "mkdir", wrapped)

    def assert_new_object_counts(obj: MkdirTask) -> None:
        base_dir = obj.furu_dir
        internal_dir = obj.furu_dir / ".furu"
        assert all_calls.count(base_dir) == 1
        assert all_calls.count(internal_dir) == 2
        assert furu_calls.count(internal_dir) == 1
        assert furu_calls.count(base_dir) == 0
        assert len(furu_calls) == 1

    all_calls.clear()
    furu_calls.clear()
    assert task.get() == 1
    assert_new_object_counts(task)

    all_calls.clear()
    furu_calls.clear()
    assert task.get() == 1
    assert len(all_calls) == 0
    assert len(furu_calls) == 0

    other_task = MkdirTask(value=2)

    all_calls.clear()
    furu_calls.clear()
    assert other_task.get() == 2
    assert_new_object_counts(other_task)
