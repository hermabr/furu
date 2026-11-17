"""``@furu.dependency(remember=True)``: a list that is rebuilt only when what it depends on changed."""

import json
import os
import shutil
from multiprocessing import get_context
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from runs import run_of
from tracedrepo import Repo, fresh_process

import furu
from furu import Spec, plan
from furu.config import get_config
from furu.storage._layout import remembered_dir_in
from furu.worker.backends.local import LocalThreadWorkerBackend

# The seeds of each call of LooTable.runs in this process.
_calls: list[int] = []


class Cifar(Spec[int]):
    split: str

    def create(self) -> int:
        return 1


class Train(Spec[int]):
    data: Cifar
    seed: int

    def create(self) -> int:
        return self.seed


class LooTable(Spec[int]):
    data: Cifar
    seeds: int

    @furu.dependency(remember=True)
    def runs(self) -> list[Train]:
        _calls.append(self.seeds)
        return [Train(data=self.data, seed=seed) for seed in range(self.seeds)]

    def create(self) -> int:
        return sum(furu.load_existing(self.runs))


class InfluenceReport(Spec[int]):
    table: LooTable
    top_k: int

    def create(self) -> int:
        return self.table.create() + self.top_k


class Shortlist(Spec[int]):
    table: LooTable

    def create(self) -> int:
        return len(self.table.runs)


class EnvSweep(Spec[int]):
    name: str

    @furu.dependency(remember=True)
    def runs(self) -> list[Cifar]:
        return [Cifar(split=split) for split in os.environ["SPLITS"].split(",")]

    def create(self) -> int:
        return len(self.runs)


@pytest.fixture(autouse=True)
def _a_fresh_process() -> None:
    _calls.clear()
    fresh_process()


def _remembered(spec: Spec) -> list[dict[str, Any]]:
    """The lines of each file in the remembered/ folder of spec's class, by file."""
    folder = remembered_dir_in(spec._class_dir)
    return [
        {"launch": path.stem.removeprefix("launch-"), **json.loads(line)}
        for path in sorted(folder.glob("launch-*.jsonl"), key=os.path.getmtime)
        for line in path.read_text().splitlines()
    ]


def _plans() -> list[Path]:
    plans = get_config().directories.anchored().plans.glob("*.plan")
    return sorted(plans, key=os.path.getmtime)


def test_a_launch_names_a_finished_spec_without_calling_its_method() -> None:
    data = Cifar(split="train")
    table = LooTable(data=data, seeds=3)

    assert furu.create(table) == 3
    assert _calls == [3]
    (line,) = _remembered(table)
    assert line == {
        "launch": _plans()[0].stem,
        "method": "runs",
        "fields": line["fields"],
        "returned": table._returned["runs"],
        "trace": line["trace"],
    }
    trace = remembered_dir_in(table._class_dir) / "traces" / f"{line['trace']}.json"
    assert json.loads(trace.read_text())["code"]["tests/test_remember.py"]["ran"] == [
        "LooTable.runs"
    ]

    # A later launch takes the id from the line.
    fresh_process()
    again = LooTable(data=data, seeds=3)
    assert furu.create(again) == 3
    assert again.object_id == table.object_id
    assert _calls == [3]

    # A report on top: the table's line in the plan holds the fingerprint, as
    # it does for a method that was called.
    report = InfluenceReport(table=LooTable(data=data, seeds=3), top_k=20)
    assert furu.create(report) == 23
    assert _calls == [3]
    assert plan.read(_plans()[-1])[2]["fields"]["runs"] == {
        "|kind": "returned",
        "|value": table._returned["runs"],
    }
    assert len(_remembered(table)) == 1

    # Another field: the method is called, and the launch saves a file of its own.
    more = LooTable(data=data, seeds=4)
    assert furu.create(more) == 6
    assert _calls == [3, 4]
    assert [line["launch"] for line in _remembered(table)] == [
        line["launch"],
        _plans()[-1].stem,
    ]


def test_a_remembered_list_that_differs_stops_whoever_reads_it(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("SPLITS", "train,val")
    assert furu.create(EnvSweep(name="a")) == 2
    assert "EnvSweep.runs" in caplog.text
    assert "reads os.environ, which furu does not trace" in caplog.text

    # furu does not notice what the method reads outside the spec and its code.
    fresh_process()
    monkeypatch.setenv("SPLITS", "train,val,test")
    sweep = EnvSweep(name="a")
    assert sweep.done

    with pytest.raises(RuntimeError, match="remember=True reads something") as stopped:
        _ = sweep.runs
    assert f"{sweep._log_label}.runs returned something else" in str(stopped.value)


def test_a_worker_rebuilds_a_remembered_list_that_its_create_reads() -> None:
    data = Cifar(split="worker")
    furu.create(LooTable(data=data, seeds=3))
    fresh_process()
    _calls.clear()

    shortlist = Shortlist(table=LooTable(data=data, seeds=3))
    assert furu.create(shortlist, on=[LocalThreadWorkerBackend()]) == 3

    # The launch sent the fingerprint, and the worker's process called the method.
    assert _calls == []


def test_storage_that_cannot_be_written_to_still_loads() -> None:
    table = LooTable(data=Cifar(split="read-only"), seeds=2)
    furu.create(table)
    shutil.rmtree(remembered_dir_in(table._class_dir))
    fresh_process()

    table._class_dir.chmod(0o555)
    try:
        assert furu.create(LooTable(data=Cifar(split="read-only"), seeds=2)) == 1
    finally:
        table._class_dir.chmod(0o755)
    assert _calls == [2, 2]
    assert not remembered_dir_in(table._class_dir).exists()


def _name_then_launch(named: Any, launch: Any) -> None:
    table = LooTable(data=Cifar(split="race"), seeds=2)
    assert table.object_id
    named.release()
    launch.wait(timeout=10)
    furu.create(table)


def test_two_launches_at_once_each_save_a_file() -> None:
    ctx = get_context("fork")
    named, launch = ctx.Semaphore(0), ctx.Event()
    launches = [
        ctx.Process(target=_name_then_launch, args=(named, launch), daemon=True)
        for _ in range(2)
    ]
    for process in launches:
        process.start()
    # Both have called the method before either saves.
    for _ in launches:
        assert named.acquire(timeout=10)
    launch.set()
    for process in launches:
        process.join(timeout=10)
        assert process.exitcode == 0

    table = LooTable(data=Cifar(split="race"), seeds=2)
    first, second = _remembered(table)
    assert first["launch"] != second["launch"]
    assert {**first, "launch": ""} == {**second, "launch": ""}
    assert table.done
    assert _calls == []


TABLES = """\
import furu

CALLS = []


def seeds(count):
    return range(count)


def unused():
    return 1


class Subset(furu.Spec[int]):
    count: int

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(code_version="traced")

    def create(self) -> int:
        return self.count


class Train(furu.Spec[int]):
    seed: int

    def create(self) -> int:
        return self.seed


class Table(furu.Spec[int]):
    count: int

    @furu.dependency(remember=True)
    def runs(self) -> list[Train]:
        CALLS.append(self.count)
        return [Train(seed=seed) for seed in seeds(self.count)]

    def create(self) -> int:
        return len(self.runs)


class TracedTable(Table):
    def metadata(self) -> furu.Metadata:
        return furu.Metadata(code_version="traced")

    def create(self) -> int:
        return len(self.runs) * 10


class Articles(furu.Spec[int]):
    count: int

    @furu.dependency(remember=True)
    def batches(self) -> list[Train]:
        CALLS.append(self.count)
        count = Subset(count=self.count).create() + Train(seed=0).create()
        return [Train(seed=seed) for seed in range(count)]

    def create(self) -> int:
        return len(self.batches)
"""


@pytest.fixture
def tables(repo: Repo) -> ModuleType:
    repo.write("tables.py", TABLES)
    return repo.load("tables")


def test_an_edit_calls_the_method_again_only_when_it_is_to_code_the_method_ran(
    repo: Repo, tables: ModuleType
) -> None:
    assert tables.Table(count=2).create() == 2
    assert tables.CALLS == [2]
    before = run_of(tables.Table(count=2))

    repo.edit("tables.py", "return 1", "return 2  # unused")
    tables = repo.load("tables")
    assert tables.Table(count=2).done
    assert tables.CALLS == []

    # A function it calls: the list comes out the same, so the spec is still
    # finished, and the launch saves the new line though nothing runs.
    repo.edit("tables.py", "return range(count)", "return range(0, count)")
    tables = repo.load("tables")
    assert tables.Table(count=2).create() == 2
    assert tables.CALLS == [2]
    assert run_of(tables.Table(count=2)) == before
    fresh_process()
    assert tables.Table(count=2).done
    assert tables.CALLS == [2]


def test_a_traced_class_is_checked_for_its_own_code_without_calling_its_method(
    repo: Repo, tables: ModuleType
) -> None:
    assert tables.TracedTable(count=2).create() == 20
    # The method ran before create() started, each under a trace of its own.
    trace = json.loads(
        (run_of(tables.TracedTable(count=2)).parents[1] / "trace.json").read_text()
    )
    assert trace["code"][f"{repo.package}/tables.py"]["ran"] == ["TracedTable.create"]

    repo.edit("tables.py", "len(self.runs) * 10", "len(self.runs) * 100")
    tables = repo.load("tables")
    table = tables.TracedTable(count=2)
    assert not table.done
    assert tables.CALLS == []
    # The spec has to run, so its list is built, and compared.
    assert table.create() == 200
    assert tables.CALLS == [2]


def test_a_method_is_called_again_when_a_result_it_loaded_is_out_of_date(
    repo: Repo, tables: ModuleType
) -> None:
    articles = tables.Articles(count=2)
    assert articles.create() == 2
    (line,) = _remembered(articles)
    folder = remembered_dir_in(articles._class_dir)
    trace = json.loads((folder / "traces" / f"{line['trace']}.json").read_text())
    subset = tables.Subset(count=2)
    assert trace["built_on"] == [
        f"{subset._stored_name}/{subset._class_part}/{run_of(subset).parents[1].name}"
    ]
    # Each create() the method ran is traced on its own, or not at all.
    assert trace["code"][f"{repo.package}/tables.py"]["ran"] == [
        "Articles.batches",
        "Subset.metadata",
    ]

    tables = repo.load("tables")
    assert tables.Articles(count=2).done
    assert tables.CALLS == []

    repo.edit("tables.py", "return self.count\n", "return self.count + 1\n")
    tables = repo.load("tables")
    assert tables.Articles(count=2).create() == 3
    assert tables.CALLS == [2]


def test_a_class_folder_with_nothing_but_remembered_lists_holds_no_results(
    repo: Repo, tables: ModuleType
) -> None:
    # Naming a spec writes nothing: the next launch saves what was remembered.
    assert not tables.Table(count=2).done
    assert not tables.Table(count=2)._class_dir.exists()
    tables.Train(seed=9).create()
    assert [path.name for path in tables.Table(count=2)._class_dir.iterdir()] == [
        "remembered"
    ]

    # A changed class looks in another folder, and leaves no result behind.
    repo.edit(
        "tables.py",
        "    count: int\n\n    @furu",
        "    count: int\n    scale: int = 1\n\n    @furu",
    )
    assert repo.load("tables").Table(count=2).create() == 2
