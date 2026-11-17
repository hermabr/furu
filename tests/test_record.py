"""The record of what ran: a launch's file and each process's file in provenance/."""

import os
import socket
import subprocess
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import pytest
from gitrepo import commit_and_push, git
from runs import events_of, run_of

import furu
from furu import Metadata, Spec, plan, provenance, record
from furu.config import get_config
from furu.worker.backends.local import LocalThreadWorkerBackend


class Cifar(Spec[int]):
    split: str

    def create(self) -> int:
        return 1


class Train(Spec[int]):
    data: Cifar
    seed: int

    def metadata(self) -> Metadata:
        return Metadata(
            environment={"OMP_NUM_THREADS": "1"},
            required_environment_variables=("HOME",),
        )

    @furu.batched(lambda spec, worker: (None, 100))
    def create(specs: list["Train"], directory: furu.SpecDirectory) -> list[int]:
        return [spec.seed for spec in specs]


class LooTable(Spec[int]):
    data: Cifar
    notes: tuple[str, ...]

    @furu.dependency
    def runs(self) -> list[Train]:
        return [Train(data=self.data, seed=seed) for seed in range(3)]

    def create(self) -> int:
        return sum(furu.load_existing(self.runs))


class Recount(Spec[int]):
    data: Cifar

    @furu.dependency
    def runs(self) -> list[Train]:
        return [Train(data=self.data, seed=7)]

    def create(self) -> int:
        return furu.load_existing(Train(data=self.data, seed=7))


class Hinted(Spec[int]):
    seed: int

    @furu.dependency
    def hints(self) -> Cifar | None:
        return Cifar(split="hints") if self.seed else None

    def create(self) -> int:
        return self.seed


class Search(Spec[int]):
    arm: str

    def create(self) -> int:
        return sum(furu.create([Hinted(seed=seed) for seed in range(2)]))


def _ref(spec: Spec) -> dict[str, Any]:
    return {"|kind": "spec", "|value": spec.object_id}


def _pointer(fingerprint: str) -> dict[str, Any]:
    return {"|kind": "value", "|value": fingerprint}


def _recorded(spec: Spec) -> tuple[dict[str, Any], dict[str, Any]]:
    """The line about the process that ran spec's run, and the run's line in its file."""
    _, start, end = events_of(spec)[-3:]
    worker, *lines = plan.read(Path(start["provenance"]))
    (line,) = [line for line in lines if line.get("run") == run_of(spec).name]
    assert (line["start"], line["end"]) == (start["at"], end["at"])
    return worker["worker"], line


@pytest.mark.parametrize("on_workers", [False, True])
def test_a_launch_writes_what_its_runs_point_to_and_each_run_gets_a_line(
    on_workers: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FURU_IGNORE_STORED", "pkg.Old:5d0e")
    on = [LocalThreadWorkerBackend()] if on_workers else None
    data = Cifar(split=f"train-{on_workers}")
    table = LooTable(data=data, notes=tuple(f"note {n}" for n in range(200)))

    assert furu.create(table, on=on) == 3

    directories = get_config().directories.anchored()
    (launch,) = directories.plans.glob("*.plan")
    facts, *shared = plan.read(directories.provenance / f"launch-{launch.stem}")
    assert facts.keys() == {
        *("launch", "git", "environment", "submitted", "ignore_stored")
    }
    assert facts["launch"] == launch.stem
    assert facts["git"]["commit"] == git(Path.cwd(), "rev-parse", "HEAD")
    assert facts["ignore_stored"] == "pkg.Old:5d0e"
    # What a dependency method returned and a value over 1 KB are written
    # once, here, and so is each distinct Metadata.
    values = {line["value"]: line["is"] for line in shared if "value" in line}
    runs = table._returned["runs"]
    (notes,) = values.keys() - {runs}
    assert values[runs] == [_ref(run) for run in table.runs]
    assert values[notes]["|value"] == list(table.notes)
    metadata = {
        plan.dump(line["is"]): line["metadata"] for line in shared if "metadata" in line
    }
    default = {
        "environment": {},
        "required_environment_variables": [],
        "reuse": "same_environment",
    }
    threads = default | {
        "environment": {"OMP_NUM_THREADS": "1"},
        "required_environment_variables": ["HOME"],
    }
    assert metadata.keys() == {plan.dump(default), plan.dump(threads)}

    worker, line = _recorded(table)
    assert worker["host"] == socket.gethostname()
    assert worker["backend"] == ("local-thread" if on_workers else "local")
    assert (worker["pid"] == os.getpid()) is not on_workers
    # The line holds the spec's own fields and what it points to. The runs
    # its create() loaded are in the value it points to.
    assert line == {
        "run": table._artifact_hash,
        "storage": str(directories.objects),
        "version": "fixed",
        "launch": launch.stem,
        "metadata": metadata[plan.dump(default)],
        "start": line["start"],
        "end": line["end"],
        "specs": [
            {
                "spec": table.object_id,
                "fields": {
                    "data": _ref(data),
                    "notes": _pointer(notes),
                    "runs": {"|kind": "returned", "|value": runs},
                },
            }
        ],
    }
    # A batch is one line, with a spec line per row.
    _, batch = _recorded(table.runs[0])
    assert batch["metadata"] == metadata[plan.dump(threads)]
    assert sorted(batch["specs"], key=lambda spec: spec["fields"]["seed"]) == [
        {"spec": run.object_id, "fields": {"data": _ref(data), "seed": run.seed}}
        for run in table.runs
    ]

    # A launch with nothing to run writes nothing.
    assert furu.create(table, on=on) == 3
    assert len(list(directories.provenance.glob("launch-*"))) == 1


def test_found_specs_have_their_values_recorded_first_and_are_listed_as_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search = Search(arm="full article")
    found = [Hinted(seed=seed) for seed in range(2)]
    returned = {
        "value": found[1]._returned["hints"],
        "is": _ref(Cifar(split="hints")),
    }
    seen = []

    def save_found(lines: list[plan.Line]) -> Path:
        seen.append(returned in plan.read(record.own().path))
        return real(lines)

    real = plan.save_found
    monkeypatch.setattr(plan, "save_found", save_found)

    assert search.create() == 1

    # The values were on disk before the lines that point to them.
    assert seen == [True]
    # The search holds no spec, so its line lists the results it loaded.
    _, line = _recorded(search)
    assert line["loaded"] == [spec.object_id for spec in found]
    # A found run names the launch whose create() asked for it.
    assert _recorded(found[1])[1]["launch"] == line["launch"]


def test_a_spec_its_dependency_method_returned_is_not_listed_as_loaded() -> None:
    recount = Recount(data=Cifar(split="val"))

    # On a worker the spec is built from its line, and its create() never
    # reads the field.
    assert furu.create(recount, on=[LocalThreadWorkerBackend()]) == 7

    assert "loaded" not in _recorded(recount)[1]


def _own_files() -> tuple[Path, Path]:
    return record.own().path, plan.save_found([])


def test_a_forked_process_creates_files_of_its_own() -> None:
    plans = get_config().directories.anchored().plans
    with plan.listed_in(plans / "9f3c.plan"):
        mine = _own_files()
        with get_context("fork").Pool(1) as pool:
            theirs = pool.apply(_own_files)

    assert not {*mine} & {*theirs}


def test_a_process_creates_its_own_file_and_a_killed_writer_costs_one_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    notebook = plan.File.new(tmp_path)
    assert notebook.path.name == f"{socket.gethostname()}-pid-{os.getpid()}"

    # A restarted Slurm job keeps its id, and never adds to the file it finds.
    monkeypatch.setenv("SLURM_JOB_ID", "18389503")
    first, second = plan.File.new(tmp_path), plan.File.new(tmp_path)
    assert [first.path.name, second.path.name] == ["job-18389503", "job-18389503-2"]

    first.add([{"run": "b81e"}])
    first.add([{"run": "41d7"}])
    assert plan.read(first.path) == [{"run": "b81e"}, {"run": "41d7"}]
    first.path.write_bytes(first.path.read_bytes()[:-8])
    assert plan.read(first.path) == [{"run": "b81e"}]


_created: list[int] = []


class _Node(Spec[int]):
    value: int

    def create(self) -> int:
        _created.append(self.value)
        return self.value + 1


def test_a_launch_needs_pushed_code_and_loading_does_not(
    git_repo: Path, fresh_caches: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Undo the furu pytest plugin's opt-out, in a real uv project: a launch
    # also checks the lock's path dependencies against the repo.
    monkeypatch.setattr(provenance, "_require_pushed_code", True)
    (git_repo / "pyproject.toml").write_text(
        '[project]\nname = "repo"\nversion = "0"\nrequires-python = ">=3.12"\n'
    )
    subprocess.run(["uv", "lock", "-q", "--offline"], cwd=git_repo, check=True)
    commit_and_push(git_repo)
    monkeypatch.chdir(git_repo)

    assert furu.create(_Node(value=1)) == 2
    (launch,) = get_config().directories.anchored().provenance.glob("launch-*")
    assert plan.read(launch)[0]["git"] == {
        "commit": git(git_repo, "rev-parse", "origin/main"),
        "branch": "main",
        "remote": str(git_repo.parent / "repo-origin.git"),
        "repo_root": str(git_repo.resolve()),
    }

    (git_repo / "tracked.txt").write_text("edited\n")
    with pytest.raises(RuntimeError, match="uncommitted changes"):
        furu.create(_Node(value=2))
    assert _created == [1]
    assert furu.create(_Node(value=1)) == 2


def test_outside_a_git_repo_a_launch_fails_before_it_computes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(RuntimeError, match="not inside a git repository"):
        furu.create(_Node(value=6))

    assert 6 not in _created
