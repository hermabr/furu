import importlib
import itertools
import json
import os
import sys
import threading
import uuid
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import furu
from furu.code_trace import StaleCodeError
from furu.config import _Config, get_config
from furu.migration.links import own_version
from furu.spec_metadata import Metadata
from furu.storage._layout import (
    attempt_dir_in,
    run_log_path_in,
    scratch_dir_in,
    trace_path_in,
)
from furu.testing import override_config
from furu.worker.execute import ChildSlot
from furu.worker.protocol import JobCompletedResult, JobFailedResult

# Source files get distinct mtimes in the past, so the stale-process guard
# (any traced file modified after this process started) stays quiet unless a
# test opts in, and every rewrite invalidates bytecode and parse caches.
_MTIMES = itertools.count(1_000_000_000)


@dataclass
class Repo:
    root: Path
    package: str

    def write(self, name: str, text: str) -> None:
        path = self.root / self.package / name
        path.write_text(text.replace("PKG", self.package), encoding="utf-8")
        mtime = next(_MTIMES) * 1_000_000_000
        os.utime(path, ns=(mtime, mtime))

    def read(self, name: str) -> str:
        text = (self.root / self.package / name).read_text(encoding="utf-8")
        return text.replace(self.package, "PKG")

    def edit(self, name: str, old: str, new: str) -> None:
        text = self.read(name)
        assert old in text, old
        self.write(name, text.replace(old, new))

    def load(self, name: str) -> ModuleType:
        """Import the package's current code, as a fresh process would."""
        for module in [m for m in sys.modules if m.split(".")[0] == self.package]:
            del sys.modules[module]
        importlib.invalidate_caches()
        return importlib.import_module(f"{self.package}.{name.removesuffix('.py')}")


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Repo]:
    root = tmp_path / "repo"
    repo = Repo(root=root, package=f"traced_{uuid.uuid4().hex[:12]}")
    (root / repo.package).mkdir(parents=True)
    repo.write("__init__.py", "")
    monkeypatch.setenv("_FURU_REPO_ROOT", str(root))
    monkeypatch.syspath_prepend(str(root))
    yield repo
    for module in [m for m in sys.modules if m.split(".")[0] == repo.package]:
        del sys.modules[module]


PLOTS = """\
from typing import ClassVar

import furu

from PKG import common
from PKG.common import load_curves
from PKG.train import TrainRun

BASE = 2
SCALE = BASE * 2
UNUSED = 7
REGISTRY = {}
REGISTRY["x"] = 1


def style_axes(values):
    return [value * SCALE for value in values]


def unused_helper(x):
    return x - 1


class LossPlot(furu.Spec[str]):
    name: str
    title: ClassVar[str] = "loss"

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(code_version="traced")

    def create(self) -> str:
        values = [*style_axes(load_curves()), common.OTHER, REGISTRY["x"]]
        return f"{self.title}:{values}"


class Sibling(furu.Spec[str]):
    lr: ClassVar[float] = 0.1

    def create(self) -> str:
        return TrainRun
"""

COMMON = """\
import math

CONST = 3
OTHER = 99
UNRELATED = 5


def load_curves():
    return [CONST, math.floor(1.5)]


def not_called():
    return UNRELATED
"""


@pytest.fixture
def plots(repo: Repo) -> ModuleType:
    repo.write("common.py", COMMON)
    repo.write("train.py", 'TrainRun = "train"\n')
    repo.write("plots.py", PLOTS)
    return repo.load("plots")


def _version(spec: furu.Spec) -> str:
    version = own_version(spec)
    assert version is not None, f"{spec._log_label} has no valid version"
    return version.name


def _trace(spec: furu.Spec) -> dict[str, Any]:
    version = own_version(spec)
    assert version is not None
    return json.loads(trace_path_in(version).read_text())


def _run_log(spec: furu.Spec) -> str:
    return run_log_path_in(spec._base_dir).read_text(encoding="utf-8")


def test_traced_result_is_published_under_its_code_hash(
    repo: Repo, plots: ModuleType
) -> None:
    plot = plots.LossPlot(name="a")

    assert plot.create() == "loss:[12, 4, 99, 1]"

    assert _version(plot).startswith("v-")
    assert not _version(plot).startswith("v-fixed")
    assert not attempt_dir_in(plot._base_dir).exists()
    assert {p.name for p in plot._base_dir.iterdir()} - {"compute.lock"} == {
        "spec.json",
        "run.log",
        _version(plot),
    }
    assert sorted(p.name for p in (plot._base_dir / _version(plot)).iterdir()) == [
        "provenance.json",
        "result",
        "trace.json",
    ]
    code = _trace(plot)["code"]
    package = repo.package
    assert set(code) == {f"{package}/plots.py", f"{package}/common.py"}
    assert code[f"{package}/plots.py"]["ran"] == ["LossPlot.create", "style_axes"]
    # A kept class keeps its header and attributes, so furu and ClassVar count.
    assert code[f"{package}/plots.py"]["uses"] == [
        "BASE",
        "ClassVar",
        "REGISTRY",
        "SCALE",
        "common",
        "furu",
        "load_curves",
    ]
    assert code[f"{package}/common.py"] == {
        "ran": ["load_curves"],
        "uses": ["CONST", "OTHER", "math"],
        "fingerprint": code[f"{package}/common.py"]["fingerprint"],
    }
    assert _trace(plot)["dependencies"] == {}


@pytest.mark.parametrize(
    ("file", "old", "new"),
    [
        pytest.param(
            "plots.py", "BASE = 2\n", "# hello\n\nBASE = 2  # base\n", id="comment"
        ),
        pytest.param(
            "plots.py",
            "[value * SCALE for value in values]",
            "[\n        value*SCALE\n        for value in values\n    ]",
            id="reformat",
        ),
        pytest.param(
            "plots.py",
            "def style_axes(values):\n",
            'def style_axes(values):\n    """Scale the axes."""\n',
            id="docstring",
        ),
        pytest.param("plots.py", "return x - 1", "return x - 100", id="unused-body"),
        pytest.param(
            "plots.py",
            "def unused_helper(x):",
            "def unused_helper(x, y=3):",
            id="unused-signature",
        ),
        pytest.param(
            "plots.py", "def unused_helper(", "def renamed_helper(", id="unused-rename"
        ),
        pytest.param(
            "plots.py",
            "class Sibling",
            "def brand_new(y):\n    return y**2\n\n\nclass Sibling",
            id="new-function",
        ),
        pytest.param(
            "plots.py",
            '    title: ClassVar[str] = "loss"\n',
            '    title: ClassVar[str] = "loss"\n\n    def other(self) -> int:\n        return 1\n',
            id="new-method",
        ),
        pytest.param(
            "plots.py",
            "from PKG.train import TrainRun\n",
            "from PKG.train import TrainRun as TR\n\nTrainRun = TR\n",
            id="sibling-only-import",
        ),
        pytest.param(
            "plots.py",
            "import furu\n",
            "import collections\n\nimport furu\n",
            id="unused-import",
        ),
        pytest.param("plots.py", "UNUSED = 7", "UNUSED = 8", id="unused-constant"),
        pytest.param(
            "plots.py",
            "lr: ClassVar[float] = 0.1",
            "lr: ClassVar[float] = 0.2",
            id="sibling-class-attribute",
        ),
        pytest.param(
            "plots.py", "return TrainRun", "return TrainRun * 2", id="sibling-create"
        ),
        pytest.param(
            "common.py",
            "UNRELATED = 5",
            "UNRELATED = 6",
            id="other-file-unused-constant",
        ),
        pytest.param(
            "common.py",
            "return UNRELATED",
            "return -UNRELATED",
            id="other-file-unused-function",
        ),
        pytest.param("train.py", '"train"', '"other"', id="unrelated-module"),
    ],
)
def test_edits_outside_what_ran_keep_the_version(
    repo: Repo, plots: ModuleType, file: str, old: str, new: str
) -> None:
    plot = plots.LossPlot(name="a")
    plot.create()
    before = _version(plot)

    repo.edit(file, old, new)

    assert plot.status == "done"
    assert _version(plot) == before


@pytest.mark.parametrize(
    ("file", "old", "new"),
    [
        pytest.param(
            "plots.py", "value * SCALE", "value * SCALE + 1", id="ran-function-body"
        ),
        pytest.param(
            "plots.py", "SCALE = BASE * 2", "SCALE = BASE * 3", id="used-constant"
        ),
        pytest.param(
            "plots.py", "BASE = 2", "BASE = 5", id="constant-used-through-another"
        ),
        pytest.param(
            "plots.py", 'REGISTRY["x"] = 1', 'REGISTRY["x"] = 2', id="side-effect"
        ),
        pytest.param(
            "plots.py",
            'title: ClassVar[str] = "loss"',
            'title: ClassVar[str] = "loss!"',
            id="class-attribute",
        ),
        pytest.param(
            "plots.py",
            "common.OTHER, REGISTRY",
            "common.OTHER, 0, REGISTRY",
            id="create",
        ),
        pytest.param(
            "common.py", "CONST = 3", "CONST = 4", id="imported-function-constant"
        ),
        pytest.param("common.py", "OTHER = 99", "OTHER = 98", id="module-attribute"),
    ],
)
def test_edits_to_what_ran_rerun(
    repo: Repo, plots: ModuleType, file: str, old: str, new: str
) -> None:
    plot = plots.LossPlot(name="a")
    before_result = plot.create()
    before = _version(plot)

    repo.edit(file, old, new)

    assert plot.status == "missing"
    plot = repo.load("plots").LossPlot(name="a")
    assert plot.status == "missing"
    assert plot.create() != before_result
    assert _version(plot) != before
    assert (plot._base_dir / before).is_dir()  # the old version is kept


def test_rerun_names_the_changed_file(repo: Repo, plots: ModuleType) -> None:
    plots.LossPlot(name="a").create()
    repo.edit("common.py", "CONST = 3", "CONST = 4")
    plot = repo.load("plots").LossPlot(name="a")

    plot.create()

    assert f"rerunning {plot._log_label}: {repo.package}/common.py changed" in _run_log(
        plot
    )


def test_revert_makes_the_old_version_valid_again(
    repo: Repo, plots: ModuleType
) -> None:
    plot = plots.LossPlot(name="a")
    plot.create()
    original = _version(plot)

    repo.edit("plots.py", "SCALE = BASE * 2", "SCALE = BASE * 3")
    edited = repo.load("plots").LossPlot(name="a")
    edited.create()
    assert _version(edited) != original

    repo.edit("plots.py", "SCALE = BASE * 3", "SCALE = BASE * 2")

    assert edited.status == "done"
    assert _version(edited) == original


DEPENDENCIES = """\
import furu

from PKG.child import Child


def parent_helper():
    return 100


class TracedParent(furu.Spec[int]):
    n: int

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(code_version="traced")

    def create(self) -> int:
        return Child(n=self.n).create() + parent_helper()


class FixedParent(furu.Spec[int]):
    n: int

    def create(self) -> int:
        return Child(n=self.n).create() * 10
"""

CHILD = """\
import furu


def child_helper(n):
    return n + 1


class Child(furu.Spec[int]):
    n: int

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(code_version="traced")

    def create(self) -> int:
        return child_helper(self.n)
"""


@pytest.fixture
def dependencies(repo: Repo) -> ModuleType:
    repo.write("child.py", CHILD)
    repo.write("parents.py", DEPENDENCIES)
    return repo.load("parents")


def test_a_restarted_parent_records_the_child_as_a_dependency_version(
    repo: Repo, dependencies: ModuleType
) -> None:
    parent = dependencies.TracedParent(n=1)

    assert parent.create() == 102

    # The missing child blocked the parent's first attempt; the rerun loaded it.
    assert f"blocked {parent._log_label} · 1 missing dependency" in _run_log(parent)
    assert not attempt_dir_in(parent._base_dir).exists()
    child = dependencies.Child(n=1)
    trace = _trace(parent)
    assert trace["dependencies"] == {
        child.object_id: str(Path(*child._fully_qualified_name.split(".")))
        + f"/{child._artifact_schema_hash}/{child._artifact_hash}/{_version(child)}"
    }
    # The child's create ran under its own trace; the parent only looked up
    # where the child is stored.
    assert trace["code"][f"{repo.package}/child.py"]["ran"] == ["Child.metadata"]
    assert trace["code"][f"{repo.package}/parents.py"]["ran"] == [
        "TracedParent.create",
        "parent_helper",
    ]
    assert _trace(child)["code"][f"{repo.package}/child.py"]["ran"] == [
        "Child.create",
        "child_helper",
    ]


def test_a_traced_dependency_change_reruns_traced_and_fixed_consumers(
    repo: Repo, dependencies: ModuleType
) -> None:
    traced, fixed = dependencies.TracedParent(n=1), dependencies.FixedParent(n=1)
    assert (traced.create(), fixed.create()) == (102, 20)
    fixed_before = _version(fixed)
    traced_before = _version(traced)
    assert fixed_before.startswith("v-fixed-")
    assert _trace(fixed) == {
        "code_version": "fixed",
        "dependencies": _trace(traced)["dependencies"],
    }

    repo.edit("child.py", "return n + 1", "return n + 2")

    assert dependencies.Child(n=1).status == "missing"
    assert traced.status == "missing"
    assert fixed.status == "missing"

    module = repo.load("parents")
    traced, fixed = module.TracedParent(n=1), module.FixedParent(n=1)
    assert (traced.create(), fixed.create()) == (103, 30)
    assert _version(fixed).startswith("v-fixed-")
    assert _version(fixed) != fixed_before
    assert _version(traced) != traced_before

    repo.edit("child.py", "return n + 2", "return n + 1")

    assert (_version(traced), _version(fixed)) == (traced_before, fixed_before)


def test_a_fixed_spec_ignores_its_own_code(
    repo: Repo, dependencies: ModuleType
) -> None:
    fixed = dependencies.FixedParent(n=1)
    fixed.create()
    before = _version(fixed)

    repo.edit("parents.py", "* 10", "* 11")

    assert fixed.status == "done"
    assert _version(fixed) == before


def test_parent_code_edit_reruns_only_the_parent(
    repo: Repo, dependencies: ModuleType
) -> None:
    dependencies.TracedParent(n=1).create()
    child_before = _version(dependencies.Child(n=1))

    repo.edit("parents.py", "return 100", "return 200")

    module = repo.load("parents")
    assert module.Child(n=1).status == "done"
    assert module.TracedParent(n=1).status == "missing"
    assert module.TracedParent(n=1).create() == 202
    assert _version(module.Child(n=1)) == child_before


def test_a_fixed_spec_without_traced_dependencies_lands_in_v_fixed(
    repo: Repo, plots: ModuleType
) -> None:
    sibling = plots.Sibling()

    assert sibling.create() == "train"

    assert _version(sibling) == "v-fixed"
    assert _trace(sibling) == {"code_version": "fixed", "dependencies": {}}


DYNAMIC = """\
import furu

UNUSED = 1
SETTINGS = {"lr": 3, "wd": 4}


class Config:
    lr = 2


def by_name(name):
    return globals()[name]


def literal():
    return getattr(Config, "lr")


class Dynamic(furu.Spec[int]):
    key: str

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(code_version="traced")

    def create(self) -> int:
        return by_name("SETTINGS")[self.key] + literal()


class Literal(furu.Spec[int]):
    def metadata(self) -> furu.Metadata:
        return furu.Metadata(code_version="traced")

    def create(self) -> int:
        return literal()
"""


def test_dynamic_lookup_keeps_every_name_and_warns_once(repo: Repo) -> None:
    repo.write("dynamic.py", DYNAMIC)
    module = repo.load("dynamic")
    first, second = module.Dynamic(key="lr"), module.Dynamic(key="wd")
    assert (first.create(), second.create()) == (5, 6)

    entry = _trace(first)["code"][f"{repo.package}/dynamic.py"]
    assert entry["uses"] == "all"
    assert entry["reason"] == "globals() in by_name"
    warning = (
        f"by_name ({repo.package}/dynamic.py:12) uses globals(), so every "
        f"module-level name in {repo.package}/dynamic.py counts toward its cache key"
    )
    assert warning in _run_log(first)
    assert warning not in _run_log(second)  # once per function per process

    repo.edit("dynamic.py", "UNUSED = 1", "UNUSED = 2")

    assert first.status == "missing"


def test_literal_getattr_is_not_a_dynamic_lookup(repo: Repo) -> None:
    repo.write("dynamic.py", DYNAMIC)
    spec = repo.load("dynamic").Literal()

    assert spec.create() == 2

    entry = _trace(spec)["code"][f"{repo.package}/dynamic.py"]
    assert entry["uses"] == ["Config", "furu"]
    assert "computed name" not in _run_log(spec)
    repo.edit("dynamic.py", "UNUSED = 1", "UNUSED = 2")
    assert spec.status == "done"


def test_star_import_uses_every_name_in_the_module(repo: Repo) -> None:
    repo.write("consts.py", "A = 1\nB = 2\n")
    repo.write(
        "star.py",
        "import furu\n\nfrom PKG.consts import *\n\n\n"
        "class Star(furu.Spec[int]):\n"
        "    def metadata(self) -> furu.Metadata:\n"
        '        return furu.Metadata(code_version="traced")\n\n'
        "    def create(self) -> int:\n"
        "        return A\n",
    )
    spec = repo.load("star").Star()
    spec.create()

    assert _trace(spec)["code"][f"{repo.package}/consts.py"]["uses"] == "all"
    repo.edit("consts.py", "B = 2", "B = 3")
    assert spec.status == "missing"


RESUMABLE = """\
import os

import furu


def step_one():
    return 1


def step_two():
    return 2


class Resumable(furu.Spec[str]):
    name: str

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(code_version="traced")

    def create(self) -> str:
        checkpoint = self.directory.scratch / "step-one"
        resumed = checkpoint.exists()
        if not resumed:
            checkpoint.write_text(str(step_one()))
        if os.environ.pop("FURU_TEST_PREEMPT", None):
            raise RuntimeError("preempted")
        return f"resumed={resumed} total={int(checkpoint.read_text()) + step_two()}"
"""


def test_preempted_traced_spec_resumes_when_its_code_is_unchanged(
    repo: Repo, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo.write("resumable.py", RESUMABLE)
    spec = repo.load("resumable").Resumable(name="a")
    monkeypatch.setenv("FURU_TEST_PREEMPT", "1")
    with pytest.raises(RuntimeError, match="preempted"):
        spec.create()
    assert spec.status == "failed"
    log = (attempt_dir_in(spec._base_dir) / "trace.log").read_text()
    assert '"ran", "' + f"{repo.package}/resumable.py" + '", "step_one"]' in log

    # Unrelated edits don't count: the code that ran is unchanged.
    repo.edit(
        "resumable.py", "def step_two", "def unrelated():\n    pass\n\n\ndef step_two"
    )
    spec = repo.load("resumable").Resumable(name="a")

    assert spec.create() == "resumed=True total=3"
    # The final trace is the union of both runs.
    assert _trace(spec)["code"][f"{repo.package}/resumable.py"]["ran"] == [
        "Resumable.create",
        "step_one",
        "step_two",
    ]
    assert not attempt_dir_in(spec._base_dir).exists()


def test_preempted_traced_spec_starts_over_when_code_it_ran_changed(
    repo: Repo, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo.write("resumable.py", RESUMABLE)
    spec = repo.load("resumable").Resumable(name="a")
    monkeypatch.setenv("FURU_TEST_PREEMPT", "1")
    with pytest.raises(RuntimeError, match="preempted"):
        spec.create()

    repo.edit("resumable.py", "return 1", "return 10")
    spec = repo.load("resumable").Resumable(name="a")

    assert spec.create() == "resumed=False total=12"


def test_fixed_retry_resumes_without_a_check(
    repo: Repo, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo.write(
        "resumable.py",
        RESUMABLE.replace('code_version="traced"', 'code_version="fixed"'),
    )
    spec = repo.load("resumable").Resumable(name="a")
    monkeypatch.setenv("FURU_TEST_PREEMPT", "1")
    with pytest.raises(RuntimeError, match="preempted"):
        spec.create()
    assert scratch_dir_in(attempt_dir_in(spec._base_dir)).is_dir()

    repo.edit("resumable.py", "return 1", "return 10")
    spec = repo.load("resumable").Resumable(name="a")

    assert spec.create() == "resumed=True total=3"
    assert _version(spec) == "v-fixed"


def test_code_edited_after_import_is_not_published(
    repo: Repo, plots: ModuleType
) -> None:
    plots.LossPlot(name="a").create()
    common = repo.root / repo.package / "common.py"
    common.write_text(common.read_text().replace("CONST = 3", "CONST = 4"))

    plot = plots.LossPlot(name="b")
    with pytest.raises(StaleCodeError, match="common.py changed after this process"):
        plot.create()

    assert plot.status == "failed"
    assert own_version(plot) is None
    # A fresh process imports the edited file and starts over, since the code
    # the failed attempt ran is not what is on disk.
    repo.edit("common.py", "CONST = 4", "CONST = 4")
    plot = repo.load("plots").LossPlot(name="b")
    assert plot.create() == "loss:[16, 4, 99, 1]"
    assert (
        f"starting {plot._log_label} over: {repo.package}/common.py changed while it ran"
        in _run_log(plot)
    )


THREADED = """\
import threading

import furu


def in_thread(out):
    out.append(1)


class Threaded(furu.Spec[int]):
    def metadata(self) -> furu.Metadata:
        return furu.Metadata(code_version="traced")

    def create(self) -> int:
        out = []
        thread = threading.Thread(target=in_thread, args=(out,))
        thread.start()
        thread.join()
        return out[0]
"""


def test_code_run_by_a_thread_inside_create_is_traced(repo: Repo) -> None:
    repo.write("threaded.py", THREADED)
    spec = repo.load("threaded").Threaded()

    assert spec.create() == 1

    assert "in_thread" in _trace(spec)["code"][f"{repo.package}/threaded.py"]["ran"]
    repo.edit("threaded.py", "out.append(1)", "out.append(2)")
    assert spec.status == "missing"
    assert repo.load("threaded").Threaded().create() == 2


def test_code_version_default_comes_from_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert get_config().code_version == "fixed"
    assert Metadata().code_version == "fixed"

    monkeypatch.setenv("FURU_CODE_VERSION", "traced")
    config = _Config()
    assert config.code_version == "traced"

    with override_config(config):
        assert Metadata().code_version == "traced"
        assert Metadata(code_version="fixed").code_version == "fixed"


def test_child_slot_retries_stale_code_in_a_fresh_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = [JobFailedResult(error="stale", stale_code=True), JobCompletedResult()]
    closed: list[bool] = []
    slot = ChildSlot(worker="w", backend="local", materialize_snapshot=False)
    monkeypatch.setattr(slot, "_run", lambda job, cancelled: results.pop(0))
    monkeypatch.setattr(slot, "close", lambda: closed.append(True))

    assert slot.run(object(), cancelled=threading.Event()) == JobCompletedResult()  # ty: ignore[invalid-argument-type]
    assert closed == [True]


def test_worker_child_publishes_the_version_the_submitter_checks(
    repo: Repo, plots: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    from furu.execution.execution_coordinator import ExecutionCoordinator
    from furu.worker.backends.local import LocalThreadWorkerBackend

    monkeypatch.setenv(
        "PYTHONPATH", f"{repo.root}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"
    )
    plot = plots.LossPlot(name="worker")

    ExecutionCoordinator.run([plot], worker_backends=(LocalThreadWorkerBackend(),))

    before = _version(plot)
    assert plot.load_existing() == "loss:[12, 4, 99, 1]"
    repo.edit("common.py", "CONST = 3", "CONST = 4")
    assert plot.status == "missing"

    ExecutionCoordinator.run([plot], worker_backends=(LocalThreadWorkerBackend(),))

    assert _version(plot) != before
    assert plot.load_existing() == "loss:[16, 4, 99, 1]"
