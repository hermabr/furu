import json
import os
import re
import shutil
import threading
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from runs import ends_of, run_of
from tracedrepo import Repo, fresh_process

import furu
from furu import code_trace
from furu.code_trace import StaleCodeError, why_outdated
from furu.config import _Config, get_config
from furu.spec_metadata import Metadata
from furu.storage._layout import (
    events_path_in,
    run_dir_in,
    scratch_dir_in,
    trace_path_in,
)
from furu.storage.attempt_lock import current
from furu.testing import override_config
from furu.worker.execute import ChildSlot
from furu.worker.messages import JobCompletedResult, JobFailedResult

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


class LossPlot(furu.Spec):
    name: str
    title: ClassVar[str] = "loss"

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(code_version="traced")

    def create(self) -> str:
        values = [*style_axes(load_curves()), common.OTHER, REGISTRY["x"]]
        return f"{self.title}:{values}"


class Sibling(furu.Spec):
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


def _version_dir(spec: furu.Spec) -> Path:
    return run_of(spec).parent.parent


def _version(spec: furu.Spec) -> str:
    return _version_dir(spec).name


def _built_on_name(spec: furu.Spec) -> str:
    """How a trace names the code version that holds spec's result."""
    return f"{spec._stored_name}/{spec._class_part}/{_version(spec)}"


def _trace(spec: furu.Spec) -> dict[str, Any]:
    return json.loads(trace_path_in(_version_dir(spec)).read_text())


def test_traced_result_is_published_under_its_code_hash(
    repo: Repo, plots: ModuleType
) -> None:
    plot = plots.LossPlot(name="a")

    assert plot.create() == "loss:[12, 4, 99, 1]"

    assert _version(plot) != "fixed"
    assert not plot._attempt_dir.exists()
    assert sorted(p.name for p in _version_dir(plot).iterdir()) == [
        "runs",
        "trace.json",
    ]
    assert sorted(p.name for p in run_of(plot).iterdir()) == [
        "events.jsonl",
        "log",
        "result",
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
    assert _trace(plot)["built_on"] == []


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

    assert not plot.done
    plot = repo.load("plots").LossPlot(name="a")
    assert plot.create() != before_result
    assert _version(plot) != before
    # The old version is kept.
    assert run_dir_in(plot._class_dir / before, plot._artifact_hash).is_dir()


SWEEP = """\
import furu

SCALE = 2


class Sweep(furu.Spec):
    seed: int

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(code_version="traced")

    @furu.batched(lambda spec, worker: (None, 10))
    def create(specs, directory):
        return [spec.seed * SCALE for spec in specs]
"""


def test_a_traced_batch_is_one_run_found_under_the_live_code_version(
    repo: Repo,
) -> None:
    repo.write("sweep.py", SWEEP)
    sweep = repo.load("sweep").Sweep
    specs = [sweep(seed=1), sweep(seed=2)]
    assert furu.create(specs) == [2, 4]
    before = run_of(specs[0])
    assert run_of(specs[1]) == before
    assert _version(specs[0]) != "fixed"

    repo.edit("sweep.py", "SCALE = 2", "SCALE = 3")

    sweep = repo.load("sweep").Sweep
    specs = [sweep(seed=1), sweep(seed=2)]
    assert not specs[0].done
    assert furu.create(specs) == [3, 6]
    assert run_of(specs[0]).name == before.name
    assert _version_dir(specs[0]) != before.parent.parent


def test_rerun_names_the_changed_file(
    repo: Repo, plots: ModuleType, caplog: pytest.LogCaptureFixture
) -> None:
    plots.LossPlot(name="a").create()
    repo.edit("common.py", "CONST = 3", "CONST = 4")
    plot = repo.load("plots").LossPlot(name="a")

    plot.create()

    why = f"rerunning {plot.object_id}: {repo.package}/common.py changed"
    assert why in caplog.text


def test_a_trace_that_cannot_be_read_is_written_again_by_the_next_publish(
    plots: ModuleType,
) -> None:
    first = plots.LossPlot(name="a")
    first.create()
    trace_path_in(_version_dir(first)).write_text("")
    fresh_process()
    assert not first.done

    plots.LossPlot(name="b").create()

    assert first.done


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

    assert _version(edited) == original


def test_a_spec_under_two_live_versions_is_read_from_the_run_published_first(
    plots: ModuleType,
) -> None:
    plot = plots.LossPlot(name="a")
    plot.create()
    first = _version_dir(plot)
    # The same run under a second live version, whose folder sorts first.
    second = first.with_name("0" * 20)
    shutil.copytree(first, second)
    events = events_path_in(run_dir_in(second, plot._artifact_hash))
    events.write_text(re.sub(r"\d{4}-", "2999-", events.read_text()))
    fresh_process()

    assert _version_dir(plot) == first


DEPENDENCIES = """\
import furu

from PKG.child import Child, FixedChild


def parent_helper():
    return 100


class TracedParent(furu.Spec):
    n: int

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(code_version="traced")

    def create(self) -> int:
        return Child(n=self.n).create() + parent_helper()


class FixedParent(furu.Spec):
    n: int

    def create(self) -> int:
        return FixedChild(n=self.n).create() * 10


class FixedOnTraced(furu.Spec):
    n: int

    def create(self) -> int:
        return Child(n=self.n).create()


class Loader(furu.Spec):
    n: int
    through_spec: bool

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(code_version="traced")

    def create(self) -> int:
        child = Child(n=self.n)
        if self.through_spec:
            return child.load_existing()
        return furu.load_existing([child])[0]
"""

CHILD = """\
import furu


def child_helper(n):
    return n + 1


class Child(furu.Spec):
    n: int

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(code_version="traced")

    def create(self) -> int:
        return child_helper(self.n)


class FixedChild(furu.Spec):
    n: int

    def create(self) -> int:
        return child_helper(self.n)
"""


@pytest.fixture
def dependencies(repo: Repo) -> ModuleType:
    repo.write("child.py", CHILD)
    repo.write("parents.py", DEPENDENCIES)
    return repo.load("parents")


def test_a_restarted_parent_names_the_code_version_of_the_child_it_loaded(
    repo: Repo, dependencies: ModuleType
) -> None:
    parent = dependencies.TracedParent(n=1)

    assert parent.create() == 102

    # The missing child blocked the parent's first attempt; the rerun loaded it.
    assert [end["outcome"] for end in ends_of(parent)] == ["blocked", "ok"]
    assert not parent._attempt_dir.exists()
    child = dependencies.Child(n=1)
    trace = _trace(parent)
    assert trace["built_on"] == [_built_on_name(child)]
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


def test_a_traced_dependency_change_reruns_only_traced_consumers(
    repo: Repo, dependencies: ModuleType
) -> None:
    traced, fixed = dependencies.TracedParent(n=1), dependencies.FixedParent(n=1)
    assert (traced.create(), fixed.create()) == (102, 20)
    traced_before = _version(traced)
    # A class that does not trace its code has one version and no trace.
    assert sorted(path.name for path in fixed._class_dir.iterdir()) == [
        "attempts",
        "fixed",
        "schema.ndjson",
    ]
    assert [path.name for path in _version_dir(fixed).iterdir()] == ["runs"]

    repo.edit("child.py", "return n + 1", "return n + 2")

    assert not dependencies.Child(n=1).done
    assert not traced.done
    assert fixed.done  # its result changes only when its fields do

    traced = repo.load("parents").TracedParent(n=1)
    assert traced.create() == 103
    assert _version(traced) != traced_before

    repo.edit("child.py", "return n + 2", "return n + 1")

    assert _version(traced) == traced_before


def test_a_fixed_create_cannot_load_a_traced_result(dependencies: ModuleType) -> None:
    spec = dependencies.FixedOnTraced(n=1)

    # Its result would not follow Child's code, so it would outlive a fix.
    with pytest.raises(RuntimeError, match="whose class traces its code") as stopped:
        spec.create()

    assert "Trace FixedOnTraced too" in str(stopped.value)
    assert dependencies.Child(n=1).done  # building it is fine; loading it is not
    assert not spec.done


def test_a_fixed_spec_ignores_its_own_code(
    repo: Repo, dependencies: ModuleType
) -> None:
    fixed = dependencies.FixedParent(n=1)
    fixed.create()
    before = _version(fixed)

    repo.edit("parents.py", "* 10", "* 11")

    assert _version(fixed) == before


@pytest.mark.parametrize("through_spec", [True, False])
def test_a_result_loaded_with_load_existing_is_built_on_too(
    dependencies: ModuleType, through_spec: bool
) -> None:
    child = dependencies.Child(n=1)
    child.create()
    loader = dependencies.Loader(n=1, through_spec=through_spec)

    assert loader.create() == 2

    assert _trace(loader)["built_on"] == [_built_on_name(child)]


def test_delete_removes_runs_under_every_version_and_keeps_what_was_built_on_them(
    repo: Repo, dependencies: ModuleType
) -> None:
    parent, child = dependencies.TracedParent(n=1), dependencies.Child(n=1)
    parent.create()
    built_on = _version_dir(child)
    repo.edit("child.py", "return n + 1", "return n + 2")
    repo.load("child").Child(n=1).create()
    other = _version_dir(child)
    repo.edit("child.py", "return n + 2", "return n + 1")
    assert other != built_on

    assert child.delete(mode="force")
    fresh_process()

    for version in (built_on, other):
        assert sorted(path.name for path in version.iterdir()) == [
            "runs",
            "trace.json",
        ]
        assert list((version / "runs").iterdir()) == []
    assert not child.done
    assert parent.done

    # Without its trace.json the version cannot be checked.
    shutil.rmtree(built_on)
    fresh_process()

    assert not parent.done


CHAIN = """\
import furu


def load():
    return 1


class Audio(furu.Spec):
    def metadata(self) -> furu.Metadata:
        return furu.Metadata(code_version="traced")

    def create(self) -> int:
        return load()


class Train(furu.Spec):
    def metadata(self) -> furu.Metadata:
        return furu.Metadata(code_version="traced")

    def create(self) -> int:
        return Audio().create() + 10


class Plot(furu.Spec):
    def metadata(self) -> furu.Metadata:
        return furu.Metadata(code_version="traced")

    def create(self) -> int:
        return Train().create() + 100
"""


def test_an_upstream_edit_reaches_what_was_built_on_it_one_name_at_a_time(
    repo: Repo,
) -> None:
    repo.write("audio.py", CHAIN.partition("class Train")[0])
    repo.write(
        "train.py",
        "import furu\n\nfrom PKG.audio import Audio\n\n\nclass Train"
        + CHAIN.partition("class Train")[2].partition("class Plot")[0],
    )
    repo.write(
        "plot.py",
        "import furu\n\nfrom PKG.train import Train\n\n\nclass Plot"
        + CHAIN.partition("class Plot")[2],
    )
    plot = repo.load("plot").Plot()
    assert plot.create() == 111
    audio, train = repo.load("audio").Audio(), repo.load("train").Train()
    # A trace names the version it loaded from, not what that was built on.
    assert _trace(train)["built_on"] == [_built_on_name(audio)]
    assert _trace(plot)["built_on"] == [_built_on_name(train)]
    versions = [_version_dir(spec) for spec in (audio, train, plot)]
    names = [_built_on_name(spec) for spec in (audio, train)]

    repo.edit("audio.py", "return 1", "return 2")

    assert [why_outdated(version) for version in versions] == [
        f"{repo.package}/audio.py changed",
        f"it was built on {names[0]}, which is outdated",
        f"it was built on {names[1]}, which is outdated",
    ]
    assert not plot.done


def test_many_live_code_versions_warn_once(
    plots: ModuleType, caplog: pytest.LogCaptureFixture
) -> None:
    plot = plots.LossPlot(name="a")
    plot.create()
    for index in range(code_trace._MANY_LIVE_VERSIONS):
        copy = plot._class_dir / f"{index:020x}"
        copy.mkdir()
        shutil.copy(trace_path_in(_version_dir(plot)), trace_path_in(copy))

    with caplog.at_level("WARNING", logger="furu"):
        assert plot.done
        assert plots.LossPlot(name="b").done is False

    (record,) = [r for r in caplog.records if r.levelname == "WARNING"]
    assert record.getMessage().startswith(
        f"{plot._class_dir} has 101 live code versions"
    )


def test_parent_code_edit_reruns_only_the_parent(
    repo: Repo, dependencies: ModuleType
) -> None:
    dependencies.TracedParent(n=1).create()
    child_before = _version(dependencies.Child(n=1))

    repo.edit("parents.py", "return 100", "return 200")

    module = repo.load("parents")
    assert module.Child(n=1).done
    assert not module.TracedParent(n=1).done
    assert module.TracedParent(n=1).create() == 202
    assert _version(module.Child(n=1)) == child_before


DECLARED = """\
import furu

from PKG.child import Child, FixedChild


class TracedParent(furu.Spec):
    n: int

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(code_version="traced")

    @furu.dependency
    def child(self) -> Child:
        return Child(n=self.n)

    def create(self) -> int:
        return self.child.create() * 10


class FixedParent(furu.Spec):
    n: int

    @furu.dependency
    def child(self) -> FixedChild:
        return FixedChild(n=self.n)

    def create(self) -> int:
        return self.child.create() * 10


class Holder(furu.Spec):
    parent: FixedParent

    def create(self) -> int:
        return self.parent.create() + 1


class Search(furu.Spec):
    def metadata(self) -> furu.Metadata:
        return furu.Metadata(code_version="traced")

    def create(self) -> int:
        return FixedParent(n=3).create()
"""


@pytest.fixture
def declared(repo: Repo) -> ModuleType:
    repo.write("child.py", CHILD)
    repo.write("parents.py", DECLARED)
    return repo.load("parents")


@pytest.mark.parametrize("parent_class", ["TracedParent", "FixedParent"])
def test_changing_what_a_dependency_returns_reruns(
    repo: Repo, declared: ModuleType, parent_class: str
) -> None:
    assert getattr(declared, parent_class)(n=1).create() == 20

    repo.edit("parents.py", "Child(n=self.n)", "Child(n=self.n + 5)")

    parent = getattr(repo.load("parents"), parent_class)(n=1)
    assert not parent.done
    assert parent.create() == 70


@pytest.mark.parametrize("parent_class", ["TracedParent", "FixedParent"])
def test_a_dependency_refactor_returning_the_same_spec_keeps_the_result(
    repo: Repo, declared: ModuleType, parent_class: str
) -> None:
    parent = getattr(declared, parent_class)(n=1)
    parent.create()
    before = _version(parent)

    repo.edit("parents.py", "Child(n=self.n)", "Child(n=self.n * 1)")

    parent = getattr(repo.load("parents"), parent_class)(n=1)
    assert _version(parent) == before


def test_a_worker_calls_a_dependency_method_with_tracing_paused(
    repo: Repo, declared: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    from furu.worker.backends.local import LocalThreadWorkerBackend

    monkeypatch.setenv(
        "PYTHONPATH", f"{repo.root}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"
    )
    parent = declared.TracedParent(n=1)

    # The worker builds the spec from its line, so its create() is the first
    # to read the dependency. In this process the method runs before create()
    # starts, and it is in neither trace.
    furu.build([parent], on=[LocalThreadWorkerBackend()])

    ran = _trace(parent)["code"][f"{repo.package}/parents.py"]["ran"]
    assert ran == ["TracedParent.create"]


def test_a_traced_create_cannot_build_a_spec_that_has_a_dependency_method(
    declared: ModuleType,
) -> None:
    with pytest.raises(RuntimeError, match="built a FixedParent, whose") as stopped:
        declared.Search().create()
    assert f"{declared.__name__}.Search traces its code" in str(stopped.value)


def test_a_spec_held_in_a_field_reruns_when_its_dependency_changes(
    repo: Repo, declared: ModuleType
) -> None:
    assert declared.Holder(parent=declared.FixedParent(n=1)).create() == 21

    repo.edit("parents.py", "Child(n=self.n)", "Child(n=self.n + 5)")

    module = repo.load("parents")
    holder = module.Holder(parent=module.FixedParent(n=1))
    assert not holder.done
    assert holder.create() == 71


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


class Dynamic(furu.Spec):
    key: str

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(code_version="traced")

    def create(self) -> int:
        return by_name("SETTINGS")[self.key] + literal()


class Literal(furu.Spec):
    def metadata(self) -> furu.Metadata:
        return furu.Metadata(code_version="traced")

    def create(self) -> int:
        return literal()
"""


def test_dynamic_lookup_keeps_every_name_and_warns_once(
    repo: Repo, caplog: pytest.LogCaptureFixture
) -> None:
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
    assert caplog.text.count(warning) == 1  # once per function per process

    repo.edit("dynamic.py", "UNUSED = 1", "UNUSED = 2")

    assert not first.done


def test_literal_getattr_is_not_a_dynamic_lookup(
    repo: Repo, caplog: pytest.LogCaptureFixture
) -> None:
    repo.write("dynamic.py", DYNAMIC)
    spec = repo.load("dynamic").Literal()

    assert spec.create() == 2

    entry = _trace(spec)["code"][f"{repo.package}/dynamic.py"]
    assert entry["uses"] == ["Config", "furu"]
    assert "computed name" not in caplog.text
    repo.edit("dynamic.py", "UNUSED = 1", "UNUSED = 2")
    assert spec.done


UNTRACED = """\
import os
import sys
from pathlib import Path

import furu


class ReadsConfig(furu.Spec):
    def metadata(self) -> furu.Metadata:
        return furu.Metadata(code_version="traced")

    def create(self) -> str:
        here = Path(__file__).parent
        (here / "out.txt").write_text("written")  # a write is not an input
        for name in "abcde":
            (here / f"{name}.csv").read_text()
        return (here / "config.yaml").read_text()


class ReadsEnv(furu.Spec):
    def metadata(self) -> furu.Metadata:
        return furu.Metadata(code_version="traced")

    def create(self) -> str:
        return os.environ.get("HOME", "") + str(len(sys.argv))
"""


@pytest.mark.parametrize("snapshot", [False, True])
def test_reading_a_repo_file_warns_and_keeps_the_result(
    repo: Repo, snapshot: bool, caplog: pytest.LogCaptureFixture
) -> None:
    repo.write("config.yaml", "lr: 1\n")
    for name in "abcde":
        repo.write(f"{name}.csv", "")
    repo.write("untraced.py", UNTRACED)
    spec = repo.load("untraced").ReadsConfig()
    config = get_config()
    if snapshot:  # a snapshot worker's repo root is inside furu's data directories
        directories = config.directories.model_copy(
            update={"snapshots": repo.root.parent}
        )
        config = config.model_copy(update={"directories": directories})
    with override_config(config):
        spec.create()

    package = repo.package
    warning = (
        f"read {package}/a.csv, {package}/b.csv, {package}/c.csv "
        "and 3 more, which furu does not trace"
    )
    assert warning in caplog.text
    assert "out.txt" not in caplog.text
    assert f"{repo.package}/config.yaml" not in _trace(spec)["code"]

    repo.edit("config.yaml", "lr: 1", "lr: 2")
    assert spec.done


def test_reading_the_environment_warns(
    repo: Repo, caplog: pytest.LogCaptureFixture
) -> None:
    repo.write("untraced.py", UNTRACED)
    spec = repo.load("untraced").ReadsEnv()
    spec.create()

    assert (
        f"ReadsEnv.create ({repo.package}/untraced.py:25) reads os.environ, "
        "which furu does not trace"
    ) in caplog.text


def test_star_import_uses_every_name_in_the_module(repo: Repo) -> None:
    repo.write("consts.py", "A = 1\nB = 2\n")
    repo.write(
        "star.py",
        "import furu\n\nfrom PKG.consts import *\n\n\n"
        "class Star(furu.Spec):\n"
        "    def metadata(self) -> furu.Metadata:\n"
        '        return furu.Metadata(code_version="traced")\n\n'
        "    def create(self) -> int:\n"
        "        return A\n",
    )
    spec = repo.load("star").Star()
    spec.create()

    assert _trace(spec)["code"][f"{repo.package}/consts.py"]["uses"] == "all"
    repo.edit("consts.py", "B = 2", "B = 3")
    assert not spec.done


SCALED = """\
import furu

from PKG import helpers, registry  # noqa: F401
from PKG.helpers import offset


def double(x):
    return 2 * x


def compute_scale():
    return double(1)


def tag(cls):
    cls.label = "t"
    return cls


def build_a():
    return 1


def build_b():
    return 2


BUILDERS = {"a": build_a, "b": build_b}
SCALE = compute_scale() + offset() + helpers.bump()


def main():
    return "launch"


@tag
class Scaled(furu.Spec):
    def metadata(self) -> furu.Metadata:
        return furu.Metadata(code_version="traced")

    def create(self) -> str:
        return f"{SCALE}:{BUILDERS['a']()}:{self.label}:{helpers.TABLE}"


if __name__ == "__main__":
    main()
"""

HELPERS = """\
TABLE = []


def offset():
    return 10


def bump():
    return 100


def unrelated():
    return 0
"""


REGISTRY = """\
from PKG import helpers

helpers.TABLE.append(5)
"""


@pytest.fixture
def scaled(repo: Repo) -> ModuleType:
    repo.write("helpers.py", HELPERS)
    repo.write("registry.py", REGISTRY)
    repo.write("scaled.py", SCALED)
    return repo.load("scaled")


def test_functions_called_at_import_count_as_ran(
    repo: Repo, scaled: ModuleType
) -> None:
    spec = scaled.Scaled()

    assert spec.create() == "112:1:t:[5]"

    code = _trace(spec)["code"]
    assert code[f"{repo.package}/scaled.py"]["ran"] == [
        "Scaled.create",
        "build_a",
        "compute_scale",
        "double",
        "tag",
    ]
    assert code[f"{repo.package}/helpers.py"]["ran"] == ["bump", "offset"]


@pytest.mark.parametrize(
    ("file", "old", "new"),
    [
        pytest.param("scaled.py", "return double(1)", "return double(2)", id="call"),
        pytest.param("scaled.py", "2 * x", "3 * x", id="called-by-a-call"),
        pytest.param("scaled.py", '"t"', '"u"', id="decorator"),
        pytest.param("helpers.py", "return 10", "return 20", id="imported"),
        pytest.param("helpers.py", "return 100", "return 200", id="module-attribute"),
        pytest.param("registry.py", "append(5)", "append(6)", id="side-effect-import"),
    ],
)
def test_edits_to_code_run_at_import_rerun(
    repo: Repo, scaled: ModuleType, file: str, old: str, new: str
) -> None:
    spec = scaled.Scaled()
    before = spec.create()

    repo.edit(file, old, new)

    assert not spec.done
    assert repo.load("scaled").Scaled().create() != before


@pytest.mark.parametrize(
    ("file", "old", "new"),
    [
        pytest.param(
            "scaled.py",
            "def build_b():\n    return 2",
            "def build_b():\n    return 3",
            id="referenced-not-called",
        ),
        pytest.param("scaled.py", '"launch"', '"go"', id="named-by-main-block"),
        pytest.param(
            "scaled.py", "    main()\n", "    main()\n    main()\n", id="main"
        ),
        pytest.param("helpers.py", "return 0", "return 1", id="unrelated"),
    ],
)
def test_edits_to_code_not_run_at_import_keep_the_version(
    repo: Repo, scaled: ModuleType, file: str, old: str, new: str
) -> None:
    spec = scaled.Scaled()
    spec.create()
    before = _version(spec)

    repo.edit(file, old, new)

    assert _version(spec) == before


RESUMABLE = """\
import os

import furu


def step_one():
    return 1


def step_two():
    return 2


class Resumable(furu.Spec):
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
    assert not spec.done
    log = (current(spec._attempt_dir) / "trace.log").read_text()
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
    assert not spec._attempt_dir.exists()


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
    assert scratch_dir_in(current(spec._attempt_dir)).is_dir()

    repo.edit("resumable.py", "return 1", "return 10")
    spec = repo.load("resumable").Resumable(name="a")

    assert spec.create() == "resumed=True total=3"
    assert _version(spec) == "fixed"


def test_code_edited_after_import_is_not_published(
    repo: Repo, plots: ModuleType, caplog: pytest.LogCaptureFixture
) -> None:
    plots.LossPlot(name="a").create()
    common = repo.root / repo.package / "common.py"
    common.write_text(common.read_text().replace("CONST = 3", "CONST = 4"))

    plot = plots.LossPlot(name="b")
    with pytest.raises(StaleCodeError, match="common.py changed after this process"):
        plot.create()

    assert not plot.done
    # A fresh process imports the edited file and starts over, since the code
    # the failed attempt ran is not what is on disk.
    repo.edit("common.py", "CONST = 4", "CONST = 4")
    plot = repo.load("plots").LossPlot(name="b")
    assert plot.create() == "loss:[16, 4, 99, 1]"
    assert (
        f"starting {plot.object_id} over: {repo.package}/common.py changed"
        in caplog.text
    )


def test_touching_code_after_import_keeps_the_result(
    repo: Repo, plots: ModuleType
) -> None:
    plots.LossPlot(name="a").create()
    # A touch or a git checkout that lands on the same text changes only mtime.
    common = repo.root / repo.package / "common.py"
    common.write_text(common.read_text())

    plot = plots.LossPlot(name="b")
    assert plot.create() == "loss:[12, 4, 99, 1]"
    assert plot.done


def test_a_python_upgrade_reruns_and_says_so(
    repo: Repo,
    plots: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    plot = plots.LossPlot(name="a")
    plot.create()
    before = _version(plot)
    assert _trace(plot)["python"] == code_trace._PYTHON

    monkeypatch.setattr(code_trace, "_PYTHON", "3.99")
    fresh_process()
    assert not plot.done
    plot.create()

    assert _version(plot) != before
    assert f"rerunning {plot.object_id}: Python " in caplog.text
    assert "→ 3.99" in caplog.text


THREADED = """\
import threading

import furu


def in_thread(out):
    out.append(1)


class Threaded(furu.Spec):
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
    assert not spec.done
    assert repo.load("threaded").Threaded().create() == 2


TWO_THREADS = """\
import threading

import furu

A_IN, B_IN, A_OUT = threading.Event(), threading.Event(), threading.Event()


def helper():
    return 2


class A(furu.Spec):
    def metadata(self) -> furu.Metadata:
        return furu.Metadata(code_version="traced")

    def create(self) -> int:
        A_IN.set()
        B_IN.wait(0.3)
        return helper()


class B(A):
    def create(self) -> int:
        B_IN.set()
        A_OUT.wait(5)
        return 7


class Fixed(B):
    def metadata(self) -> furu.Metadata:
        return furu.Metadata(code_version="fixed")
"""


def test_a_traced_create_waits_for_the_one_another_thread_runs(repo: Repo) -> None:
    repo.write("two.py", TWO_THREADS)
    two = repo.load("two")
    a, b = two.A(), two.B()

    def second() -> None:
        two.A_IN.wait(5)
        b.create()

    thread = threading.Thread(target=second)
    thread.start()
    assert a.create() == 2
    two.A_OUT.set()
    thread.join()

    # B's create() started only when A's had returned, so each ran its own code.
    file = f"{repo.package}/two.py"
    assert "helper" in _trace(a)["code"][file]["ran"]
    assert "helper" not in _trace(b)["code"][file]["ran"]
    assert not code_trace.tracing()


def test_a_fixed_create_on_another_thread_leaves_a_traced_one_recording(
    repo: Repo,
) -> None:
    repo.write("two.py", TWO_THREADS)
    two = repo.load("two")
    a, fixed = two.A(), two.Fixed()

    def second() -> None:
        two.A_IN.wait(5)
        fixed.create()

    thread = threading.Thread(target=second)
    thread.start()
    assert a.create() == 2  # helper() runs while Fixed.create() does
    two.A_OUT.set()
    thread.join()

    assert "helper" in _trace(a)["code"][f"{repo.package}/two.py"]["ran"]
    assert not code_trace.tracing()


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
    slot = ChildSlot(worker="w", backend="local")
    monkeypatch.setattr(slot, "_run", lambda job, cancelled: results.pop(0))
    monkeypatch.setattr(slot, "close", lambda: closed.append(True))

    assert slot.run(object(), cancelled=threading.Event()) == JobCompletedResult()  # ty: ignore[invalid-argument-type]
    assert closed == [True]


def test_worker_child_publishes_the_version_the_submitter_checks(
    repo: Repo, plots: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    from furu.worker.backends.local import LocalThreadWorkerBackend

    monkeypatch.setenv(
        "PYTHONPATH", f"{repo.root}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"
    )
    plot = plots.LossPlot(name="worker")

    furu.build([plot], on=[LocalThreadWorkerBackend()])

    before = _version(plot)
    assert plot.load_existing() == "loss:[12, 4, 99, 1]"
    repo.edit("common.py", "CONST = 3", "CONST = 4")
    assert not plot.done

    furu.build([plot], on=[LocalThreadWorkerBackend()])

    assert _version(plot) != before
    assert plot.load_existing() == "loss:[16, 4, 99, 1]"
