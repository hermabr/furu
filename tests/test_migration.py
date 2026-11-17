import inspect
import itertools
import json
import logging
import types
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import make_dataclass
from pathlib import Path
from typing import Any

import pytest

import furu
from furu import Added, MovedFrom, Renamed, Spec
from furu.config import get_config
from furu.execution.launch import stage_table
from furu.metadata import ArtifactSpec
from furu.migration import schema
from furu.migration.steps import MigrationError, naming
from furu.storage._layout import schema_path_in
from furu.worker.backends.local import LocalThreadWorkerBackend

_runs = itertools.count()


def _spec(
    name: str = "Merge",
    *,
    migrations: object = (),
    create: Callable[[Any], int] = lambda self: next(_runs),
    **fields: Any,
) -> Callable[..., Spec[int]]:
    """The spec class ``name`` as its source reads at one point in time.

    A field is ``name=type`` or ``name=(type, default)``, and a function is a
    ``@furu.dependency`` method. Each ``create()`` returns a new number, so a
    result that is found again is told apart from one that was computed.
    """
    body: dict[str, Any] = {
        "__module__": __name__,
        "__annotations__": {},
        "migrations": migrations,
        "create": create,
    }
    for field, declared in fields.items():
        if inspect.isfunction(declared):
            body[field] = furu.dependency(declared)
        elif isinstance(declared, tuple):
            body["__annotations__"][field], body[field] = declared
        else:
            body["__annotations__"][field] = declared
    return types.new_class(name, (Spec[int],), exec_body=lambda ns: ns.update(body))


def _nested(
    name: str = "Vimeo", *, migrations: object = (), **fields: Any
) -> Callable[..., Any]:
    """A plain dataclass a spec holds in a field, as ``_spec`` builds a spec class."""
    return make_dataclass(
        name,
        [
            (field, *declared) if isinstance(declared, tuple) else (field, declared)
            for field, declared in fields.items()
        ],
        namespace={"migrations": migrations},
        module=__name__,
        frozen=True,
    )


def _schema_lines(spec: Spec) -> list[dict[str, Any]]:
    text = schema_path_in(spec._base_dir.parent).read_text()
    return [json.loads(line) for line in text.splitlines() if line]


def test_an_added_field_is_left_out_of_the_id_at_its_default() -> None:
    before = _spec(lemma=str)
    after = _spec(
        lemma=str,
        min_overlap=(float, 0.5),
        migrations=(Added("min_overlap", default=0.5),),
    )
    stored = before(lemma="comer").create()

    assert after(lemma="comer").object_id == before(lemma="comer").object_id
    assert after(lemma="comer").create() == stored

    other = after(lemma="comer", min_overlap=0.8)
    assert other._class_part == before(lemma="comer")._class_part
    assert other.object_id != before(lemma="comer").object_id
    assert other.status == "missing"


def test_an_added_default_can_be_a_nan_or_a_spec() -> None:
    nan = float("nan")
    base = _spec("Base", size=int)
    before = _spec(lemma=str)
    after = _spec(
        lemma=str,
        score=(float, nan),
        base=(Any, base(size=1)),
        migrations=(Added("score", default=nan), Added("base", default=base(size=1))),
    )
    old = before(lemma="comer").object_id

    assert after(lemma="comer").object_id == old
    assert after(lemma="comer", base=base(size=2)).object_id != old


def test_a_renamed_field_is_fingerprinted_under_its_old_name() -> None:
    before = _spec(n=int)
    after = _spec(words_per_ngram=int, migrations=(Renamed("n", to="words_per_ngram"),))
    stored = before(n=3).create()

    assert after(words_per_ngram=3).object_id == before(n=3).object_id
    assert after(words_per_ngram=3).create() == stored


def test_steps_are_undone_newest_first() -> None:
    before = _spec(lemma=str)
    after = _spec(
        lemma=str,
        overlap=(float, 0.5),
        migrations=(
            Added("min_overlap", default=0.5),
            Renamed("min_overlap", to="overlap"),
        ),
    )

    assert after(lemma="comer").object_id == before(lemma="comer").object_id


def test_a_moved_class_keeps_its_old_name_in_ids_and_folders() -> None:
    old_name = f"{__name__}.StackExchangeRevisions"
    before = _spec("StackExchangeRevisions", site=str)
    after = _spec(
        "_StackExchangeRevisions", site=str, migrations=(MovedFrom(old_name),)
    )
    stored = before(site="math").create()

    assert after(site="math").object_id == before(site="math").object_id
    assert after(site="math").create() == stored

    new = after(site="physics")
    new.create()
    storage = new._metadata.storage
    assert new._base_dir.relative_to(storage).parts == (
        old_name,
        new._class_part,
        new._artifact_hash,
    )
    assert [path.name for path in storage.iterdir()] == [old_name]


class Relocated(Spec[int]):
    migrations = (MovedFrom("old_package.Original"),)
    x: int

    def create(self) -> int:
        return self.x


def test_a_moved_class_is_rebuilt_from_its_artifact() -> None:
    spec = Relocated(x=1)
    artifact = ArtifactSpec.from_furu(spec)

    assert artifact.object_id == spec.object_id
    assert spec.object_id.startswith("old_package.Original:")
    assert Spec.from_artifact(artifact) == spec


def test_a_breaking_step_is_a_fresh_start() -> None:
    steps = (
        Added("order", default="random"),
        Added("min_tokens", breaking=True),
        Added("shuffle", default=False),
    )
    fields: dict[str, Any] = {"tokenizer": str, "order": (str, "random")}
    before = _spec(**fields, migrations=steps[:1])
    fields |= {"min_tokens": (int, 1), "shuffle": (bool, False)}
    after = _spec(**fields, migrations=steps)
    cut = _spec(**fields, migrations=steps[2:])
    stored = before(tokenizer="bpe").create()

    assert after(tokenizer="bpe")._class_part != before(tokenizer="bpe")._class_part
    # The step explains the folder from before it, so the launch does not ask.
    assert after(tokenizer="bpe").create() != stored
    # The steps up to the breaking one are ignored, so they can be deleted.
    assert cut(tokenizer="bpe").object_id == after(tokenizer="bpe").object_id


def test_the_class_part_covers_names_only() -> None:
    before = _spec(n=int)

    assert _spec(n=int | None)(n=3).object_id == before(n=3).object_id
    assert _spec(n=int, seed=(int, 0))(n=3)._class_part != before(n=3)._class_part


def test_a_dependency_method_takes_the_same_steps_as_a_field() -> None:
    leaf = _spec("Hints", options=str)

    def stream_hints(self: Any) -> Spec | None:
        return None if self.hints is None else leaf(options=self.hints)

    steps = (Added("hints", default=None), Added("stream_hints", default=None))
    before = _spec(lemma=str)
    added = _spec(
        lemma=str,
        hints=(str | None, None),
        stream_hints=stream_hints,
        migrations=steps,
    )
    renamed = _spec(
        lemma=str,
        hints=(str | None, None),
        hint_files=stream_hints,
        migrations=(*steps, Renamed("stream_hints", to="hint_files")),
    )

    assert added(lemma="comer").object_id == before(lemma="comer").object_id
    assert added(lemma="comer", hints="x").object_id != before(lemma="comer").object_id
    assert (
        renamed(lemma="comer", hints="x").object_id
        == added(lemma="comer", hints="x").object_id
    )


def test_a_nested_class_declares_its_own_steps() -> None:
    video = _spec("Video", source=Any)
    old = video(source=_nested(clip=str)(clip="x")).object_id

    added = _nested(clip=str, fps=(int, 30), migrations=(Added("fps", default=30),))
    assert video(source=added(clip="x")).object_id == old
    assert video(source=added(clip="x", fps=24)).object_id != old

    renamed = _nested(path=str, migrations=(Renamed("clip", to="path"),))
    assert video(source=renamed(path="x")).object_id == old

    moved = _nested("RawVimeo", clip=str, migrations=(MovedFrom(f"{__name__}.Vimeo"),))
    assert video(source=moved(clip="x")).object_id == old
    assert video(source=_nested("RawVimeo", clip=str)(clip="x")).object_id != old


def test_a_breaking_step_on_a_nested_class_leaves_its_holders_steps_alone() -> None:
    plain = _nested(clip=str)(clip="x")
    broken = _nested(
        clip=str, quality=(Any, None), migrations=(Added("quality", breaking=True),)
    )(clip="x")
    before = _spec("Video", source=Any)
    after = _spec(
        "Video", source=Any, seed=(int, 0), migrations=(Added("seed", default=0),)
    )

    assert before(source=broken)._class_part == before(source=plain)._class_part
    assert before(source=broken).object_id != before(source=plain).object_id
    assert after(source=broken).object_id == before(source=broken).object_id


def test_a_class_folder_records_its_names_and_each_added_default() -> None:
    before = _spec(lemma=str)
    after = _spec(
        lemma=str,
        min_overlap=(float, 0.5),
        migrations=(Added("min_overlap", default=0.5),),
    )
    spec = before(lemma="comer")
    assert not schema_path_in(spec._base_dir.parent).exists()

    spec.create()
    assert _schema_lines(spec) == [{"names": ["lemma"]}]

    # A step is recorded the first time a process looks for the class's results.
    assert after(lemma="beber").status == "missing"
    assert _schema_lines(spec) == [
        {"names": ["lemma"]},
        {"added": "min_overlap", "default": 0.5},
    ]


def test_a_new_class_folder_records_every_default_before_its_first_result() -> None:
    def stream_hints(self: Any) -> None:
        return None

    fps = Added("fps", default=30)
    video = _spec(
        "Video",
        source=Any,
        seed=(int, 0),
        stream_hints=stream_hints,
        migrations=(Added("seed", default=0), Added("stream_hints", default=None)),
    )
    source = _nested(clip=str, fps=(int, 30), migrations=(fps,))(clip="x")
    spec = video(source=source)

    assert spec.status == "missing"
    assert not spec._base_dir.parent.exists()

    spec.create()
    assert _schema_lines(spec) == [
        {"names": ["source"]},
        {"added": "seed", "default": 0},
        {"added": "stream_hints", "default": None},
        {
            "added": "fps",
            "default": 30,
            "nested": f"{__name__}.Vimeo",
            "class_part": naming(type(source)).class_part,
        },
        {"nested": f"{__name__}.Vimeo", "at": ["source"], "names": ["clip"]},
    ]


class Failing(Spec[int]):
    lemma: str

    def create(self) -> int:
        raise RuntimeError("out of memory")


def test_a_class_folder_is_recorded_before_its_first_create_runs() -> None:
    spec = Failing(lemma="comer")

    with pytest.raises(RuntimeError, match="out of memory"):
        spec.create()
    assert _schema_lines(spec) == [{"names": ["lemma"]}]


def test_a_nested_default_is_recorded_for_one_class_part_of_the_nested_class() -> None:
    video = _spec("Video", source=Any)
    first = _nested(clip=str, fps=(int, 30), migrations=(Added("fps", default=30),))
    # The fresh start in between gave every holder a new id, so the field can
    # be added again at another default.
    again = _nested(
        clip=str,
        codec=(str, "h264"),
        fps=(int, 60),
        migrations=(Added("codec", breaking=True), Added("fps", default=60)),
    )

    video(source=first(clip="x")).create()
    video(source=again(clip="x")).create()


@pytest.mark.parametrize("nested", [False, True])
def test_an_edited_default_stops_with_the_step_and_both_values(
    nested: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    def video(default: int) -> Spec[int]:
        step = Added("fps", default=default)
        if nested:
            source = _nested(clip=str, fps=(int, default), migrations=(step,))
            return _spec("Video", source=Any)(source=source(clip="x"))
        return _spec("Video", clip=str, fps=(int, default), migrations=(step,))(
            clip="x"
        )

    video(30).create()

    # Naming the folder does not get past this stop.
    monkeypatch.setenv(
        "FURU_IGNORE_STORED", f"{__name__}.Video:{video(30)._class_part}"
    )
    with pytest.raises(MigrationError) as stopped:
        video(25).create()
    assert "Added('fps', default=25)" in str(stopped.value)
    assert "declares the default 25" in str(stopped.value)
    assert "recorded 30" in str(stopped.value)
    with pytest.raises(MigrationError):
        _ = video(25).status


def test_a_default_another_checkout_recorded_after_a_lookup_stops_the_next() -> None:
    def merge(default: float) -> Spec[int]:
        step = Added("min_overlap", default=default)
        return _spec(lemma=str, min_overlap=(float, default), migrations=(step,))(
            lemma="comer"
        )

    ours = merge(0.5)
    assert ours.status == "missing"

    merge(0.6).create()
    with pytest.raises(MigrationError, match="recorded 0.6"):
        _ = ours.status


def test_a_default_is_compared_with_a_line_another_process_added_meanwhile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    after = _spec(
        lemma=str,
        min_overlap=(float, 0.6),
        migrations=(Added("min_overlap", default=0.6),),
    )
    spec = _spec(lemma=str)(lemma="comer")
    spec.create()
    path = schema_path_in(spec._base_dir.parent)
    lock = schema.lock

    @contextmanager
    def another_process_first(lock_path: Path):
        with lock(lock_path) as has_lock:
            with path.open("a") as file:
                file.write('{"added":"min_overlap","default":0.5}\n')
            yield has_lock

    monkeypatch.setattr(schema, "lock", another_process_first)

    with pytest.raises(MigrationError, match="recorded 0.5"):
        _ = after(lemma="comer").status
    assert _schema_lines(spec) == [
        {"names": ["lemma"]},
        {"added": "min_overlap", "default": 0.5},
    ]


def test_a_line_cut_short_is_skipped_and_the_next_writer_starts_a_new_line() -> None:
    after = _spec(
        lemma=str,
        min_overlap=(float, 0.5),
        migrations=(Added("min_overlap", default=0.5),),
    )
    spec = _spec(lemma=str)(lemma="comer")
    spec.create()
    path = schema_path_in(spec._base_dir.parent)
    with path.open("a") as file:
        file.write('{"added":"min_overlap","defau')
    torn = path.read_text()

    assert after(lemma="comer").status == "done"
    assert path.read_text() == torn + '\n{"added":"min_overlap","default":0.5}\n'


def test_a_names_line_cut_short_is_written_again() -> None:
    spec = _spec(lemma=str)(lemma="comer")
    path = schema_path_in(spec._base_dir.parent)
    path.parent.mkdir(parents=True)
    path.write_text('{"names":["le')

    spec.create()
    assert path.read_text() == '{"names":["le\n{"names":["lemma"]}\n'


def _in_process(spec: Spec[int]) -> int:
    return spec.create()


def _on_workers(spec: Spec[int]) -> int:
    return furu.create(spec, on=[LocalThreadWorkerBackend()])


@pytest.mark.parametrize("launch", [_in_process, _on_workers])
def test_a_launch_asks_before_it_leaves_stored_results_behind(
    launch: Callable[[Spec[int]], int],
) -> None:
    old = _spec(lemma=str)(lemma="comer")
    old.create()
    new = _spec(lemma=str, min_overlap=(float, 0.5))(lemma="comer")
    table = "    stage   missing\n    Merge         1\n"

    with pytest.raises(MigrationError) as asks:
        launch(new)
    message = str(asks.value)
    assert table in message
    assert (
        f"Merge: {old._base_dir.parent} was stored without the field 'min_overlap'"
        in message
    )
    assert message.index("Added(name, default=...)") < message.index("breaking=True")
    assert message.endswith(
        f"FURU_IGNORE_STORED={__name__}.Merge:{old._class_part} for this one launch."
    )
    # Nothing ran, and nothing was stored.
    assert not new._base_dir.parent.exists()
    if launch is _on_workers:
        (log,) = get_config().directories.anchored().executions.glob("*/execution.log")
        assert table in log.read_text()

    # The step makes the old folder the class's own again.
    declared = _spec(
        lemma=str,
        min_overlap=(float, 0.5),
        migrations=(Added("min_overlap", default=0.5),),
    )
    declared(lemma="comer", min_overlap=0.8).create()


def test_the_variable_names_the_folders_one_launch_leaves_behind(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    first = _spec(lemma=str)(lemma="comer")
    second = _spec(lemma=str, min_overlap=(float, 0.5))
    third = _spec(lemma=str, seed=(int, 0))(lemma="comer")
    folders = [
        f"{__name__}.Merge:{spec._class_part}"
        for spec in (first, second(lemma="comer"), third)
    ]
    first.create()

    monkeypatch.setenv("FURU_IGNORE_STORED", folders[0])
    furu_logger = logging.getLogger("furu")
    monkeypatch.setattr(
        furu_logger, "handlers", [*furu_logger.handlers, caplog.handler]
    )
    second(lemma="comer").create()
    assert (
        f"ignoring stored results · Merge: {first._base_dir.parent} was stored "
        "without the field 'min_overlap', and this launch stores in "
        f"{second(lemma='comer')._base_dir.parent}"
    ) in caplog.messages
    # Nothing about the variable is stored in the class folder.
    assert _schema_lines(second(lemma="comer")) == [{"names": ["lemma", "min_overlap"]}]

    monkeypatch.setenv("FURU_IGNORE_STORED", ",".join(folders[:2]))
    third.create()

    # A folder that appeared later asks again.
    monkeypatch.setenv("FURU_IGNORE_STORED", folders[0])
    with pytest.raises(MigrationError) as asks:
        second(lemma="beber").create()
    assert f"{third._base_dir.parent} was stored" in str(asks.value)
    assert f"FURU_IGNORE_STORED={folders[0]},{folders[2]} " in str(asks.value)

    # A launch with nothing missing asks nothing.
    monkeypatch.delenv("FURU_IGNORE_STORED")
    second(lemma="comer").create()


def test_a_nested_class_is_compared_with_the_names_it_was_stored_with(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def line(nested: str, *names: str) -> dict[str, Any]:
        return {"nested": f"{__name__}.{nested}", "at": ["source"], "names": [*names]}

    video = _spec("Video", source=Any)
    youtube = _nested("Youtube", video_id=str)
    stored = video(source=_nested(clip=str)(clip="x"))
    stored.create()
    assert _schema_lines(stored) == [{"names": ["source"]}, line("Vimeo", "clip")]

    # A class the file has not seen gets a line: a union grows with no step.
    video(source=youtube(video_id="abc")).create()
    assert _schema_lines(stored)[2:] == [line("Youtube", "video_id")]

    changed = _nested(clip=str, fps=(int, 30))
    with pytest.raises(MigrationError) as asks:
        video(source=changed(clip="x")).create()
    assert (
        "Video: Vimeo at source was stored without the field 'fps' in "
        f"{stored._base_dir.parent}"
    ) in str(asks.value)
    folder = f"{__name__}.Video:{stored._class_part}"
    assert f"FURU_IGNORE_STORED={folder} " in str(asks.value)
    assert len(_schema_lines(stored)) == 3

    # Only the classes that the missing specs hold are compared.
    video(source=youtube(video_id="xyz")).create()

    # The folder of the spec class that holds it names the stored set.
    monkeypatch.setenv("FURU_IGNORE_STORED", folder)
    video(source=changed(clip="x")).create()
    assert _schema_lines(stored)[3:] == [line("Vimeo", "clip", "fps")]
    monkeypatch.delenv("FURU_IGNORE_STORED")
    with pytest.raises(MigrationError):
        video(source=changed(clip="y")).create()

    fresh = _nested(clip=str, fps=(int, 30), migrations=(Added("fps", breaking=True),))
    video(source=fresh(clip="y")).create()


def test_a_spec_built_inside_a_create_is_checked_where_it_is_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old = _spec("Run", peak=float)(peak=1.0)
    old.create()
    run = _spec("Run", peak=float, seed=(int, 0))
    search = _spec("Search", n=int, create=lambda self: run(peak=1.0).create())

    # The launch never saw the class, so there is no launch to let go on.
    monkeypatch.setenv("FURU_IGNORE_STORED", f"{__name__}.Run:{old._class_part}")
    with pytest.raises(MigrationError, match="a create.. built a spec") as failed:
        search(n=1).create()
    assert f"Run: {old._base_dir.parent} was stored" in str(failed.value)
    assert run(peak=1.0).status == "missing"

    # A class the launch asked about is not asked about again.
    furu.create([run(peak=2.0), search(n=1)])
    assert run(peak=1.0).status == "done"


def test_the_launch_table_has_a_row_for_each_stage_with_specs_missing() -> None:
    report = _spec("InfluenceReport", top_k=int)
    train = _spec("Train", seed=int)

    assert stage_table(
        [report(top_k=20), *(train(seed=seed) for seed in range(12))]
    ) == (
        "stage             missing\n"
        "InfluenceReport         1\n"
        "Train                  12"
    )


@pytest.mark.parametrize(
    ("declare", "problem"),
    [
        (lambda: Added("seed"), "exactly one of default="),
        (lambda: Added("seed", default=0, breaking=True), "exactly one of default="),
        (
            lambda: _spec(lemma=str, migrations=(Added("seed", default=0),)),
            "'seed' is not a field",
        ),
        (
            lambda: _spec(lemma=str, migrations=(Renamed("word", to="lema"),)),
            "'lema' is not a field",
        ),
        (
            lambda: _spec(
                lemma=str,
                word=(str, ""),
                migrations=(Renamed("word", to="lemma"), Added("word", default="")),
            ),
            "'word' already names a field",
        ),
        (
            lambda: types.new_class("Child", (Relocated,)),
            "Child inherits this step",
        ),
    ],
)
def test_a_step_that_does_not_fit_the_class_is_refused(
    declare: Callable[[], object], problem: str
) -> None:
    with pytest.raises(TypeError, match=problem):
        declare()
