"""Handing a pipeline over: an export of the record, and a rebuild from it."""

import re
import shutil
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import pytest
from gitrepo import git

import furu
from furu import Added, MovedFrom, Spec, plan, provenance
from furu.config import _FuruDirectories, get_config
from furu.testing import override_config
from furu.worker.backends.local import LocalThreadWorkerBackend


class Video(Spec[int]):
    migrations = (MovedFrom("old_videos.Video"), Added("fps", default=30))
    name: str
    fps: int = 30

    def create(self) -> int:
        return len(self.name)


class Clip(Spec[int]):
    video: Video
    start: int

    @furu.batched(lambda spec, worker: (None, 100))
    def create(specs: list["Clip"], directory: furu.SpecDirectory) -> list[int]:
        lengths = furu.load_existing([spec.video for spec in specs])
        return [length + spec.start for spec, length in zip(specs, lengths)]


class Probe(Spec[int]):
    seed: int

    @furu.dependency
    def hints(self) -> Video | None:
        return Video(name="hints") if self.seed else None

    def create(self) -> int:
        return self.seed


class Report(Spec[int]):
    video: Video
    notes: tuple[str, ...]

    @furu.dependency
    def clips(self) -> list[Clip]:
        return [Clip(video=self.video, start=start) for start in range(3)]

    @furu.dependency(remember=True)
    def best(self) -> Clip:
        return Clip(video=self.video, start=7)

    def create(self) -> int:
        # No field holds the probes: create() finds them, and loads them.
        probes = furu.create([Probe(seed=seed) for seed in range(2)])
        return sum(furu.load_existing([*self.clips, self.best])) + sum(probes)


def _pipeline() -> tuple[Report, list[Spec]]:
    """A report, and every spec it is built from."""
    video = Video(name="lecture")
    report = Report(video=video, notes=tuple(f"note {n}" for n in range(200)))
    probes = [Probe(seed=seed) for seed in range(2)]
    below = [video, *report.clips, report.best, *probes, Video(name="hints")]
    return report, [report, *below]


def _exported(path: Path) -> list[Spec]:
    """Build the pipeline, export its report to ``path``, and return its specs."""
    report, specs = _pipeline()
    assert report.create() == 39
    directories = get_config().directories.anchored()
    # An export reads the record only, where a launch that checks for pushed
    # code leaves this file too.
    shutil.rmtree(directories.plans)
    (directories.provenance / ".gitignore").write_text("*\n")
    furu.export(report, path)
    return specs


@contextmanager
def _empty_storage(folder: Path) -> Generator[None]:
    directories = _FuruDirectories(
        **{name: folder / name for name in _FuruDirectories.model_fields}
    )
    config = get_config().model_copy(update={"directories": directories})
    with override_config(config):
        yield


@pytest.mark.parametrize("on_workers", [False, True])
def test_an_export_is_rebuilt_to_the_same_ids_and_results(
    on_workers: bool, tmp_path: Path
) -> None:
    path = tmp_path / "report.export"
    specs = _exported(path)
    results = furu.load_existing(specs)

    # The file holds a line for each spec the report was built from: the ones
    # its fields and dependency methods hold, and the ones its create() found.
    header, *lines = plan.read(path)
    assert header["asked"] == [specs[0].object_id]
    assert header["code"]["commit"] == git(Path.cwd(), "rev-parse", "HEAD")
    assert {line["spec"] for line in lines if "spec" in line} == {
        spec.object_id for spec in specs
    }
    assert lines[-1] == {"moved": {"old_videos.Video": f"{__name__}.Video"}}

    with _empty_storage(tmp_path / "theirs"):
        furu.rebuild(path, on=[LocalThreadWorkerBackend()] if on_workers else None)

        _, rebuilt = _pipeline()
        assert furu.load_existing(rebuilt) == results


def test_a_line_that_no_longer_gives_its_id_stops_the_rebuild(tmp_path: Path) -> None:
    path = tmp_path / "report.export"
    report, video, *_ = _exported(path)
    lines = plan.read(path)

    def rebuild(name: str, old: str, new: str) -> None:
        changed = plan.File(tmp_path / name)
        changed.write(plan.dump(line).replace(old, new) for line in lines)
        changed.close()
        furu.rebuild(changed.path)

    with pytest.raises(ValueError, match=re.escape(f"a line for {video.object_id}")):
        rebuild("field", '"name":"lecture"', '"name":"seminar"')
    # A value written apart from the spec that holds it counts as its field.
    with pytest.raises(ValueError, match=re.escape(f"a line for {report.object_id}")):
        rebuild("value", '"note 7"', '"note 8"')


def test_an_export_needs_pushed_code_and_finished_specs(
    git_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, video = tmp_path / "video.export", Video(name="unbuilt")
    with pytest.raises(RuntimeError, match="has no line under"):
        furu.export(video, path)

    # Undo the furu pytest plugin's opt-out.
    monkeypatch.setattr(provenance, "_require_pushed_code", True)
    monkeypatch.chdir(git_repo)
    git(git_repo, "commit", "-q", "--allow-empty", "-m", "local")
    with pytest.raises(RuntimeError, match="not on any branch of origin"):
        furu.export(video, path)
    assert not path.exists()
