"""A launch's plan file, and the lines that describe its specs."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from runs import events_of

import furu
from furu import Added, MovedFrom, Renamed, Spec, plan
from furu.config import get_config
from furu.migration import schema
from furu.migration.steps import MigrationError
from furu.serializer import artifact
from furu.storage._layout import schema_path_in
from furu.worker.backends.local import LocalThreadWorkerBackend


@dataclass(frozen=True)
class Vimeo:
    migrations = (MovedFrom("leap.Vimeo"), Added("fps", default=30))
    clip: str
    fps: int = 30


class Video(Spec[int]):
    migrations = (Renamed("src", to="source"), Added("min_overlap", default=0.5))
    source: Vimeo
    min_overlap: float = 0.5

    def create(self) -> int:
        return 1


class Cifar(Spec[int]):
    split: str

    def create(self) -> int:
        return 1


class Triage(Spec[int]):
    lemma: str
    merges: tuple[Cifar, ...]

    def create(self) -> int:
        return len(self.merges)


class Train(Spec[int]):
    data: Cifar
    seed: int

    def create(self) -> int:
        return self.data.create() + self.seed


class LooTable(Spec[int]):
    data: Cifar
    seeds: int

    @furu.dependency
    def runs(self) -> list[Train]:
        return [Train(data=self.data, seed=seed) for seed in range(self.seeds)]

    def create(self) -> int:
        return sum(furu.load_existing(self.runs))


class InfluenceReport(Spec[int]):
    table: LooTable
    top_k: int

    def create(self) -> int:
        return self.table.create() + self.top_k


def _ref(spec: Spec) -> dict[str, Any]:
    return {"|kind": "spec", "|value": spec.object_id}


def _plans() -> set[Path]:
    return set(get_config().directories.anchored().plans.glob("*.plan"))


def test_a_line_holds_a_spec_as_it_was_hashed_and_notes_what_it_left_out() -> None:
    spec = Video(source=Vimeo(clip="x"))
    line, moved = plan.lines([spec])

    # Old names, and nothing for a field at its declared default.
    assert line == {
        "spec": spec.object_id,
        "fields": {
            "src": {
                "|kind": "instance",
                "|class": "leap.Vimeo",
                "|fields": {"clip": "x"},
            }
        },
        "left_out": {"min_overlap": 0.5},
    }
    assert moved == {"moved": {"leap.Vimeo": f"{__name__}.Vimeo"}}

    assert plan.Specs([line, moved])[spec.object_id] == spec
    # The field is filled from the declared step, and a line that says it was
    # left out at another value is refused.
    with pytest.raises(ValueError, match="left min_overlap out at 0.6"):
        plan.Specs([line | {"left_out": {"min_overlap": 0.6}}, moved])[spec.object_id]
    with pytest.raises(ValueError, match=r"fields \['seed', 'src'\] were written"):
        plan.Specs([line | {"fields": line["fields"] | {"seed": 1}}, moved])[
            spec.object_id
        ]


def test_a_spec_built_from_a_line_is_not_checked_against_the_recorded_default() -> None:
    spec = Video(source=Vimeo(clip="x"))
    spec.create()
    built = plan.Specs(plan.lines([spec]))[spec.object_id]
    recorded = schema_path_in(spec._class_dir)
    recorded.write_text(recorded.read_text().replace("0.5", "0.6"))
    schema._checked.clear()

    # The script that wrote the line checked.
    assert built.done
    with pytest.raises(MigrationError, match="recorded 0.6"):
        assert spec.done


def test_a_worker_builds_a_changed_class_from_its_line_and_records_what_it_holds() -> (
    None
):
    spec = Video(source=Vimeo(clip="on a worker"))

    assert furu.create(spec, on=[LocalThreadWorkerBackend()]) == 1

    recorded = schema_path_in(spec._class_dir).read_text().splitlines()
    assert {"nested": "leap.Vimeo", "at": ["source"], "names": ["clip"]} in [
        json.loads(line) for line in recorded if line
    ]


def test_a_value_over_1_kb_is_one_line_that_the_specs_holding_it_point_to(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    merges = tuple(Cifar(split=f"part-{part}") for part in range(20))
    triages = [Triage(lemma=lemma, merges=merges) for lemma in ("comer", "beber")]
    written_out = []

    def to_json(obj: object, *how: Any, **named: Any) -> Any:
        if obj is merges:
            written_out.append(obj)
        return real(obj, *how, **named)

    real = artifact.to_json
    monkeypatch.setattr(artifact, "to_json", to_json)

    assert furu.create(triages) == [20, 20]
    # Naming the second spec took the fingerprint the launch had for the first.
    assert len(written_out) == 1

    (launch,) = _plans()
    lines = plan.read(launch)[1:]
    (value,) = [line for line in lines if "value" in line]
    assert value["is"] == {"|kind": "tuple", "|value": [_ref(m) for m in merges]}
    pointer = {"|kind": "value", "|value": value["value"]}
    assert [line["fields"] for line in lines if "lemma" in line.get("fields", ())] == [
        {"lemma": "comer", "merges": pointer},
        {"lemma": "beber", "merges": pointer},
    ]
    assert plan.Specs(lines)[triages[1].object_id] == triages[1]


@pytest.mark.parametrize("on_workers", [False, True])
def test_a_launch_writes_what_it_runs_and_what_that_reaches_before_it_runs(
    on_workers: bool,
) -> None:
    on = [LocalThreadWorkerBackend()] if on_workers else None
    data = Cifar(split=f"train-{on_workers}")
    table = LooTable(data=data, seeds=2)
    report = InfluenceReport(table=table, top_k=20)

    assert furu.create(report, on=on) == 23
    (first,) = _plans()
    assert Path(events_of(report)[1]["plan"]) == first
    header, *lines = plan.read(first)
    assert header["asked"] == [report.object_id]
    assert header["code"].keys() == {"repository", "commit", "lock"}
    returned = {
        "value": table._returned["runs"],
        "is": [_ref(run) for run in table.runs],
    }
    assert returned in lines
    assert {line["spec"] for line in lines if line.get("runs")} == {
        spec.object_id for spec in (report, table, data, *table.runs)
    }

    # A launch with nothing to run writes nothing.
    assert furu.create(report, on=on) == 23
    assert _plans() == {first}

    # A finished spec reached through a field has a line, with one fingerprint
    # for what its dependency method returned. The runs have none.
    again = InfluenceReport(table=table, top_k=21)
    assert furu.create(again, on=on) == 24
    (second,) = _plans() - {first}
    assert plan.read(second)[1:] == [
        {
            "spec": again.object_id,
            "fields": {"table": _ref(table), "top_k": 21},
            "runs": True,
        },
        {
            "spec": table.object_id,
            "fields": {
                "data": _ref(data),
                "runs": {"|kind": "value", "|value": table._returned["runs"]},
                "seeds": 2,
            },
        },
        {"spec": data.object_id, "fields": {"split": data.split}},
    ]
