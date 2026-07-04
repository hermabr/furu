from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import furu
import furu.migration as migration_module
from furu import Added, MovedFrom, Renamed, Retyped, Rewrite, Spec, Stale
from furu.migration import MigrationError
from furu.storage._layout import (
    result_link_path_in,
    result_manifest_path_in,
)
from furu.utils import JsonValue, fully_qualified_name


class _Counter:
    def __init__(self) -> None:
        self.calls = 0


_COUNTER = _Counter()


@pytest.fixture(autouse=True)
def _reset_counter() -> None:
    _COUNTER.calls = 0


def _transplant_generation(donor: Spec[Any], target_cls: type[Spec[Any]]) -> Path:
    """Copy the donor's stored schema generation into the target class's tree.

    Old generations of a class live under its own fully-qualified-name tree; a
    class's source can only ever hold its current schema, so tests fabricate
    older generations by transplanting a donor class's store and rewriting the
    class name inside the recorded JSON.
    """
    donor_name = donor._fully_qualified_name
    target_name = fully_qualified_name(target_cls)
    source_schema_directory = donor._base_dir.parent
    target_schema_directory = (
        donor._storage_root
        / Path(*target_name.split("."))
        / source_schema_directory.name
    )
    shutil.copytree(source_schema_directory, target_schema_directory)
    for path in target_schema_directory.rglob("*.json"):
        path.write_text(path.read_text().replace(donor_name, target_name))
    return target_schema_directory


# --- rename + add through a class move -------------------------------------------


class _OldTrainRun(Spec[dict[str, str]]):
    learning_rate: float
    dataset: str

    def create(self) -> dict[str, str]:
        _COUNTER.calls += 1
        return {"dataset": self.dataset, "learning_rate": str(self.learning_rate)}


class _TrainRun(Spec[dict[str, str]]):
    dataset: str
    lr: float
    seed: int = 0

    migrations = (
        MovedFrom(fully_qualified_name(_OldTrainRun)),
        Renamed("learning_rate", to="lr"),
        Added("seed"),
    )

    def create(self) -> dict[str, str]:
        _COUNTER.calls += 1
        return {"dataset": self.dataset, "lr": str(self.lr), "seed": str(self.seed)}


def test_rename_plus_add_reuses_old_result_through_result_link() -> None:
    old = _OldTrainRun(learning_rate=0.001, dataset="cifar10")
    assert old.create() == {"dataset": "cifar10", "learning_rate": "0.001"}
    _COUNTER.calls = 0

    new = _TrainRun(dataset="cifar10", lr=0.001)
    assert new.create() == {"dataset": "cifar10", "learning_rate": "0.001"}
    assert _COUNTER.calls == 0

    link_path = result_link_path_in(new._base_dir)
    link = json.loads(link_path.read_text())
    assert link["source"]["base_dir"] == str(old._base_dir)
    assert link["source"]["schema_hash"] == old._artifact_schema_hash
    assert link["current"]["fully_qualified_name"] == new._fully_qualified_name
    assert link["current"]["schema_hash"] == new._artifact_schema_hash
    assert link["current"]["artifact_hash"] == new._artifact_hash
    assert link["migration_path"] == [
        f"MovedFrom({old._fully_qualified_name!r})",
        "Renamed('learning_rate', to='lr')",
        "Added('seed')",
    ]

    assert new.status == "done"
    assert new.load_existing() == {"dataset": "cifar10", "learning_rate": "0.001"}
    assert _COUNTER.calls == 0


def test_added_field_binds_only_the_default_value() -> None:
    _OldTrainRun(learning_rate=0.001, dataset="cifar10").create()

    assert _TrainRun(dataset="cifar10", lr=0.001).status == "done"
    # Old results correspond to the added field's default; any other value is a
    # different spec whose result genuinely never existed.
    assert _TrainRun(dataset="cifar10", lr=0.001, seed=7).status == "missing"


def test_no_matching_source_computes_fresh() -> None:
    _OldTrainRun(learning_rate=0.001, dataset="cifar10").create()
    _COUNTER.calls = 0

    new = _TrainRun(dataset="mnist", lr=0.5)
    assert new.status == "missing"
    assert new.create() == {"dataset": "mnist", "lr": "0.5", "seed": "0"}
    assert _COUNTER.calls == 1
    assert not result_link_path_in(new._base_dir).exists()


def test_load_raises_when_link_source_is_missing() -> None:
    old = _OldTrainRun(learning_rate=0.001, dataset="cifar10")
    old.create()
    new = _TrainRun(dataset="cifar10", lr=0.001)
    new.create()

    result_manifest_path_in(old._base_dir).unlink()

    with pytest.raises(RuntimeError, match="points to a missing result"):
        new.load_existing()


# --- multi-hop through an existing link -------------------------------------------


class _MidRun(Spec[dict[str, str]]):
    dataset: str
    lr: float

    migrations = (
        MovedFrom(fully_qualified_name(_OldTrainRun)),
        Renamed("learning_rate", to="lr"),
    )

    def create(self) -> dict[str, str]:
        _COUNTER.calls += 1
        return {"dataset": self.dataset, "lr": str(self.lr)}


class _FinalRun(Spec[dict[str, str]]):
    dataset: str
    lr: float
    seed: int = 0

    migrations = (
        MovedFrom(fully_qualified_name(_MidRun)),
        Added("seed"),
    )

    def create(self) -> dict[str, str]:
        _COUNTER.calls += 1
        return {"dataset": self.dataset, "lr": str(self.lr), "seed": str(self.seed)}


def test_multi_hop_link_points_directly_at_ultimate_source() -> None:
    old = _OldTrainRun(learning_rate=0.001, dataset="cifar10")
    old.create()
    _COUNTER.calls = 0

    mid = _MidRun(dataset="cifar10", lr=0.001)
    assert mid.create() == {"dataset": "cifar10", "learning_rate": "0.001"}

    final = _FinalRun(dataset="cifar10", lr=0.001)
    assert final.create() == {"dataset": "cifar10", "learning_rate": "0.001"}
    assert _COUNTER.calls == 0

    link = json.loads(result_link_path_in(final._base_dir).read_text())
    assert link["source"]["base_dir"] == str(old._base_dir)
    assert link["source"]["fully_qualified_name"] == old._fully_qualified_name
    assert link["migration_path"] == [
        f"MovedFrom({old._fully_qualified_name!r})",
        "Renamed('learning_rate', to='lr')",
        f"MovedFrom({mid._fully_qualified_name!r})",
        "Added('seed')",
    ]


# --- stale: orphaned generations block compute -------------------------------------


class _LegacyRun(Spec[dict[str, str]]):
    dataset: str
    learning_rate: float

    def create(self) -> dict[str, str]:
        _COUNTER.calls += 1
        return {"dataset": self.dataset, "learning_rate": str(self.learning_rate)}


class _EvolvedNoChain(Spec[dict[str, str]]):
    dataset: str
    lr: float

    def create(self) -> dict[str, str]:
        _COUNTER.calls += 1
        return {"dataset": self.dataset, "lr": str(self.lr)}


class _EvolvedWithChain(Spec[dict[str, str]]):
    dataset: str
    lr: float

    migrations = (Renamed("learning_rate", to="lr"),)

    def create(self) -> dict[str, str]:
        _COUNTER.calls += 1
        return {"dataset": self.dataset, "lr": str(self.lr)}


def test_orphaned_generation_is_stale_and_blocks_compute() -> None:
    legacy = _LegacyRun(dataset="cifar10", learning_rate=0.001)
    legacy.create()
    orphan_directory = _transplant_generation(legacy, _EvolvedNoChain)
    _COUNTER.calls = 0

    spec = _EvolvedNoChain(dataset="cifar10", lr=0.001)
    assert spec.status == "stale"

    with pytest.raises(Stale) as excinfo:
        spec.create()
    message = str(excinfo.value)
    assert str(orphan_directory) in message
    assert "- learning_rate" in message
    assert "+ lr" in message
    assert "migration chain" in message
    assert "deleting" in message
    assert _COUNTER.calls == 0

    with pytest.raises(Stale):
        spec.load_existing()
    with pytest.raises(Stale):
        furu.load_existing([spec])

    shutil.rmtree(orphan_directory)
    assert spec.status == "missing"
    assert spec.create() == {"dataset": "cifar10", "lr": "0.001"}
    assert _COUNTER.calls == 1


def test_declared_chain_lifts_the_stale_block() -> None:
    legacy = _LegacyRun(dataset="cifar10", learning_rate=0.001)
    legacy.create()
    _transplant_generation(legacy, _EvolvedWithChain)
    _COUNTER.calls = 0

    spec = _EvolvedWithChain(dataset="cifar10", lr=0.001)
    assert spec.status == "done"
    assert not result_link_path_in(spec._base_dir).exists()

    assert spec.create() == {"dataset": "cifar10", "learning_rate": "0.001"}
    assert _COUNTER.calls == 0
    assert result_link_path_in(spec._base_dir).exists()

    # A covered generation is not an orphan: unmatched specs are simply missing.
    assert _EvolvedWithChain(dataset="cifar10", lr=0.5).status == "missing"


# --- Retyped: widened unions ---------------------------------------------------


@dataclass(frozen=True)
class _SGD:
    momentum: float


@dataclass(frozen=True)
class _AdamW:
    beta: float


@dataclass(frozen=True)
class _Lion:
    alpha: float


class _WidenGen0(Spec[str]):
    optimizer: _SGD
    lr: float

    def create(self) -> str:
        _COUNTER.calls += 1
        return "gen0"


class _WidenGen1(Spec[str]):
    optimizer: _SGD | _AdamW
    lr: float

    def create(self) -> str:
        _COUNTER.calls += 1
        return "gen1"


class _WidenPartial(Spec[str]):
    optimizer: _SGD | _AdamW | _Lion
    lr: float

    migrations = (Retyped("optimizer", was=_SGD),)

    def create(self) -> str:
        _COUNTER.calls += 1
        return "current"


class _WidenFull(Spec[str]):
    optimizer: _SGD | _AdamW | _Lion
    lr: float

    migrations = (
        Retyped("optimizer", was=_SGD),
        Retyped("optimizer", was=_SGD | _AdamW),
    )

    def create(self) -> str:
        _COUNTER.calls += 1
        return "current"


def test_widened_union_reuses_old_results_through_retyped() -> None:
    gen0 = _WidenGen0(optimizer=_SGD(momentum=0.9), lr=0.1)
    assert gen0.create() == "gen0"
    _transplant_generation(gen0, _WidenFull)
    _COUNTER.calls = 0

    spec = _WidenFull(optimizer=_SGD(momentum=0.9), lr=0.1)
    assert spec.status == "done"
    assert spec.create() == "gen0"
    assert _COUNTER.calls == 0

    link = json.loads(result_link_path_in(spec._base_dir).read_text())
    assert len(link["migration_path"]) == 2
    assert all(
        step.startswith("Retyped('optimizer'") for step in link["migration_path"]
    )


def test_second_widening_leaves_intermediate_generation_stale() -> None:
    gen1 = _WidenGen1(optimizer=_AdamW(beta=0.5), lr=0.1)
    assert gen1.create() == "gen1"
    _transplant_generation(gen1, _WidenPartial)
    _COUNTER.calls = 0

    # _WidenPartial only declares was=_SGD: the intermediate _SGD | _AdamW
    # generation has no chain to current and blocks compute.
    spec = _WidenPartial(optimizer=_AdamW(beta=0.5), lr=0.1)
    assert spec.status == "stale"
    with pytest.raises(Stale):
        spec.create()
    assert _COUNTER.calls == 0

    # With its own was=_SGD | _AdamW line, the same generation is covered.
    _transplant_generation(gen1, _WidenFull)
    covered = _WidenFull(optimizer=_AdamW(beta=0.5), lr=0.1)
    assert covered.status == "done"
    assert covered.create() == "gen1"
    assert _COUNTER.calls == 0


class _DeadRetyped(Spec[str]):
    optimizer: _SGD
    lr: float

    migrations = (Retyped("optimizer", was=_SGD),)

    def create(self) -> str:
        return "current"


def test_retyped_without_type_change_is_a_dead_step() -> None:
    spec = _DeadRetyped(optimizer=_SGD(momentum=0.9), lr=0.1)
    with pytest.raises(MigrationError, match="dead step"):
        _ = spec.status
    with pytest.raises(MigrationError, match="dead step"):
        spec.create()


# --- ambiguity is rejected, never searched -----------------------------------------


class _AmbiguousChains(Spec[int]):
    n: int

    migrations = (
        MovedFrom(fully_qualified_name(_LegacyRun)),
        MovedFrom(fully_qualified_name(_LegacyRun)),
    )

    def create(self) -> int:
        return self.n


def test_two_chains_claiming_the_same_source_schema_are_rejected() -> None:
    with pytest.raises(MigrationError, match="ambiguous"):
        _ = _AmbiguousChains(n=1).status


def _identity_rewrite(fields: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return dict(fields)


class _AmbiguousViaRewrite(Spec[str]):
    optimizer: _SGD | _AdamW
    lr: float

    migrations = (
        Rewrite(_identity_rewrite),
        Retyped("optimizer", was=_SGD),
    )

    def create(self) -> str:
        return "current"


def test_snapshot_matching_two_chains_is_rejected() -> None:
    gen0 = _WidenGen0(optimizer=_SGD(momentum=0.9), lr=0.1)
    gen0.create()
    _transplant_generation(gen0, _AmbiguousViaRewrite)

    with pytest.raises(MigrationError, match="ambiguous"):
        _ = _AmbiguousViaRewrite(optimizer=_SGD(momentum=0.9), lr=0.1).status


# --- Rewrite: the fenced escape hatch ----------------------------------------------


class _RewriteDonor(Spec[dict[str, str]]):
    dataset: str
    version: float

    def create(self) -> dict[str, str]:
        _COUNTER.calls += 1
        return {"dataset": self.dataset, "version": str(self.version)}


def _version_to_int(fields: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    version = fields["version"]
    assert isinstance(version, float)
    return {"dataset": fields["dataset"], "version": int(version)}


class _RewrittenRun(Spec[dict[str, str]]):
    dataset: str
    version: int

    migrations = (Rewrite(_version_to_int),)

    def create(self) -> dict[str, str]:
        _COUNTER.calls += 1
        return {"dataset": self.dataset, "version": str(self.version)}


def test_rewrite_reshapes_values() -> None:
    donor = _RewriteDonor(dataset="cifar10", version=2.0)
    donor.create()
    _transplant_generation(donor, _RewrittenRun)
    _COUNTER.calls = 0

    spec = _RewrittenRun(dataset="cifar10", version=2)
    assert spec.create() == {"dataset": "cifar10", "version": "2.0"}
    assert _COUNTER.calls == 0


def _touches_unknown_field(fields: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {"dataset": fields["missing_field"], "version": 1}


class _RewriteWallsRun(Spec[dict[str, str]]):
    dataset: str
    version: int

    migrations = (Rewrite(_touches_unknown_field),)

    def create(self) -> dict[str, str]:
        return {}


def test_rewrite_walls_reject_unknown_source_fields() -> None:
    donor = _RewriteDonor(dataset="cifar10", version=2.0)
    donor.create()
    _transplant_generation(donor, _RewriteWallsRun)

    with pytest.raises(KeyError, match="not a source field"):
        _ = _RewriteWallsRun(dataset="cifar10", version=2).status


def _renames_a_field(fields: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {"dataset2": fields["dataset"], "version": fields["version"]}


class _RewriteRenamesRun(Spec[dict[str, str]]):
    dataset: str
    version: int

    migrations = (Rewrite(_renames_a_field),)

    def create(self) -> dict[str, str]:
        return {}


def test_rewrite_must_preserve_field_names() -> None:
    donor = _RewriteDonor(dataset="cifar10", version=2.0)
    donor.create()
    _transplant_generation(donor, _RewriteRenamesRun)

    with pytest.raises(MigrationError, match="must preserve field names"):
        _ = _RewriteRenamesRun(dataset="cifar10", version=2).status


# --- phase one: validation at class creation ---------------------------------------


def test_renamed_typo_fails_at_class_creation_naming_valid_fields() -> None:
    with pytest.raises(TypeError) as excinfo:

        class _BadRename(Spec[int]):
            lr: float

            migrations = (Renamed("learning_rate", to="lrx"),)

            def create(self) -> int:
                return 0

    message = str(excinfo.value)
    assert "'lrx' is not a field" in message
    assert "['lr']" in message


def test_added_typo_fails_at_class_creation() -> None:
    with pytest.raises(TypeError, match="'sed' is not a field"):

        class _BadAdded(Spec[int]):
            seed: int = 0

            migrations = (Added("sed"),)

            def create(self) -> int:
                return 0


def test_added_without_default_fails_at_class_creation() -> None:
    with pytest.raises(TypeError, match="has no default"):

        class _AddedNoDefault(Spec[int]):
            seed: int

            migrations = (Added("seed"),)

            def create(self) -> int:
                return 0


def test_retyped_typo_fails_at_class_creation() -> None:
    with pytest.raises(TypeError, match="'optimizr' is not a field"):

        class _BadRetyped(Spec[int]):
            optimizer: _SGD

            migrations = (Retyped("optimizr", was=_SGD),)

            def create(self) -> int:
                return 0


def test_migrations_must_be_a_tuple_of_steps() -> None:
    with pytest.raises(TypeError, match="must be a tuple"):

        class _ListMigrations(Spec[int]):
            n: int

            migrations = [Renamed("m", to="n")]

            def create(self) -> int:
                return 0


def test_chained_renames_walk_backward_through_the_changelog() -> None:
    class _ChainedRename(Spec[int]):
        c: int = 0

        migrations = (Renamed("a", to="b"), Renamed("b", to="c"))

        def create(self) -> int:
            return self.c

    assert len(_ChainedRename.migrations) == 2


def test_annotated_migrations_attribute_is_rejected() -> None:
    with pytest.raises(TypeError, match="migrations"):

        class _AnnotatedMigrations(Spec[int]):
            migrations: tuple = ()  # ty: ignore[invalid-attribute-override]

            def create(self) -> int:
                return 0


# --- the sideways scan is memoized --------------------------------------------------


class _ScanCounted(Spec[int]):
    n: int

    def create(self) -> int:
        return self.n


def test_sideways_scan_runs_once_per_class_per_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scans: list[Path] = []
    real_list = migration_module._list_schema_directories

    def counting(tree_directory: Path) -> list[Path]:
        scans.append(tree_directory)
        return real_list(tree_directory)

    monkeypatch.setattr(migration_module, "_list_schema_directories", counting)

    spec = _ScanCounted(n=1)
    assert spec.status == "missing"
    assert spec.status == "missing"
    assert _ScanCounted(n=2).status == "missing"
    assert len(scans) == 1

    assert spec.create() == 1
    assert len(scans) == 1

    # The scan never runs when a result manifest exists.
    assert spec.status == "done"
    assert _ScanCounted(n=1).status == "done"
    assert len(scans) == 1
