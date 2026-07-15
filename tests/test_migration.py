from __future__ import annotations

import errno
import json
import shutil
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest

import furu
import furu.migration.links as migration_links
import furu.migration.resolution as migration_resolution
from furu import Added, MovedFrom, Renamed, Retyped, Rewrite, Spec, Stale
from furu.migration.steps import MigrationError
from furu.result.codec import Codec
from furu.storage._layout import (
    compute_lock_path_in,
    result_dir_in,
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


def _transplant_generation(
    donor: Spec[Any],
    target_cls: type[Spec[Any]],
    renames: Mapping[str, str] | None = None,
) -> Path:
    """Copy the donor's stored schema generation into the target class's tree.

    Old generations of a class live under its own fully-qualified-name tree; a
    class's source can only ever hold its current schema, so tests fabricate
    older generations by transplanting a donor class's store and rewriting the
    class name inside the recorded JSON. Extra ``renames`` map embedded donor
    class names to their current spelling the same way.
    """
    donor_name = donor._fully_qualified_name
    target_name = fully_qualified_name(target_cls)
    source_schema_directory = donor._base_dir.parent
    target_schema_directory = (
        donor._metadata.storage
        / Path(*target_name.split("."))
        / source_schema_directory.name
    )
    shutil.copytree(source_schema_directory, target_schema_directory)
    for path in target_schema_directory.rglob("*.json"):
        text = path.read_text()
        for old, new in {donor_name: target_name, **(renames or {})}.items():
            text = text.replace(old, new)
        path.write_text(text)
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
        Added("seed", default=0),
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
        "Added('seed', default=0)",
    ]

    assert new.status == "done"
    assert new.load_existing() == {"dataset": "cifar10", "learning_rate": "0.001"}
    assert _COUNTER.calls == 0


def test_result_link_creation_rechecks_after_compute_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _OldTrainRun(learning_rate=0.001, dataset="cifar10")
    source.create()
    target = _TrainRun(dataset="cifar10", lr=0.001)
    link_path = result_link_path_in(target._base_dir)

    assert migration_links.result_dir_for_loading(target) == result_dir_in(
        source._base_dir
    )
    link_text = link_path.read_text(encoding="utf-8")
    link_path.unlink()

    @contextmanager
    def competing_writer(lock_path: Path):
        assert lock_path == compute_lock_path_in(target._base_dir)
        link_path.write_text(link_text, encoding="utf-8")
        yield lambda: True

    monkeypatch.setattr(migration_links, "lock", competing_writer)
    monkeypatch.setattr(
        migration_links,
        "atomic_write_text",
        lambda *_: pytest.fail("the competing worker's valid link was replaced"),
    )

    assert migration_links.result_dir_for_loading(target) == result_dir_in(
        source._base_dir
    )


def test_result_link_read_gracefully_reresolves_estale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _OldTrainRun(learning_rate=0.001, dataset="cifar10")
    source.create()
    target = _TrainRun(dataset="cifar10", lr=0.001)
    assert migration_links.result_dir_for_loading(target) is not None
    link_path = result_link_path_in(target._base_dir)
    path_type = type(link_path)
    read_text = path_type.read_text
    stale = True

    def stale_once(
        path: Path, encoding: str | None = None, errors: str | None = None
    ) -> str:
        nonlocal stale
        if path == link_path and stale:
            stale = False
            raise OSError(errno.ESTALE, "stale file handle")
        return read_text(path, encoding=encoding, errors=errors)

    monkeypatch.setattr(path_type, "read_text", stale_once)

    assert migration_links.result_dir_for_loading(target) == result_dir_in(
        source._base_dir
    )


class _PathValue:
    def __init__(self, path: Path) -> None:
        self.path = path


class _PathValueCodec(Codec[_PathValue]):
    @classmethod
    def matches(cls, value: object) -> bool:
        return isinstance(value, _PathValue)

    @classmethod
    def save(
        cls, value: _PathValue, artifact_directory: Path
    ) -> Mapping[str, object]:
        return {"path": value.path}

    @classmethod
    def load(
        cls, metadata: Mapping[str, object], artifact_directory: Path
    ) -> _PathValue:
        path = cast(Path, metadata["path"])
        path.read_text(encoding="utf-8")
        return _PathValue(path)


class _OldPathResult(Spec[_PathValue]):
    key: str
    result_codecs = (_PathValueCodec,)

    def create(self) -> _PathValue:
        path = self.directory.data / "payload.txt"
        path.write_text(self.key, encoding="utf-8")
        return _PathValue(path)


class _MigratedPathResult(Spec[_PathValue]):
    key: str
    version: int = 1
    result_codecs = (_PathValueCodec,)
    migrations = (
        MovedFrom(fully_qualified_name(_OldPathResult)),
        Added("version", default=1),
    )

    def create(self) -> _PathValue:
        raise AssertionError("migrated result should be loaded from cache")


def test_migrated_codec_metadata_path_uses_source_data_directory() -> None:
    source = _OldPathResult(key="contents")
    source.create()
    migrated = _MigratedPathResult(key="contents")

    assert migrated.create().path == (source.directory.data / "payload.txt").resolve()
    assert migrated.load_existing().path == (
        source.directory.data / "payload.txt"
    ).resolve()


def test_added_field_binds_only_the_default_value() -> None:
    _OldTrainRun(learning_rate=0.001, dataset="cifar10").create()

    assert _TrainRun(dataset="cifar10", lr=0.001).status == "done"
    # Old results correspond to the migration's pinned default; any other value
    # is a different spec whose result genuinely never existed.
    assert _TrainRun(dataset="cifar10", lr=0.001, seed=7).status == "missing"


def test_no_matching_source_computes_fresh() -> None:
    _OldTrainRun(learning_rate=0.001, dataset="cifar10").create()
    _COUNTER.calls = 0

    new = _TrainRun(dataset="mnist", lr=0.5)
    assert new.status == "missing"
    assert new.create() == {"dataset": "mnist", "lr": "0.5", "seed": "0"}
    assert _COUNTER.calls == 1
    assert not result_link_path_in(new._base_dir).exists()


@pytest.mark.parametrize(
    "delete_schema_directory",
    [False, True],
    ids=["source-artifact", "source-schema-directory"],
)
def test_dangling_link_is_missing_and_recomputes(
    delete_schema_directory: bool,
) -> None:
    old = _OldTrainRun(learning_rate=0.001, dataset="cifar10")
    old.create()
    new = _TrainRun(dataset="cifar10", lr=0.001)
    new.create()
    _COUNTER.calls = 0

    deleted = old._base_dir.parent if delete_schema_directory else old._base_dir
    shutil.rmtree(deleted)

    assert new.status == "missing"
    with pytest.raises(furu.Missing, match="could not find a result"):
        new.load_existing()

    assert new.create() == {"dataset": "cifar10", "lr": "0.001", "seed": "0"}
    assert _COUNTER.calls == 1
    assert result_manifest_path_in(new._base_dir).exists()
    assert not result_link_path_in(new._base_dir).exists()


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
        Added("seed", default=0),
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
        "Added('seed', default=0)",
    ]


def test_multi_hop_scan_ignores_a_dangling_intermediate_link() -> None:
    old = _OldTrainRun(learning_rate=0.001, dataset="cifar10")
    old.create()
    mid = _MidRun(dataset="cifar10", lr=0.001)
    mid.create()
    shutil.rmtree(old._base_dir)
    _COUNTER.calls = 0

    final = _FinalRun(dataset="cifar10", lr=0.001)
    assert final.status == "missing"
    assert final.create() == {"dataset": "cifar10", "lr": "0.001", "seed": "0"}
    assert _COUNTER.calls == 1
    assert not result_link_path_in(final._base_dir).exists()


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


# --- breaking: old results are superseded, never reused or stale --------------------


class _BreakingSeedRun(Spec[dict[str, str]]):
    dataset: str
    learning_rate: float
    seed: int  # no field default needed: old results are void, new runs must choose

    migrations = (Added("seed", breaking=True),)

    def create(self) -> dict[str, str]:
        _COUNTER.calls += 1
        return {"dataset": self.dataset, "seed": str(self.seed)}


def test_breaking_added_supersedes_old_results() -> None:
    legacy = _LegacyRun(dataset="cifar10", learning_rate=0.001)
    legacy.create()
    old_directory = _transplant_generation(legacy, _BreakingSeedRun)
    _COUNTER.calls = 0

    spec = _BreakingSeedRun(dataset="cifar10", learning_rate=0.001, seed=1)
    assert spec.status == "missing"
    assert spec.create() == {"dataset": "cifar10", "seed": "1"}
    assert _COUNTER.calls == 1
    assert not result_link_path_in(spec._base_dir).exists()
    # Superseded data is never deleted by furu; cleanup stays a deliberate act.
    assert old_directory.exists()


class _PostBreakDonor(Spec[str]):
    dataset: str
    learning_rate: float
    seed: int

    def create(self) -> str:
        _COUNTER.calls += 1
        return "post-break-donor"


class _PostBreakRun(Spec[str]):
    dataset: str
    lr: float
    seed: int = 0

    migrations = (
        Added("seed", breaking=True),
        Renamed("learning_rate", to="lr"),
    )

    def create(self) -> str:
        _COUNTER.calls += 1
        return "recomputed"


def test_break_kills_earlier_generations_while_later_steps_migrate() -> None:
    donor = _PostBreakDonor(dataset="cifar10", learning_rate=0.001, seed=0)
    donor.create()
    _transplant_generation(donor, _PostBreakRun)
    legacy = _LegacyRun(dataset="legacy-only", learning_rate=0.5)
    legacy.create()
    _transplant_generation(legacy, _PostBreakRun)
    _COUNTER.calls = 0

    # The pre-break generation is recognized (so not stale) but its results are
    # dead: the exact legacy spec recomputes fresh.
    resurrected = _PostBreakRun(dataset="legacy-only", lr=0.5)
    assert resurrected.status == "missing"
    assert resurrected.create() == "recomputed"
    assert _COUNTER.calls == 1

    # The post-break generation still migrates normally through Renamed.
    covered = _PostBreakRun(dataset="cifar10", lr=0.001)
    assert covered.status == "done"
    assert covered.create() == "post-break-donor"
    assert _COUNTER.calls == 1


class _BreakingRetypedRun(Spec[str]):
    optimizer: _SGD | _AdamW
    lr: float

    migrations = (Retyped("optimizer", was=_SGD, breaking=True),)

    def create(self) -> str:
        _COUNTER.calls += 1
        return "recomputed"


def test_breaking_retyped_supersedes_instead_of_reusing() -> None:
    gen0 = _WidenGen0(optimizer=_SGD(momentum=0.9), lr=0.1)
    assert gen0.create() == "gen0"
    _transplant_generation(gen0, _BreakingRetypedRun)
    _COUNTER.calls = 0

    spec = _BreakingRetypedRun(optimizer=_SGD(momentum=0.9), lr=0.1)
    assert spec.status == "missing"
    assert spec.create() == "recomputed"
    assert _COUNTER.calls == 1


# --- Added binds history to the pinned default, not the field default ---------------


class _RetriesDonor(Spec[str]):
    dataset: str

    def create(self) -> str:
        _COUNTER.calls += 1
        return "old-run"


class _RetriesRun(Spec[str]):
    dataset: str
    num_retries: int = 3  # new runs default to 3; old runs behaved like 1

    migrations = (Added("num_retries", default=1),)

    def create(self) -> str:
        _COUNTER.calls += 1
        return "recomputed"


def test_added_default_pins_history_independently_of_the_field_default() -> None:
    donor = _RetriesDonor(dataset="cifar10")
    donor.create()
    _transplant_generation(donor, _RetriesRun)
    _COUNTER.calls = 0

    assert _RetriesRun(dataset="cifar10", num_retries=1).status == "done"
    assert _RetriesRun(dataset="cifar10", num_retries=1).create() == "old-run"
    assert _RetriesRun(dataset="cifar10").status == "missing"
    assert _COUNTER.calls == 0


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

            migrations = (Added("sed", default=0),)

            def create(self) -> int:
                return 0


def test_added_without_default_fails_at_class_creation() -> None:
    with pytest.raises(TypeError, match="needs default="):

        class _AddedNoDefault(Spec[int]):
            seed: int

            migrations = (Added("seed"),)

            def create(self) -> int:
                return 0


def test_breaking_added_with_default_fails_at_class_creation() -> None:
    with pytest.raises(TypeError, match="never backfill"):

        class _BreakingWithDefault(Spec[int]):
            seed: int = 0

            migrations = (Added("seed", default=0, breaking=True),)

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
    scans: list[type] = []
    real_resolve = migration_resolution._resolve_class

    def counting(obj: Spec[Any]) -> migration_resolution._ClassResolution:
        scans.append(type(obj))
        return real_resolve(obj)

    monkeypatch.setattr(migration_resolution, "_resolve_class", counting)

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


# --- cascading: a child chain carries every spec that embeds it ---------------------


class _CascadeTokenizerV0(Spec[dict[str, int]]):
    vocabulary_size: int

    def create(self) -> dict[str, int]:
        _COUNTER.calls += 1
        return {"vocabulary_size": self.vocabulary_size}


class _CascadeTokenizer(Spec[dict[str, int]]):
    vocab_size: int

    migrations = (Renamed("vocabulary_size", to="vocab_size"),)

    def create(self) -> dict[str, int]:
        _COUNTER.calls += 1
        return {"vocab_size": self.vocab_size}


class _CascadeModelV0(Spec[dict[str, str]]):
    tokenizer: _CascadeTokenizerV0
    layers: int

    def create(self) -> dict[str, str]:
        _COUNTER.calls += 1
        return {"model": f"layers={self.layers}"}


class _CascadeModel(Spec[dict[str, str]]):
    tokenizer: _CascadeTokenizer  # embedded child - nothing declared here
    layers: int

    def create(self) -> dict[str, str]:
        _COUNTER.calls += 1
        return {"model": f"layers={self.layers}"}


def test_child_migration_cascades_to_parent() -> None:
    old = _CascadeModelV0(
        tokenizer=_CascadeTokenizerV0(vocabulary_size=32000), layers=12
    )
    old.create()
    old_directory = _transplant_generation(
        old,
        _CascadeModel,
        renames={
            fully_qualified_name(_CascadeTokenizerV0): fully_qualified_name(
                _CascadeTokenizer
            )
        },
    )
    _COUNTER.calls = 0

    model = _CascadeModel(tokenizer=_CascadeTokenizer(vocab_size=32000), layers=12)
    assert model.status == "done"
    assert model.create() == {"model": "layers=12"}
    assert model.load_existing() == {"model": "layers=12"}
    assert _COUNTER.calls == 0

    link = json.loads(result_link_path_in(model._base_dir).read_text())
    assert link["source"]["base_dir"] == str(old_directory / old._base_dir.name)
    assert link["migration_path"] == [
        "_CascadeTokenizer: Renamed('vocabulary_size', to='vocab_size')",
    ]

    # A different embedded spec is a different artifact: no accidental reuse.
    other = _CascadeModel(tokenizer=_CascadeTokenizer(vocab_size=999), layers=12)
    assert other.status == "missing"


class _CascadePipelineV0(Spec[str]):
    model: _CascadeModelV0
    benchmark: str

    def create(self) -> str:
        _COUNTER.calls += 1
        return "old-eval"


class _CascadePipeline(Spec[str]):
    model: _CascadeModel  # Model embeds Tokenizer embeds the chain
    benchmark: str

    def create(self) -> str:
        _COUNTER.calls += 1
        return "recomputed"


def test_cascade_resolves_transitively_through_the_dag() -> None:
    old = _CascadePipelineV0(
        model=_CascadeModelV0(
            tokenizer=_CascadeTokenizerV0(vocabulary_size=32000), layers=12
        ),
        benchmark="mmlu",
    )
    old.create()
    _transplant_generation(
        old,
        _CascadePipeline,
        renames={
            fully_qualified_name(_CascadeModelV0): fully_qualified_name(_CascadeModel),
            fully_qualified_name(_CascadeTokenizerV0): fully_qualified_name(
                _CascadeTokenizer
            ),
        },
    )
    _COUNTER.calls = 0

    pipeline = _CascadePipeline(
        model=_CascadeModel(tokenizer=_CascadeTokenizer(vocab_size=32000), layers=12),
        benchmark="mmlu",
    )
    assert pipeline.status == "done"
    assert pipeline.create() == "old-eval"
    assert _COUNTER.calls == 0

    link = json.loads(result_link_path_in(pipeline._base_dir).read_text())
    assert link["migration_path"] == [
        "_CascadeTokenizer: Renamed('vocabulary_size', to='vocab_size')",
    ]


class _CascadeTrainerV0(Spec[str]):
    tokenizer: _CascadeTokenizerV0
    learning_rate: float

    def create(self) -> str:
        _COUNTER.calls += 1
        return "old-train"


class _CascadeTrainer(Spec[str]):
    tokenizer: _CascadeTokenizer
    lr: float

    migrations = (Renamed("learning_rate", to="lr"),)

    def create(self) -> str:
        _COUNTER.calls += 1
        return "recomputed"


def test_child_and_parent_steps_compose_innermost_first() -> None:
    old = _CascadeTrainerV0(
        tokenizer=_CascadeTokenizerV0(vocabulary_size=32000), learning_rate=0.1
    )
    old.create()
    _transplant_generation(
        old,
        _CascadeTrainer,
        renames={
            fully_qualified_name(_CascadeTokenizerV0): fully_qualified_name(
                _CascadeTokenizer
            )
        },
    )
    _COUNTER.calls = 0

    trainer = _CascadeTrainer(tokenizer=_CascadeTokenizer(vocab_size=32000), lr=0.1)
    assert trainer.status == "done"
    assert trainer.create() == "old-train"
    assert _COUNTER.calls == 0

    link = json.loads(result_link_path_in(trainer._base_dir).read_text())
    assert link["migration_path"] == [
        "_CascadeTokenizer: Renamed('vocabulary_size', to='vocab_size')",
        "Renamed('learning_rate', to='lr')",
    ]


class _CascadeEnsembleV0(Spec[str]):
    tokenizers: tuple[_CascadeTokenizerV0, ...]

    def create(self) -> str:
        _COUNTER.calls += 1
        return "old-ensemble"


class _CascadeEnsemble(Spec[str]):
    tokenizers: tuple[_CascadeTokenizer, ...]

    def create(self) -> str:
        _COUNTER.calls += 1
        return "recomputed"


def test_cascade_covers_children_inside_container_fields() -> None:
    old = _CascadeEnsembleV0(
        tokenizers=(
            _CascadeTokenizerV0(vocabulary_size=100),
            _CascadeTokenizerV0(vocabulary_size=200),
        )
    )
    old.create()
    _transplant_generation(
        old,
        _CascadeEnsemble,
        renames={
            fully_qualified_name(_CascadeTokenizerV0): fully_qualified_name(
                _CascadeTokenizer
            )
        },
    )
    _COUNTER.calls = 0

    ensemble = _CascadeEnsemble(
        tokenizers=(
            _CascadeTokenizer(vocab_size=100),
            _CascadeTokenizer(vocab_size=200),
        )
    )
    assert ensemble.status == "done"
    assert ensemble.create() == "old-ensemble"
    assert _COUNTER.calls == 0


class _CascadeBreakTokenizerV0(Spec[int]):
    vocab_size: int

    def create(self) -> int:
        return 0


class _CascadeBreakTokenizer(Spec[int]):
    vocab_size: int
    normalization: str

    migrations = (Added("normalization", breaking=True),)

    def create(self) -> int:
        return 0


class _CascadeBreakModelV0(Spec[str]):
    tokenizer: _CascadeBreakTokenizerV0

    def create(self) -> str:
        _COUNTER.calls += 1
        return "old"


class _CascadeBreakModel(Spec[str]):
    tokenizer: _CascadeBreakTokenizer

    def create(self) -> str:
        _COUNTER.calls += 1
        return "recomputed"


def test_breaking_child_step_cuts_the_parent_to_recompute() -> None:
    old = _CascadeBreakModelV0(tokenizer=_CascadeBreakTokenizerV0(vocab_size=100))
    old.create()
    old_directory = _transplant_generation(
        old,
        _CascadeBreakModel,
        renames={
            fully_qualified_name(_CascadeBreakTokenizerV0): fully_qualified_name(
                _CascadeBreakTokenizer
            )
        },
    )
    _COUNTER.calls = 0

    model = _CascadeBreakModel(
        tokenizer=_CascadeBreakTokenizer(vocab_size=100, normalization="nfc")
    )
    # The cut propagates: recompute, not a Stale error.
    assert model.status == "missing"
    assert model.create() == "recomputed"
    assert _COUNTER.calls == 1
    assert not result_link_path_in(model._base_dir).exists()
    assert old_directory.exists()


class _CascadeSeedTokenizerV0(Spec[int]):
    vocab_size: int

    def create(self) -> int:
        return 0


class _CascadeSeedTokenizer(Spec[int]):
    vocab_size: int
    seed: int = 3  # new runs default to 3; old runs behaved like 1

    migrations = (Added("seed", default=1),)

    def create(self) -> int:
        return 0


class _CascadeSeedModelV0(Spec[str]):
    tokenizer: _CascadeSeedTokenizerV0

    def create(self) -> str:
        return "old"


class _CascadeSeedModel(Spec[str]):
    tokenizer: _CascadeSeedTokenizer

    def create(self) -> str:
        return "recomputed"


def test_child_added_default_pins_history_through_the_cascade() -> None:
    old = _CascadeSeedModelV0(tokenizer=_CascadeSeedTokenizerV0(vocab_size=8))
    old.create()
    _transplant_generation(
        old,
        _CascadeSeedModel,
        renames={
            fully_qualified_name(_CascadeSeedTokenizerV0): fully_qualified_name(
                _CascadeSeedTokenizer
            )
        },
    )

    pinned = _CascadeSeedModel(tokenizer=_CascadeSeedTokenizer(vocab_size=8, seed=1))
    assert pinned.status == "done"
    field_default = _CascadeSeedModel(tokenizer=_CascadeSeedTokenizer(vocab_size=8))
    assert field_default.status == "missing"


class _CascadeRelocatedTokenizer(Spec[int]):
    vocab_size: int

    def create(self) -> int:
        return 0


class _CascadeMovedTokenizer(Spec[int]):
    vocab_size: int

    migrations = (MovedFrom(fully_qualified_name(_CascadeRelocatedTokenizer)),)

    def create(self) -> int:
        return 0


class _CascadeMovedModelV0(Spec[str]):
    tokenizer: _CascadeRelocatedTokenizer

    def create(self) -> str:
        _COUNTER.calls += 1
        return "old"


class _CascadeMovedModel(Spec[str]):
    tokenizer: _CascadeMovedTokenizer

    def create(self) -> str:
        _COUNTER.calls += 1
        return "recomputed"


def test_child_movedfrom_rewrites_the_embedded_class_marker() -> None:
    old = _CascadeMovedModelV0(tokenizer=_CascadeRelocatedTokenizer(vocab_size=100))
    old.create()
    # The old class name is genuinely recorded in the snapshot: no rename.
    _transplant_generation(old, _CascadeMovedModel)
    _COUNTER.calls = 0

    model = _CascadeMovedModel(tokenizer=_CascadeMovedTokenizer(vocab_size=100))
    assert model.status == "done"
    assert model.create() == "old"
    assert _COUNTER.calls == 0

    link = json.loads(result_link_path_in(model._base_dir).read_text())
    assert link["migration_path"] == [
        "_CascadeMovedTokenizer: "
        f"MovedFrom({fully_qualified_name(_CascadeRelocatedTokenizer)!r})",
    ]


@dataclass(frozen=True)
class _CascadeOptimizerV0:
    learning_rate: float


@dataclass(frozen=True)
class _CascadeOptimizer:
    lr: float

    migrations: ClassVar[tuple[furu.MigrationStep, ...]] = (
        Renamed("learning_rate", to="lr"),
    )


class _CascadeOptRunV0(Spec[str]):
    optimizer: _CascadeOptimizerV0
    epochs: int

    def create(self) -> str:
        _COUNTER.calls += 1
        return "old"


class _CascadeOptRun(Spec[str]):
    optimizer: _CascadeOptimizer  # cascades exactly like a Spec field
    epochs: int

    def create(self) -> str:
        _COUNTER.calls += 1
        return "recomputed"


def test_plain_dataclass_child_cascades_like_a_spec() -> None:
    old = _CascadeOptRunV0(optimizer=_CascadeOptimizerV0(learning_rate=0.1), epochs=3)
    old.create()
    _transplant_generation(
        old,
        _CascadeOptRun,
        renames={
            fully_qualified_name(_CascadeOptimizerV0): fully_qualified_name(
                _CascadeOptimizer
            )
        },
    )
    _COUNTER.calls = 0

    run = _CascadeOptRun(optimizer=_CascadeOptimizer(lr=0.1), epochs=3)
    assert run.status == "done"
    assert run.create() == "old"
    assert _COUNTER.calls == 0

    link = json.loads(result_link_path_in(run._base_dir).read_text())
    assert link["migration_path"] == [
        "_CascadeOptimizer: Renamed('learning_rate', to='lr')",
    ]


class _CascadeSilentTokenizerV0(Spec[int]):
    vocabulary_size: int

    def create(self) -> int:
        return 0


class _CascadeSilentTokenizer(Spec[int]):
    vocab_size: int  # renamed without declaring a chain

    def create(self) -> int:
        return 0


class _CascadeSilentModelV0(Spec[str]):
    tokenizer: _CascadeSilentTokenizerV0

    def create(self) -> str:
        return "old"


class _CascadeSilentModel(Spec[str]):
    tokenizer: _CascadeSilentTokenizer

    def create(self) -> str:
        return "recomputed"


def test_stale_report_attributes_the_diff_to_the_embedded_class() -> None:
    old = _CascadeSilentModelV0(tokenizer=_CascadeSilentTokenizerV0(vocabulary_size=8))
    old.create()
    _transplant_generation(
        old,
        _CascadeSilentModel,
        renames={
            fully_qualified_name(_CascadeSilentTokenizerV0): fully_qualified_name(
                _CascadeSilentTokenizer
            )
        },
    )

    model = _CascadeSilentModel(tokenizer=_CascadeSilentTokenizer(vocab_size=8))
    assert model.status == "stale"
    with pytest.raises(Stale) as excinfo:
        model.create()
    message = str(excinfo.value)
    assert "- tokenizer.vocabulary_size" in message
    assert "+ tokenizer.vocab_size" in message
    assert "inside embedded _CascadeSilentTokenizer" in message
    assert "_CascadeSilentTokenizer.migrations" in message


class _CascadeAmbiguousTokenizer(Spec[int]):
    n: int

    migrations = (
        MovedFrom(fully_qualified_name(_LegacyRun)),
        MovedFrom(fully_qualified_name(_LegacyRun)),
    )

    def create(self) -> int:
        return 0


class _CascadeAmbiguousModel(Spec[int]):
    tokenizer: _CascadeAmbiguousTokenizer

    def create(self) -> int:
        return 0


def test_ambiguous_child_chain_is_rejected_at_parent_resolution() -> None:
    with pytest.raises(MigrationError, match="ambiguous"):
        _ = _CascadeAmbiguousModel(tokenizer=_CascadeAmbiguousTokenizer(n=1)).status


@dataclass(frozen=True)
class _CascadeBrokenOptimizer:
    lr: float

    migrations: ClassVar[tuple[furu.MigrationStep, ...]] = (
        Renamed("learning_rate", to="lrx"),
    )


def test_embedded_dataclass_chain_is_validated_when_a_spec_embeds_it() -> None:
    with pytest.raises(TypeError, match="'lrx' is not a field"):

        class _BrokenOptimizerRun(Spec[int]):
            optimizer: _CascadeBrokenOptimizer

            def create(self) -> int:
                return 0
