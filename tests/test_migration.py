from __future__ import annotations

import errno
import json
import shutil
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, cast

import pytest
from layout import result_manifest_path_in, version_dir_in

import furu
import furu.migration.links as migration_links
import furu.migration.resolution as migration_resolution
from furu import Added, MovedFrom, Renamed, Retyped, Rewrite, Spec, Stale
from furu.constants import FIELDSMARKER
from furu.migration.steps import MigrationError
from furu.result.bundle import WRAPPER_KEY
from furu.result.codec import Codec
from furu.storage._layout import (
    compute_lock_path_in,
    data_dir_in,
    result_link_path_in,
)
from furu.utils import JsonFields, JsonValue, fully_qualified_name


class _Counter:
    def __init__(self) -> None:
        self.calls = 0


_COUNTER = _Counter()


@pytest.fixture(autouse=True)
def _reset_counter() -> None:
    _COUNTER.calls = 0


def _transplant_generation(
    donor: Spec,
    target_cls: type[Spec],
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


_NAN = float("nan")


class _NaNDefaultTrainRun(Spec[dict[str, str]]):
    dataset: str
    lr: float
    score: float = _NAN

    migrations = (
        MovedFrom(fully_qualified_name(_OldTrainRun)),
        Renamed("learning_rate", to="lr"),
        Added("score", default=_NAN),
    )

    def create(self) -> dict[str, str]:
        return {"dataset": self.dataset, "lr": str(self.lr)}


def test_rename_plus_add_reuses_old_result_through_result_link() -> None:
    old = _OldTrainRun(learning_rate=0.001, dataset="cifar10")
    assert old.create() == {"dataset": "cifar10", "learning_rate": "0.001"}
    _COUNTER.calls = 0

    new = _TrainRun(dataset="cifar10", lr=0.001)
    assert new.create() == {"dataset": "cifar10", "learning_rate": "0.001"}
    assert _COUNTER.calls == 0

    link_path = result_link_path_in(new._base_dir)
    link = json.loads(link_path.read_text())
    assert link["source"]["version_dir"] == str(version_dir_in(old._base_dir))
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

    assert migration_links.version_for_loading(target) == (
        version_dir_in(source._base_dir)
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

    assert migration_links.version_for_loading(target) == (
        version_dir_in(source._base_dir)
    )


def test_result_link_read_gracefully_reresolves_estale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _OldTrainRun(learning_rate=0.001, dataset="cifar10")
    source.create()
    target = _TrainRun(dataset="cifar10", lr=0.001)
    assert migration_links.version_for_loading(target) is not None
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

    assert migration_links.version_for_loading(target) == (
        version_dir_in(source._base_dir)
    )


class _PathValue:
    def __init__(self, path: Path) -> None:
        self.path = path


class _PathValueCodec(Codec[_PathValue]):
    @classmethod
    def matches(cls, value: object) -> bool:
        return isinstance(value, _PathValue)

    @classmethod
    def save(cls, value: _PathValue, artifact_directory: Path) -> Mapping[str, object]:
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

    source_payload = data_dir_in(version_dir_in(source._base_dir)) / "payload.txt"
    assert migrated.create().path == source_payload.resolve()
    assert migrated.load_existing().path == source_payload.resolve()


def test_added_field_binds_only_the_default_value() -> None:
    _OldTrainRun(learning_rate=0.001, dataset="cifar10").create()

    assert _TrainRun(dataset="cifar10", lr=0.001).status == "done"
    # Old results correspond to the migration's pinned default; any other value
    # is a different spec whose result genuinely never existed.
    assert _TrainRun(dataset="cifar10", lr=0.001, seed=7).status == "missing"


def test_added_nan_default_matches_by_serialized_value() -> None:
    _OldTrainRun(learning_rate=0.001, dataset="cifar10").create()

    assert _NaNDefaultTrainRun(dataset="cifar10", lr=0.001).status == "done"


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
    assert result_manifest_path_in(version_dir_in(new._base_dir)).exists()
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
    """The whole changelog lives on the current class: a run only reuses a
    result whose generation its own chain reaches."""

    dataset: str
    lr: float
    seed: int = 0

    migrations = (
        MovedFrom(fully_qualified_name(_OldTrainRun)),
        Renamed("learning_rate", to="lr"),
        MovedFrom(fully_qualified_name(_MidRun)),
        Added("seed", default=0),
    )

    def create(self) -> dict[str, str]:
        _COUNTER.calls += 1
        return {"dataset": self.dataset, "lr": str(self.lr), "seed": str(self.seed)}


class _FinalRunViaMid(Spec[dict[str, str]]):
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


def test_full_changelog_links_directly_at_the_ultimate_source() -> None:
    old = _OldTrainRun(learning_rate=0.001, dataset="cifar10")
    old.create()
    _COUNTER.calls = 0

    mid = _MidRun(dataset="cifar10", lr=0.001)
    assert mid.create() == {"dataset": "cifar10", "learning_rate": "0.001"}

    final = _FinalRun(dataset="cifar10", lr=0.001)
    assert final.create() == {"dataset": "cifar10", "learning_rate": "0.001"}
    assert _COUNTER.calls == 0

    link = json.loads(result_link_path_in(final._base_dir).read_text())
    assert link["source"]["version_dir"] == str(version_dir_in(old._base_dir))
    assert link["source"]["fully_qualified_name"] == old._fully_qualified_name
    assert link["migration_path"] == [
        f"MovedFrom({old._fully_qualified_name!r})",
        "Renamed('learning_rate', to='lr')",
        f"MovedFrom({mid._fully_qualified_name!r})",
        "Added('seed', default=0)",
    ]


def test_a_directory_holding_only_a_link_is_not_a_source() -> None:
    # _MidRun's directory links at _OldTrainRun's result; a chain that reaches
    # _MidRun but not _OldTrainRun cannot explain that result, so it recomputes.
    old = _OldTrainRun(learning_rate=0.001, dataset="cifar10")
    old.create()
    _MidRun(dataset="cifar10", lr=0.001).create()
    _COUNTER.calls = 0

    final = _FinalRunViaMid(dataset="cifar10", lr=0.001)
    assert final.status == "missing"
    assert final.create() == {"dataset": "cifar10", "lr": "0.001", "seed": "0"}
    assert _COUNTER.calls == 1
    assert not result_link_path_in(final._base_dir).exists()


def test_link_to_a_source_the_chain_no_longer_covers_is_no_result() -> None:
    legacy = _LegacyRun(dataset="cifar10", learning_rate=0.001)
    legacy.create()
    target = _TrainRun(dataset="cifar10", lr=0.001)
    target._base_dir.mkdir(parents=True)
    result_link_path_in(target._base_dir).write_text(
        migration_links._ResultLink(
            current=migration_links._ResultLinkCurrent(
                fully_qualified_name=target._fully_qualified_name,
                schema_hash=target._artifact_schema_hash,
                artifact_hash=target._artifact_hash,
                fields=cast(JsonFields, target._artifact_data[FIELDSMARKER]),
            ),
            source=migration_links._ResultLinkSource(
                fully_qualified_name=legacy._fully_qualified_name,
                schema_hash=legacy._artifact_schema_hash,
                artifact_hash=legacy._artifact_hash,
                version_dir=version_dir_in(legacy._base_dir),
            ),
            migration_path=(),
        ).model_dump_json()
    )
    _COUNTER.calls = 0

    assert target.status == "missing"
    assert target.create() == {"dataset": "cifar10", "lr": "0.001", "seed": "0"}
    assert _COUNTER.calls == 1


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

    migrations = (Retyped("optimizer", was=_SGD),)

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


# --- the recorded steps pick the chain, never a search ------------------------------


def test_a_repeated_step_is_rejected_at_class_creation() -> None:
    with pytest.raises(TypeError, match=r"migrations\[1\].*repeats migrations\[0\]"):

        class _RepeatedMove(Spec[int]):
            n: int

            migrations = (
                MovedFrom(fully_qualified_name(_LegacyRun)),
                MovedFrom(fully_qualified_name(_LegacyRun)),
            )

            def create(self) -> int:
                return self.n


def _identity_rewrite(fields: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return dict(fields)


class _RewriteThenRetype(Spec[str]):
    optimizer: _SGD | _AdamW
    lr: float

    migrations = (
        Rewrite(_identity_rewrite),
        Retyped("optimizer", was=_SGD),
    )

    def create(self) -> str:
        return "current"


def test_recorded_steps_pick_one_chain_where_shapes_alone_cannot() -> None:
    """The Rewrite's wildcard also fits gen0's shape after Retyped alone; the
    steps gen0 recorded (none) say both still apply."""
    gen0 = _WidenGen0(optimizer=_SGD(momentum=0.9), lr=0.1)
    gen0.create()
    _transplant_generation(gen0, _RewriteThenRetype)

    spec = _RewriteThenRetype(optimizer=_SGD(momentum=0.9), lr=0.1)
    assert spec.status == "done"
    assert spec.load_existing() == "gen0"
    link = json.loads(result_link_path_in(spec._base_dir).read_text())
    assert link["migration_path"] == [
        "Rewrite(_identity_rewrite)",
        f"Retyped('optimizer', was={fully_qualified_name(_SGD)})",
    ]


class _RewriteThenChildAddedV0(Spec[str]):
    tokenizer: _CascadeTokenizerV0
    lr: float

    def create(self) -> str:
        return "old"


class _RewriteThenChildAdded(Spec[str]):
    tokenizer: _CascadeTokenizer
    lr: float

    migrations = (Rewrite(_identity_rewrite),)

    def create(self) -> str:
        _COUNTER.calls += 1
        return "recomputed"


def test_child_migration_is_not_ambiguous_with_a_trailing_rewrite() -> None:
    """After the child normalises, the snapshot equals the current schema; the Rewrite's wildcard must not also claim it."""
    old = _RewriteThenChildAddedV0(
        tokenizer=_CascadeTokenizerV0(vocabulary_size=32000), lr=0.1
    )
    old.create()
    _transplant_generation(
        old,
        _RewriteThenChildAdded,
        renames={
            fully_qualified_name(_CascadeTokenizerV0): fully_qualified_name(
                _CascadeTokenizer
            )
        },
    )
    _COUNTER.calls = 0

    current = _RewriteThenChildAdded(
        tokenizer=_CascadeTokenizer(vocab_size=32000), lr=0.1
    )
    assert current.status == "done"
    assert current.load_existing() == "old"
    assert _COUNTER.calls == 0


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


def test_deleted_match_falls_back_to_the_next_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Both donors rewrite to version=2, so either can serve the target.
    first = _RewriteDonor(dataset="cifar10", version=2.1)
    second = _RewriteDonor(dataset="cifar10", version=2.9)
    first.create()
    second.create()
    schema_directory = _transplant_generation(first, _RewrittenRun)
    reads = _count_reads_under(monkeypatch, schema_directory)
    target = _RewrittenRun(dataset="cifar10", version=2)

    assert target.status == "done"
    assert target.status == "done"
    assert len(reads) == 2

    shutil.rmtree(
        next(path for path in sorted(schema_directory.iterdir()) if path.is_dir())
    )
    assert target.status == "done"
    assert len(reads) == 2


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

    migrations = (Added("seed", breaking=True),)

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


# --- Added default_factory computes the backfill from each old run's fields --------


@dataclass(frozen=True)
class _ModelConfig:
    width: int
    depth: int = 2


class _FlatWidthDonor(Spec[str]):
    dataset: str
    width: int

    def create(self) -> str:
        _COUNTER.calls += 1
        return f"old-run-{self.width}"


def _model_from_width(fields: Mapping[str, JsonValue]) -> _ModelConfig:
    width = fields["width"]
    assert isinstance(width, int)
    return _ModelConfig(width=width)


class _NestedModelRun(Spec[str]):
    dataset: str
    width: int
    model: _ModelConfig = _ModelConfig(width=8)

    migrations = (Added("model", default_factory=_model_from_width),)

    def create(self) -> str:
        _COUNTER.calls += 1
        return "recomputed"


def test_added_default_factory_backfills_an_object_from_the_old_runs_fields() -> None:
    for width in (4, 16):
        _FlatWidthDonor(dataset="cifar10", width=width).create()
    _transplant_generation(_FlatWidthDonor(dataset="cifar10", width=4), _NestedModelRun)
    _COUNTER.calls = 0

    matching = _NestedModelRun(dataset="cifar10", width=4, model=_ModelConfig(width=4))
    assert matching.status == "done"
    assert matching.create() == "old-run-4"
    assert (
        _NestedModelRun(
            dataset="cifar10", width=16, model=_ModelConfig(width=16)
        ).status
        == "done"
    )
    assert (
        _NestedModelRun(dataset="cifar10", width=4, model=_ModelConfig(width=8)).status
        == "missing"
    )
    assert _COUNTER.calls == 0
    assert json.loads(result_link_path_in(matching._base_dir).read_text())[
        "migration_path"
    ] == ["Added('model', default_factory=_model_from_width)"]


def _model_from_missing_depth(fields: Mapping[str, JsonValue]) -> _ModelConfig:
    return _ModelConfig(width=cast(int, fields["depth"]))


class _BadDeriveRun(Spec[str]):
    dataset: str
    width: int
    model: _ModelConfig = _ModelConfig(width=8)

    migrations = (Added("model", default_factory=_model_from_missing_depth),)

    def create(self) -> str:
        return "recomputed"


def test_added_default_factory_sees_only_the_fields_present_at_that_step() -> None:
    donor = _FlatWidthDonor(dataset="cifar10", width=4)
    donor.create()
    _transplant_generation(donor, _BadDeriveRun)

    with pytest.raises(KeyError, match="'depth' is not a source field"):
        _ = _BadDeriveRun(
            dataset="cifar10", width=4, model=_ModelConfig(width=4)
        ).status


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


def test_added_with_default_and_factory_fails_at_class_creation() -> None:
    with pytest.raises(TypeError, match="exactly one of default="):

        class _AddedBoth(Spec[int]):
            seed: int

            migrations = (Added("seed", default=0, default_factory=lambda fields: 0),)

            def create(self) -> int:
                return 0


def test_added_without_default_fails_at_class_creation() -> None:
    with pytest.raises(TypeError, match="exactly one of default="):

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

            migrations = [Renamed("m", to="n")]  # noqa: RUF012  # ty: ignore[invalid-assignment]

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

    def counting(obj: Spec) -> migration_resolution._ClassResolution:
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


def _count_reads_under(
    monkeypatch: pytest.MonkeyPatch, schema_directory: Path
) -> list[Path]:
    reads: list[Path] = []
    real_spec_path_in = migration_links.spec_path_in

    def counting(artifact_dir: Path) -> Path:
        if artifact_dir.parent == schema_directory:
            reads.append(artifact_dir)
        return real_spec_path_in(artifact_dir)

    monkeypatch.setattr(migration_links, "spec_path_in", counting)
    return reads


def test_old_results_are_read_once_per_class_per_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old = _OldTrainRun(learning_rate=0.001, dataset="cifar10")
    old.create()
    _OldTrainRun(learning_rate=0.01, dataset="mnist").create()
    reads = _count_reads_under(monkeypatch, old._base_dir.parent)

    assert _TrainRun(dataset="cifar10", lr=0.001).status == "done"
    assert _TrainRun(dataset="mnist", lr=0.01).status == "done"
    assert _TrainRun(dataset="mnist", lr=0.5).status == "missing"
    assert len(reads) == 2

    # A deleted source falls through even though the index still lists it.
    shutil.rmtree(old._base_dir)
    assert _TrainRun(dataset="cifar10", lr=0.001).status == "missing"
    assert len(reads) == 2


def test_pinned_added_default_skips_the_old_generation_entirely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old = _OldTrainRun(learning_rate=0.001, dataset="cifar10")
    old.create()
    reads = _count_reads_under(monkeypatch, old._base_dir.parent)

    # Every migrated result carries seed=0, so seed=7 has no source to look for.
    assert _TrainRun(dataset="cifar10", lr=0.001, seed=7).status == "missing"
    assert reads == []


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
    assert link["source"]["version_dir"] == str(
        version_dir_in(old_directory / old._base_dir.name)
    )
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
        (
            "_CascadeMovedTokenizer: "
            f"MovedFrom({fully_qualified_name(_CascadeRelocatedTokenizer)!r})"
        ),
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


class _SelfNestedRunV0(Spec[str]):
    width: int
    init: _SelfNestedRunV0 | None = None

    def create(self) -> str:
        _COUNTER.calls += 1
        return f"width={self.width}"


class _SelfNestedRun(Spec[str]):
    width: int
    init: _SelfNestedRun | None = None
    depth: int = 1

    migrations = (Added("depth", default=1),)

    def create(self) -> str:
        _COUNTER.calls += 1
        return "recomputed"


def test_own_chain_migrates_nested_instances_of_the_same_class() -> None:
    old = _SelfNestedRunV0(
        width=2, init=_SelfNestedRunV0(width=1, init=_SelfNestedRunV0(width=0))
    )
    old.create()
    _transplant_generation(old, _SelfNestedRun)
    _COUNTER.calls = 0

    run = _SelfNestedRun(
        width=2, init=_SelfNestedRun(width=1, init=_SelfNestedRun(width=0))
    )
    assert run.status == "done"
    assert run.create() == "width=2"
    assert _COUNTER.calls == 0


# --- result_rewrite: old results come back in today's shape -------------------------


class _PairDonor(Spec[tuple[str, float]]):
    dataset: str

    def create(self) -> tuple[str, float]:
        _COUNTER.calls += 1
        return (self.dataset, 0.5)


def _widen_to_triple(result: JsonValue) -> JsonValue:
    body = cast(dict[str, JsonValue], result)[WRAPPER_KEY]
    cast(list[JsonValue], cast(dict[str, JsonValue], body)["items"]).append(None)
    return result


class _TripleRun(Spec[tuple[str, float, float | None]]):
    dataset: str
    eval: bool = False

    migrations = (
        MovedFrom(fully_qualified_name(_PairDonor)),
        Added("eval", default=False, result_rewrite=_widen_to_triple),
    )

    def create(self) -> tuple[str, float, float | None]:
        _COUNTER.calls += 1
        return (self.dataset, 0.5, 0.9 if self.eval else None)


def test_result_rewrite_reshapes_migrated_results_on_every_load() -> None:
    donor = _PairDonor(dataset="cifar10")
    assert donor.create() == ("cifar10", 0.5)
    _COUNTER.calls = 0

    spec = _TripleRun(dataset="cifar10")
    assert spec.status == "done"
    assert spec.create() == ("cifar10", 0.5, None)
    assert spec.load_existing() == ("cifar10", 0.5, None)
    assert furu.load_existing([spec]) == [("cifar10", 0.5, None)]
    assert _TripleRun(dataset="cifar10").create() == ("cifar10", 0.5, None)
    assert _COUNTER.calls == 0

    link = json.loads(result_link_path_in(spec._base_dir).read_text())
    assert link["migration_path"] == [
        f"MovedFrom({donor._fully_qualified_name!r})",
        "Added('eval', default=False, result_rewrite=_widen_to_triple)",
    ]

    # The pinned default keeps eval=True runs away from the old results.
    assert _TripleRun(dataset="cifar10", eval=True).status == "missing"
    # A result computed today is not rewritten.
    assert _TripleRun(dataset="mnist").create() == ("mnist", 0.5, None)
    assert _TripleRun(dataset="mnist").load_existing() == ("mnist", 0.5, None)


@dataclass(frozen=True)
class _MetricsV0:
    loss: float


@dataclass(frozen=True)
class _MetricsV1:
    loss: float
    val_loss: float | None


@dataclass(frozen=True)
class _Metrics:
    loss: float
    val_loss: float | None
    checkpoint: str | None


class _MetricsRunV0(Spec[_MetricsV0]):
    dataset: str

    def create(self) -> _MetricsV0:
        return _MetricsV0(loss=0.5)


class _MetricsRunV1(Spec[_MetricsV1]):
    dataset: str
    evaluate: bool = False

    migrations = (Added("evaluate", default=False),)

    def create(self) -> _MetricsV1:
        return _MetricsV1(loss=0.25, val_loss=0.3 if self.evaluate else None)


def _with_field(name: str) -> Callable[[JsonValue], JsonValue]:
    def add(result: JsonValue) -> JsonValue:
        body = cast(dict[str, JsonValue], result)[WRAPPER_KEY]
        cast(dict[str, JsonValue], cast(dict[str, JsonValue], body)[FIELDSMARKER])[
            name
        ] = None
        return result

    return add


class _MetricsRun(Spec[_Metrics]):
    dataset: str
    evaluate: bool = False
    save_checkpoint: bool = False

    migrations = (
        Added("evaluate", default=False, result_rewrite=_with_field("val_loss")),
        Added(
            "save_checkpoint", default=False, result_rewrite=_with_field("checkpoint")
        ),
    )

    def create(self) -> _Metrics:
        _COUNTER.calls += 1
        return _Metrics(loss=0.0, val_loss=None, checkpoint=None)


def test_result_rewrites_grow_a_dataclass_field_by_field_before_it_is_built() -> None:
    # Both donors stored the *current* dataclass name with fewer fields; only
    # the rewrites from each generation onward make it loadable.
    renames = {
        fully_qualified_name(_MetricsV0): fully_qualified_name(_Metrics),
        fully_qualified_name(_MetricsV1): fully_qualified_name(_Metrics),
    }
    v0 = _MetricsRunV0(dataset="cifar10")
    v0.create()
    _transplant_generation(v0, _MetricsRun, renames=renames)
    v1 = _MetricsRunV1(dataset="mnist")
    v1.create()
    _transplant_generation(v1, _MetricsRun, renames=renames)
    _COUNTER.calls = 0

    assert _MetricsRun(dataset="cifar10").create() == _Metrics(
        loss=0.5, val_loss=None, checkpoint=None
    )
    assert _MetricsRun(dataset="mnist").create() == _Metrics(
        loss=0.25, val_loss=None, checkpoint=None
    )
    assert _COUNTER.calls == 0


def test_result_rewrite_on_a_breaking_step_fails_at_class_creation() -> None:
    with pytest.raises(TypeError, match="breaking step discards old results"):

        class _Bad(Spec[int]):
            seed: int

            migrations = (
                Added("seed", breaking=True, result_rewrite=lambda result: result),
            )

            def create(self) -> int:
                return 0


@dataclass(frozen=True)
class _KnobsWithResultRewrite:
    width: int
    depth: int = 1

    migrations: ClassVar[tuple[Added, ...]] = (
        Added("depth", default=1, result_rewrite=lambda result: result),
    )


def test_result_rewrite_on_an_embedded_dataclass_fails_at_class_creation() -> None:
    with pytest.raises(TypeError, match="belongs on a Spec"):

        class _UsesKnobs(Spec[int]):
            knobs: _KnobsWithResultRewrite

            def create(self) -> int:
                return 0


# --- parallel branches: worktrees share one store --------------------------------


def _fresh_process() -> None:
    """Forget what this process resolved, as a new worker would."""
    migration_resolution._RESOLUTION_CACHE.clear()
    migration_links._SOURCES_CACHE.clear()


class _TrunkRun(Spec[str]):
    a: int

    def create(self) -> str:
        return f"trunk a={self.a}"


class _WarmupBranchRun(Spec[str]):
    a: int
    warmup: int = 0

    migrations = (Added("warmup", default=0),)

    def create(self) -> str:
        return f"warmup a={self.a}"


class _DropoutBranchRun(Spec[str]):
    a: int
    dropout: float = 0.0

    migrations = (Added("dropout", default=0.0),)

    def create(self) -> str:
        return f"dropout a={self.a}"


def _grow_branches() -> tuple[_TrunkRun, _WarmupBranchRun, _DropoutBranchRun]:
    """A trunk result, then one result on each of two branches grown from it."""
    trunk = _TrunkRun(a=1)
    trunk.create()
    _transplant_generation(trunk, _WarmupBranchRun)
    warmup = _WarmupBranchRun(a=2)
    warmup.create()
    _transplant_generation(trunk, _DropoutBranchRun)
    dropout = _DropoutBranchRun(a=3)
    dropout.create()
    return trunk, warmup, dropout


def test_sibling_branch_schemas_are_not_stale() -> None:
    _, warmup, dropout = _grow_branches()
    _transplant_generation(warmup, _DropoutBranchRun)
    _transplant_generation(warmup, _TrunkRun)
    _transplant_generation(dropout, _TrunkRun)
    _fresh_process()

    assert _DropoutBranchRun(a=4).status == "missing"
    assert _DropoutBranchRun(a=1).status == "done"
    assert _DropoutBranchRun(a=1).load_existing() == "trunk a=1"
    assert _TrunkRun(a=4).status == "missing"


class _WarmupRetypedRun(Spec[str]):
    a: int
    warmup: float = 0.0

    migrations = (Added("warmup", default=0.0),)

    def create(self) -> str:
        return f"a={self.a}"


def test_undeclared_retype_of_an_added_field_stays_stale() -> None:
    _, warmup, _ = _grow_branches()
    _transplant_generation(warmup, _WarmupRetypedRun)

    assert _WarmupRetypedRun(a=4).status == "stale"


# --- merged branches: the recorded steps say what each schema still needs ----------


class _MergedRun(Spec[str]):
    a: int
    warmup: int = 0
    dropout: float = 0.0
    seed: int = 0

    migrations = (
        Added("warmup", default=0),
        Added("dropout", default=0.0),
        Added("seed", default=0),
    )

    def create(self) -> str:
        return f"merged a={self.a}"


class _MergedReversedRun(Spec[str]):
    a: int
    warmup: int = 0
    dropout: float = 0.0

    migrations = (Added("dropout", default=0.0), Added("warmup", default=0))

    def create(self) -> str:
        return f"merged a={self.a}"


@pytest.mark.parametrize("merged", [_MergedRun, _MergedReversedRun])
def test_merged_branches_reuse_every_branch_in_any_order(
    merged: type[_MergedRun | _MergedReversedRun],
) -> None:
    for donor in _grow_branches():
        _transplant_generation(donor, merged)

    assert [merged(a=a).load_existing() for a in (1, 2, 3)] == [
        "trunk a=1",
        "warmup a=2",
        "dropout a=3",
    ]
    assert merged(a=4).status == "missing"


def test_a_merged_schema_gets_only_the_steps_it_lacks() -> None:
    for donor in _grow_branches():
        _transplant_generation(donor, _MergedRun)

    # The dropout branch lacks warmup, declared before the dropout step it has.
    assert _MergedRun(a=3).load_existing() == "dropout a=3"
    link = json.loads(result_link_path_in(_MergedRun(a=3)._base_dir).read_text())
    assert link["migration_path"] == [
        "Added('warmup', default=0)",
        "Added('seed', default=0)",
    ]


class _LrTrunk(Spec[str]):
    a: int
    lr: int = 1

    def create(self) -> str:
        return f"trunk a={self.a}"


class _LrRenamedBranch(Spec[str]):
    a: int
    learning_rate: int = 1

    migrations = (Renamed("lr", to="learning_rate"),)

    def create(self) -> str:
        return f"renamed a={self.a}"


class _LrRetypedBranch(Spec[str]):
    a: int
    lr: int | None = 1

    migrations = (Retyped("lr", was=int),)

    def create(self) -> str:
        return f"retyped a={self.a}"


class _LrMerged(Spec[str]):
    a: int
    learning_rate: int | None = 1

    migrations = (Retyped("lr", was=int), Renamed("lr", to="learning_rate"))

    def create(self) -> str:
        return f"merged a={self.a}"


class _LrMergedKeepingRenamed(Spec[str]):
    a: int
    learning_rate: int | None = 1

    migrations = (
        Renamed("lr", to="learning_rate"),
        Retyped("learning_rate", was=int),
    )

    def create(self) -> str:
        return f"merged a={self.a}"


class _LrMergedBreaking(Spec[str]):
    a: int
    learning_rate: int | None = 1

    migrations = (
        Retyped("lr", was=int, breaking=True),
        Renamed("lr", to="learning_rate"),
    )

    def create(self) -> str:
        return f"merged a={self.a}"


def _grow_lr_branches(merged: type[Spec]) -> None:
    """Trunk a=1; the retyped branch ran a=2 and the renamed branch a=3."""
    trunk = _LrTrunk(a=1)
    trunk.create()
    _transplant_generation(trunk, merged)
    for branch, a in ((_LrRetypedBranch, 2), (_LrRenamedBranch, 3)):
        _transplant_generation(trunk, branch)
        run = branch(a=a)
        run.create()
        _transplant_generation(run, merged)


def test_branches_changing_the_same_field_block_with_both_steps_named() -> None:
    _grow_lr_branches(_LrMerged)

    assert _LrMerged(a=1).load_existing() == "trunk a=1"
    assert _LrMerged(a=2).load_existing() == "retyped a=2"
    # The renamed branch's schema already has the rename, so the retype would
    # land after a rename of the same field.
    blocked = _LrMerged(a=3)
    assert blocked.status == "stale"
    with pytest.raises(Stale) as excinfo:
        blocked.create()
    message = str(excinfo.value)
    assert (
        "migrations[0] (Retyped('lr', was=builtins.int)) would have to run after "
        "migrations[1] (Renamed('lr', to='learning_rate')), which this schema "
        "already has"
    ) in message
    assert "two branches changed the same thing in parallel" in message


def test_ordering_steps_like_one_branch_keeps_its_results() -> None:
    _grow_lr_branches(_LrMergedKeepingRenamed)

    assert _LrMergedKeepingRenamed(a=3).load_existing() == "renamed a=3"
    assert _LrMergedKeepingRenamed(a=1).load_existing() == "trunk a=1"
    # The retyped branch recorded Retyped('lr'), which no longer exists: its
    # results are recomputed, as a sibling branch's would be.
    assert _LrMergedKeepingRenamed(a=2).status == "missing"


def test_a_breaking_step_resolves_a_clash_by_recomputing() -> None:
    _grow_lr_branches(_LrMergedBreaking)

    assert _LrMergedBreaking(a=2).load_existing() == "retyped a=2"
    assert _LrMergedBreaking(a=3).status == "missing"
    assert _LrMergedBreaking(a=1).status == "missing"


class _WidthTrunk(Spec[str]):
    a: int
    width: int = 3

    def create(self) -> str:
        return f"trunk a={self.a}"


class _SizeBranch(Spec[str]):
    a: int
    size: int = 3

    migrations = (Renamed("width", to="size"),)

    def create(self) -> str:
        return f"size a={self.a}"


class _DepthMerged(Spec[str]):
    a: int
    size: int = 3
    depth: int = 6

    migrations = (
        Added("depth", default_factory=lambda fields: cast(int, fields["width"]) * 2),
        Renamed("width", to="size"),
    )

    def create(self) -> str:
        return f"merged a={self.a}"


def test_a_default_factory_never_runs_after_a_step_it_was_declared_before() -> None:
    trunk = _WidthTrunk(a=1)
    trunk.create()
    _transplant_generation(trunk, _DepthMerged)
    _transplant_generation(trunk, _SizeBranch)
    size = _SizeBranch(a=2, size=4)
    size.create()
    _transplant_generation(size, _DepthMerged)

    assert _DepthMerged(a=1).load_existing() == "trunk a=1"
    # The factory reads width, which the size branch already renamed.
    assert _DepthMerged(a=2, size=4, depth=8).status == "stale"


class _MovedFromWarmupDonor(Spec[dict[str, str]]):
    dataset: str
    learning_rate: float
    warmup: int = 0

    migrations = (Added("warmup", default=0),)

    def create(self) -> dict[str, str]:
        return {"branch": "warmup"}


class _MovedDonor(Spec[dict[str, str]]):
    dataset: str
    learning_rate: float

    migrations = (MovedFrom(fully_qualified_name(_LegacyRun)),)

    def create(self) -> dict[str, str]:
        return {"branch": "moved"}


class _MovedMergedRun(Spec[dict[str, str]]):
    dataset: str
    learning_rate: float
    warmup: int = 0

    migrations = (
        MovedFrom(fully_qualified_name(_LegacyRun)),
        Added("warmup", default=0),
    )

    def create(self) -> dict[str, str]:
        return {"branch": "merged"}


def test_a_move_merges_with_a_field_added_under_the_old_name() -> None:
    warmup = _MovedFromWarmupDonor(dataset="w", learning_rate=0.1)
    warmup.create()
    _transplant_generation(warmup, _LegacyRun)
    moved = _MovedDonor(dataset="m", learning_rate=0.1)
    moved.create()
    _transplant_generation(moved, _MovedMergedRun)

    assert _MovedMergedRun(dataset="w", learning_rate=0.1).load_existing() == {
        "branch": "warmup"
    }
    assert _MovedMergedRun(dataset="m", learning_rate=0.1).load_existing() == {
        "branch": "moved"
    }


# --- step identities: functions and was= types are left out ----------------------


class _EpochsRun(Spec[str]):
    epochs: int

    def create(self) -> str:
        return f"epochs={self.epochs}"


def _epochs_to_steps(fields: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    return {**fields, "steps": cast(int, fields["steps"]) * 100}


def _scale_epochs(fields: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    return {**fields, "steps": cast(int, fields["steps"]) * 100}


class _StepsRun(Spec[str]):
    steps: int

    migrations = (Renamed("epochs", to="steps"), Rewrite(_epochs_to_steps))

    def create(self) -> str:
        return f"steps={self.steps}"


class _StepsRenamedFunctionRun(Spec[str]):
    steps: int
    warmup: int = 0

    # _epochs_to_steps renamed in a refactor.
    migrations = (
        Renamed("epochs", to="steps"),
        Rewrite(_scale_epochs),
        Added("warmup", default=0),
    )

    def create(self) -> str:
        return f"renamed steps={self.steps}"


def _grow_steps(target: type[Spec]) -> None:
    """epochs=3 under the trunk, steps=700 under _StepsRun; both moved to target."""
    epochs = _EpochsRun(epochs=3)
    epochs.create()
    _transplant_generation(epochs, _StepsRun)
    steps = _StepsRun(steps=700)
    steps.create()
    _transplant_generation(epochs, target)
    _transplant_generation(steps, target)
    _fresh_process()


def test_renaming_a_rewrite_function_keeps_reusing_results() -> None:
    _grow_steps(_StepsRenamedFunctionRun)

    assert _StepsRenamedFunctionRun(steps=300).load_existing() == "epochs=3"
    assert _StepsRenamedFunctionRun(steps=700).load_existing() == "steps=700"


class _StepsDroppedRenameRun(Spec[str]):
    steps: int
    warmup: int = 0

    # The Renamed before the Rewrite was dropped from the chain.
    migrations = (Rewrite(_epochs_to_steps), Added("warmup", default=0))

    def create(self) -> str:
        return f"dropped steps={self.steps}"


def test_a_rewrite_that_now_follows_another_step_recomputes() -> None:
    # The recorded Rewrite (after the Renamed) matches no declared step, so the
    # schema is skipped rather than risk applying the Rewrite twice.
    steps = _StepsRun(steps=700)
    steps.create()
    _transplant_generation(steps, _StepsDroppedRenameRun)
    _fresh_process()

    assert _StepsDroppedRenameRun(steps=700).status == "missing"


class _RatioRun(Spec[str]):
    ratio: int

    def create(self) -> str:
        return f"ratio={self.ratio}"


class _FloatRatioRun(Spec[str]):
    ratio: float

    migrations = (Retyped("ratio", was=int),)

    def create(self) -> str:
        return f"float ratio={self.ratio}"


class _CorrectedWasRun(Spec[str]):
    ratio: float
    scale: int = 1

    # was= corrected after _FloatRatioRun's results were stored.
    migrations = (Retyped("ratio", was=int | None), Added("scale", default=1))

    def create(self) -> str:
        return f"corrected ratio={self.ratio}"


def test_correcting_retyped_was_keeps_reusing_results() -> None:
    ratio = _RatioRun(ratio=2)
    ratio.create()
    _transplant_generation(ratio, _FloatRatioRun)
    float_ratio = _FloatRatioRun(ratio=0.5)
    float_ratio.create()
    _transplant_generation(float_ratio, _CorrectedWasRun)
    _fresh_process()

    assert _CorrectedWasRun(ratio=0.5).load_existing() == "float ratio=0.5"


class _OptionalRatioRun(Spec[str]):
    ratio: int | None

    migrations = (Retyped("ratio", was=int),)

    def create(self) -> str:
        return f"optional ratio={self.ratio}"


class _RetypedTwiceRun(Spec[str]):
    ratio: int | str | None

    migrations = (Retyped("ratio", was=int), Retyped("ratio", was=int | None))

    def create(self) -> str:
        return f"twice ratio={self.ratio}"


class _StrRatioRun(Spec[str]):
    ratio: str

    # The int -> float retype was dropped from the chain.
    migrations = (Retyped("ratio", was=float),)

    def create(self) -> str:
        return f"str ratio={self.ratio}"


def test_dropping_an_earlier_retype_of_a_field_is_stale() -> None:
    # Retypes are counted, so the float -> str retype takes the dropped one's
    # identity and the stored schema seems to have it; its float field then
    # doesn't fit, which blocks instead of reusing.
    float_ratio = _FloatRatioRun(ratio=0.5)
    float_ratio.create()
    _transplant_generation(float_ratio, _StrRatioRun)
    _fresh_process()

    assert _StrRatioRun(ratio="0.5").status == "stale"


def test_a_field_retyped_twice_tells_its_retypes_apart() -> None:
    ratio = _RatioRun(ratio=2)
    ratio.create()
    _transplant_generation(ratio, _OptionalRatioRun)
    optional = _OptionalRatioRun(ratio=None)
    optional.create()
    for donor in (ratio, optional):
        _transplant_generation(donor, _RetypedTwiceRun)
    _fresh_process()

    assert _RetypedTwiceRun(ratio=2).load_existing() == "ratio=2"
    assert _RetypedTwiceRun(ratio=None).load_existing() == "optional ratio=None"


class _XTrunk(Spec[str]):
    a: int

    def create(self) -> str:
        return f"trunk a={self.a}"


class _XBranch(Spec[str]):
    a: int
    x: int = 0

    migrations = (Added("x", default=0), Rewrite(lambda f: {**f, "x": f["a"]}))

    def create(self) -> str:
        return f"x a={self.a}"


class _YBranch(Spec[str]):
    a: int
    y: int = 0

    migrations = (Added("y", default=0), Rewrite(lambda f: {**f, "y": f["a"]}))

    def create(self) -> str:
        return f"y a={self.a}"


class _XYMerged(Spec[str]):
    a: int
    x: int = 0
    y: int = 0

    migrations = (
        Added("x", default=0),
        Rewrite(lambda f: {**f, "x": f["a"]}),
        Added("y", default=0),
        Rewrite(lambda f: {**f, "y": f["a"]}),
    )

    def create(self) -> str:
        return f"merged a={self.a}"


def test_lambda_rewrites_on_two_branches_merge() -> None:
    trunk = _XTrunk(a=1)
    trunk.create()
    for branch, a in ((_XBranch, 2), (_YBranch, 3)):
        _transplant_generation(trunk, branch)
        run = branch(a=a, **{"x" if branch is _XBranch else "y": a})
        run.create()
        _transplant_generation(run, _XYMerged)
    _transplant_generation(trunk, _XYMerged)
    _fresh_process()

    assert _XYMerged(a=1, x=1, y=1).load_existing() == "trunk a=1"
    assert _XYMerged(a=2, x=2, y=2).load_existing() == "x a=2"
    # The y branch already has its Rewrite, so the x branch's would run after
    # it, and Rewrites never swap order.
    assert _XYMerged(a=3, x=3, y=3).status == "stale"


class _NoWarmupRun(Spec[str]):
    a: int

    def create(self) -> str:
        return f"no warmup a={self.a}"


def test_removing_a_field_with_its_added_recomputes_quietly() -> None:
    # Documented on Added: the old results look like a sibling branch's.
    _, warmup, _ = _grow_branches()
    _transplant_generation(warmup, _NoWarmupRun)
    _fresh_process()

    assert _NoWarmupRun(a=2).status == "missing"


def _x_plus_one(fields: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    return {**fields, "x": cast(int, fields["a"]) + 1}


def _x_times_two(fields: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    return {**fields, "x": cast(int, fields["a"]) * 2}


class _XPlusBranch(Spec[str]):
    a: int
    x: int = 0

    migrations = (Added("x", default=0), Rewrite(_x_plus_one))

    def create(self) -> str:
        return f"plus a={self.a}"


class _XTimesBranch(Spec[str]):
    a: int
    x: int = 0

    migrations = (Added("x", default=0), Rewrite(_x_times_two))

    def create(self) -> str:
        return f"times a={self.a}"


def test_branches_sharing_a_schema_directory_do_not_block_each_other() -> None:
    # Both branches add x, so they write one schema directory, which records
    # whichever branch wrote first; it is each branch's current schema.
    trunk = _XTrunk(a=1)
    trunk.create()
    _transplant_generation(trunk, _XPlusBranch)
    _XPlusBranch(a=5, x=6).create()
    _transplant_generation(trunk, _XTimesBranch)
    plus_directory = _XPlusBranch(a=5, x=6)._base_dir.parent
    times_directory = _XTimesBranch(a=5, x=6)._base_dir.parent
    shutil.copytree(plus_directory, times_directory, dirs_exist_ok=True)
    _fresh_process()

    assert _XTimesBranch(a=1, x=2).load_existing() == "trunk a=1"
    assert _XTimesBranch(a=7, x=14).status == "missing"
