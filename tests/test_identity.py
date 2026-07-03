import json
from dataclasses import dataclass
from typing import Annotated

import pytest
from pydantic import BaseModel

import furu
from furu import Spec
from furu.diff import FieldDiff
from furu.storage._layout import schema_snapshot_path_in
from furu.utils import fully_qualified_name


@dataclass(frozen=True)
class SubsetOrder:
    fraction: float = 1.0
    variant: str = "global-random-subset"


class OptimizerSettings(BaseModel):
    name: str = "adamw"
    beta1: float = 0.9


class DiffCorpus(Spec[str]):
    tokenizer: str = "bpe"
    order: SubsetOrder = SubsetOrder()

    def create(self) -> str:
        return self.tokenizer


class ShuffledCorpus(DiffCorpus):
    seed: int = 0


class TrainRun(Spec[str]):
    corpus: DiffCorpus
    model: str = "100m"
    max_tokens: int = 1_000_000_000
    learning_rate: float = 3e-4

    def create(self) -> str:
        return self.model


class EvalRun(Spec[str]):
    run: TrainRun

    def create(self) -> str:
        return self.run.model


class OptimizerRun(Spec[str]):
    optimizer: OptimizerSettings

    def create(self) -> str:
        return self.optimizer.name


class NotedValue(Spec[str]):
    name: str
    note: Annotated[str, furu.skip_hash] = ""

    def create(self) -> str:
        return self.name


class UrlsValue(Spec[int]):
    urls: list[str]

    def create(self) -> int:
        return len(self.urls)


class SnapshotValue(Spec[str]):
    name: str

    def create(self) -> str:
        return self.name


class FailingValue(Spec[str]):
    name: str

    def create(self) -> str:
        raise RuntimeError("boom")


def test_explain_renders_nested_spec_block() -> None:
    run = TrainRun(corpus=DiffCorpus())
    assert run.explain() == (
        f"TrainRun"
        f"  schema={run._artifact_schema_hash[:5]}"
        f"  fields={run._artifact_hash[:5]}\n"
        f"  corpus         DiffCorpus · key={run.corpus._artifact_hash[:5]}…\n"
        '  model          "100m"\n'
        "  max_tokens     1_000_000_000\n"
        "  learning_rate  0.0003"
    )


def test_explain_renders_plain_dataclass_field() -> None:
    corpus = DiffCorpus()
    assert corpus.explain() == (
        f"DiffCorpus"
        f"  schema={corpus._artifact_schema_hash[:5]}"
        f"  fields={corpus._artifact_hash[:5]}\n"
        '  tokenizer  "bpe"\n'
        "  order      SubsetOrder(fraction=1.0, variant='global-random-subset')"
    )


def test_explain_full_depth_expands_nested_spec_block() -> None:
    run = TrainRun(corpus=DiffCorpus())
    assert run.explain(depth="full") == (
        f"TrainRun"
        f"  schema={run._artifact_schema_hash[:5]}"
        f"  fields={run._artifact_hash[:5]}\n"
        f"  corpus         DiffCorpus"
        f"  schema={run.corpus._artifact_schema_hash[:5]}"
        f"  fields={run.corpus._artifact_hash[:5]}\n"
        '    tokenizer  "bpe"\n'
        "    order      SubsetOrder\n"
        "      fraction  1.0\n"
        '      variant   "global-random-subset"\n'
        '  model          "100m"\n'
        "  max_tokens     1_000_000_000\n"
        "  learning_rate  0.0003"
    )


def test_explain_integer_depth_limits_expansion() -> None:
    evaluation = EvalRun(run=TrainRun(corpus=DiffCorpus()))
    run = evaluation.run
    assert evaluation.explain(depth=1) == (
        f"EvalRun"
        f"  schema={evaluation._artifact_schema_hash[:5]}"
        f"  fields={evaluation._artifact_hash[:5]}\n"
        f"  run  TrainRun"
        f"  schema={run._artifact_schema_hash[:5]}"
        f"  fields={run._artifact_hash[:5]}\n"
        f"    corpus         DiffCorpus · key={run.corpus._artifact_hash[:5]}…\n"
        '    model          "100m"\n'
        "    max_tokens     1_000_000_000\n"
        "    learning_rate  0.0003"
    )


def test_explain_full_depth_expands_pydantic_model() -> None:
    run = OptimizerRun(optimizer=OptimizerSettings())
    assert run.explain(depth="full") == (
        f"OptimizerRun"
        f"  schema={run._artifact_schema_hash[:5]}"
        f"  fields={run._artifact_hash[:5]}\n"
        "  optimizer  OptimizerSettings\n"
        '    name   "adamw"\n'
        "    beta1  0.9"
    )


def test_diff_matches_proposal_example() -> None:
    run_a = TrainRun(
        corpus=DiffCorpus(order=SubsetOrder(fraction=0.1)), learning_rate=3e-4
    )
    run_b = TrainRun(
        corpus=DiffCorpus(order=SubsetOrder(fraction=0.01)), learning_rate=1e-3
    )
    assert furu.diff(run_a, run_b) == [
        FieldDiff(path="learning_rate", a=3e-4, b=1e-3),
        FieldDiff(path="corpus.order.fraction", a=0.1, b=0.01),
    ]


def test_diff_of_identical_specs_is_empty() -> None:
    assert furu.diff(TrainRun(corpus=DiffCorpus()), TrainRun(corpus=DiffCorpus())) == []


def test_diff_class_mismatch_is_the_first_field_diff() -> None:
    corpus = DiffCorpus()
    run = TrainRun(corpus=corpus)
    assert furu.diff(corpus, run) == [
        FieldDiff(
            path="",
            a=fully_qualified_name(DiffCorpus),
            b=fully_qualified_name(TrainRun),
        )
    ]


def test_diff_nested_class_mismatch_does_not_recurse() -> None:
    run_a = TrainRun(corpus=DiffCorpus())
    run_b = TrainRun(corpus=ShuffledCorpus())
    assert furu.diff(run_a, run_b) == [
        FieldDiff(
            path="corpus",
            a=fully_qualified_name(DiffCorpus),
            b=fully_qualified_name(ShuffledCorpus),
        )
    ]


def test_diff_omits_skip_hash_fields() -> None:
    assert (
        furu.diff(NotedValue(name="x", note="a"), NotedValue(name="x", note="b")) == []
    )


def test_diff_reports_collections_as_whole_values() -> None:
    assert furu.diff(UrlsValue(urls=["a", "b"]), UrlsValue(urls=["a", "c"])) == [
        FieldDiff(path="urls", a=["a", "b"], b=["a", "c"])
    ]


def test_schema_snapshot_written_on_first_store_and_never_rewritten() -> None:
    first = SnapshotValue(name="first")
    schema_path = schema_snapshot_path_in(first._base_dir)
    assert not schema_path.exists()

    first.create()
    assert json.loads(schema_path.read_text()) == first._schema_data

    stat_before = schema_path.stat()
    second = SnapshotValue(name="second")
    assert schema_snapshot_path_in(second._base_dir) == schema_path
    second.create()
    stat_after = schema_path.stat()
    assert (stat_after.st_ino, stat_after.st_mtime_ns) == (
        stat_before.st_ino,
        stat_before.st_mtime_ns,
    )


def test_schema_snapshot_not_written_when_create_fails() -> None:
    failing = FailingValue(name="x")
    with pytest.raises(RuntimeError, match="boom"):
        failing.create()
    assert not schema_snapshot_path_in(failing._base_dir).exists()
