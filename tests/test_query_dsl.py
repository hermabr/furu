"""Tests for the Python DSL that builds query AST nodes."""

from __future__ import annotations

from typing import cast

import pytest

from furu.query import Q
from furu.query.ast import AndQuery, EqQuery, OrQuery


class BaseType:
    pass


class ConcreteType(BaseType):
    pass


def test_field_ref_paths_and_index_access() -> None:
    assert cast(EqQuery, Q.config.dataset.name == "mnist").path == "config.dataset.name"
    assert cast(EqQuery, Q.config.deps[0].name == "x").path == "config.deps.0.name"
    assert cast(EqQuery, Q.config["weird_key"] == 1).path == "config.weird_key"


def test_field_ref_rejects_dot_in_string_key() -> None:
    with pytest.raises(ValueError, match="may not contain '.'"):
        Q.config["not.allowed"]


def test_scalar_comparisons_compile_to_ast() -> None:
    assert (Q.exp.result_status == "success").op == "eq"
    assert (Q.exp.metrics.loss < 1.0).op == "lt"
    assert (Q.exp.metrics.loss > 0.0).op == "gt"
    assert (Q.exp.metrics.loss <= 1.0).op == "lte"
    assert (Q.exp.metrics.loss >= 0.0).op == "gte"


def test_query_methods_compile_to_ast() -> None:
    assert Q.config.seed.exists().op == "exists"
    assert Q.config.seed.missing().op == "missing"
    assert Q.config.lr.between(1e-4, 1e-3, inclusive="left").op == "between"
    assert Q.config.tag.in_("a", "b").op == "in"
    assert Q.config.tag.not_in("c", "d").op == "nin"
    assert Q.config.name.contains("res").op == "contains"
    assert Q.config.name.startswith("run").op == "startswith"
    assert Q.config.name.endswith(".json").op == "endswith"
    assert Q.config.message.regex("^abc", flags="i").op == "regex"


def test_type_methods_accept_classes_strings_and_instances() -> None:
    assert Q.config.data.type_is("dashboard.pipelines.TrainModel").op == "type_is"
    assert Q.config.data.type_is(BaseType).op == "type_is"
    assert Q.config.data.is_a(ConcreteType).op == "is_a"
    assert Q.config.data.related_to(ConcreteType()).op == "related_to"


def test_boolean_composition_flattens_nested_operators() -> None:
    combined = (Q.exp.result_status == "success") & (Q.config.lr > 0.0)
    and_query = cast(AndQuery, combined)
    assert and_query.op == "and"
    assert len(and_query.args) == 2

    nested = (Q.exp.result_status == "success") & (
        (Q.config.lr > 0.0) & (Q.config.batch == 2)
    )
    nested_query = cast(AndQuery, nested)
    assert nested_query.op == "and"
    assert len(nested_query.args) == 3

    either = (Q.config.seed == 1) | (Q.config.seed == 2)
    either_query = cast(OrQuery, either)
    assert either_query.op == "or"
    assert len(either_query.args) == 2


def test_negation_and_truthiness_guard() -> None:
    inverted = ~(Q.config.seed == 1)
    assert inverted.op == "not"
    with pytest.raises(TypeError, match="not truthy"):
        bool(Q.config.seed == 1)
