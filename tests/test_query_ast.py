"""Tests for query AST model definitions."""

from pydantic import TypeAdapter

from furu.query.ast import (
    ContainsQuery,
    EqQuery,
    ExistsQuery,
    FalseQuery,
    GteQuery,
    InQuery,
    Query,
    RegexQuery,
    TrueQuery,
)


def test_query_union_builds_true_false_nodes() -> None:
    adapter = TypeAdapter(Query)

    assert isinstance(adapter.validate_python({"op": "true"}), TrueQuery)
    assert isinstance(adapter.validate_python({"op": "false"}), FalseQuery)


def test_comparison_node_round_trips() -> None:
    adapter = TypeAdapter(Query)
    query = adapter.validate_python(
        {"op": "eq", "path": "exp.result_status", "value": "success"}
    )

    assert isinstance(query, EqQuery)
    assert query.path == "exp.result_status"
    assert query.value == "success"


def test_boolean_composition_node_round_trips() -> None:
    adapter = TypeAdapter(Query)
    query = adapter.validate_python(
        {
            "op": "and",
            "args": [
                {"op": "gte", "path": "config.lr", "value": 0.1},
                {"op": "exists", "path": "config.seed"},
            ],
        },
    )
    assert query.op == "and"
    assert len(query.args) == 2

    and_payload = query.model_dump(mode="json")
    assert and_payload["op"] == "and"
    assert and_payload["args"][0]["op"] == "gte"


def test_membership_and_string_helpers_models() -> None:
    assert isinstance(InQuery(op="in", path="config.tags", values=["a", "b"]), InQuery)
    assert isinstance(
        ContainsQuery(
            op="contains",
            path="config.name",
            value="foo",
            case_sensitive=False,
        ),
        ContainsQuery,
    )


def test_regex_query_requires_flags() -> None:
    assert isinstance(
        RegexQuery(op="regex", path="config.message", pattern="^x$", flags="i"),
        RegexQuery,
    )
    adapter = TypeAdapter(Query)
    parsed = adapter.validate_python(
        {
            "op": "startswith",
            "path": "exp.namespace",
            "prefix": "foo",
            "case_sensitive": True,
        }
    )
    assert parsed.model_dump(mode="json")["prefix"] == "foo"


def test_exists_query_uses_path() -> None:
    assert isinstance(ExistsQuery(op="exists", path="state.something"), ExistsQuery)
    assert ExistsQuery(path="state.something", op="exists").path == "state.something"


def test_gte_query_defaults() -> None:
    query = GteQuery(op="gte", path="config.version", value=1)
    assert query.value == 1
