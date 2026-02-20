import pytest

from furu.query import (
    AndNode,
    BetweenNode,
    EqNode,
    ExistsNode,
    FALSE,
    InNode,
    IsANode,
    NotNode,
    OrNode,
    Q,
    QueryExpr,
    RelatedToNode,
    TRUE,
    TypeIsNode,
)


class BaseType:
    pass


class ChildType(BaseType):
    pass


def test_field_ref_path_building_and_comparisons() -> None:
    query = Q.config.dataset.name == "mnist"
    assert isinstance(query, QueryExpr)
    assert query.to_ast() == EqNode(path="config.dataset.name", value="mnist")

    indexed = Q.config.deps[0].name == "foo"
    assert indexed.to_ast() == EqNode(path="config.deps.0.name", value="foo")

    keyed = Q.config["weird_key"] == "bar"
    assert keyed.to_ast() == EqNode(path="config.weird_key", value="bar")


def test_field_ref_rejects_invalid_index_keys() -> None:
    with pytest.raises(ValueError, match="cannot contain '\\.'"):
        _ = Q.config["bad.key"]

    with pytest.raises(ValueError, match="cannot be empty"):
        _ = Q.config[""]

    with pytest.raises(ValueError, match="Negative indices"):
        _ = Q.config[-1]


def test_boolean_composition_flattens_and_prevents_truthiness() -> None:
    q1 = Q.exp.result_status == "success"
    q2 = Q.config.dataset.name == "mnist"
    q3 = Q.config.seed == 123

    combined_and = q1 & (q2 & q3)
    and_node = combined_and.to_ast()
    assert isinstance(and_node, AndNode)
    assert and_node.args == [q1.to_ast(), q2.to_ast(), q3.to_ast()]

    combined_or = (q1 | q2) | q3
    or_node = combined_or.to_ast()
    assert isinstance(or_node, OrNode)
    assert or_node.args == [q1.to_ast(), q2.to_ast(), q3.to_ast()]

    inverted = ~q1
    assert inverted.to_ast() == NotNode(arg=q1.to_ast())

    with pytest.raises(TypeError, match="cannot be used as booleans"):
        bool(q1)


def test_dsl_methods_map_to_expected_nodes() -> None:
    assert Q.config.seed.exists().to_ast() == ExistsNode(path="config.seed")

    between = Q.config.lr.between(1e-4, 1e-3, inclusive="left")
    assert between.to_ast() == BetweenNode(
        path="config.lr",
        low=1e-4,
        high=1e-3,
        inclusive="left",
    )

    in_query = Q.config.dataset.name.in_("mnist", "cifar10")
    assert in_query.to_ast() == InNode(
        path="config.dataset.name",
        values=["mnist", "cifar10"],
    )


def test_type_methods_accept_string_class_and_instance() -> None:
    expected_child = f"{ChildType.__module__}.{ChildType.__qualname__}"
    expected_base = f"{BaseType.__module__}.{BaseType.__qualname__}"

    assert Q.config.data.type_is(ChildType).to_ast() == TypeIsNode(
        path="config.data",
        type=expected_child,
    )
    assert Q.config.data.is_a(ChildType()).to_ast() == IsANode(
        path="config.data",
        type=expected_child,
    )
    assert Q.config.data.related_to(expected_base).to_ast() == RelatedToNode(
        path="config.data",
        type=expected_base,
    )


def test_true_false_constants_are_query_exprs() -> None:
    assert isinstance(TRUE, QueryExpr)
    assert isinstance(FALSE, QueryExpr)
