import pytest

from furu.query import AndNode, EqNode, NotNode, validate_query


def test_validate_query_accepts_small_tree() -> None:
    query = AndNode(
        args=[
            EqNode(path="config.lr", value=0.01),
            NotNode(arg=EqNode(path="exp.result_status", value="failed")),
        ]
    )

    validate_query(query, max_node_count=10, max_depth=10)


def test_validate_query_rejects_node_limit() -> None:
    query = AndNode(
        args=[
            EqNode(path="config.lr", value=0.01),
            EqNode(path="config.seed", value=42),
        ]
    )

    with pytest.raises(ValueError, match="max node count"):
        validate_query(query, max_node_count=2, max_depth=10)


def test_validate_query_rejects_depth_limit() -> None:
    query = NotNode(arg=NotNode(arg=NotNode(arg=EqNode(path="config.lr", value=0.1))))

    with pytest.raises(ValueError, match="max depth"):
        validate_query(query, max_node_count=10, max_depth=3)
