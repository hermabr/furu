from furu.query import (
    BetweenNode,
    ContainsNode,
    EndswithNode,
    EqNode,
    ExistsNode,
    GtNode,
    GteNode,
    InNode,
    IsANode,
    LtNode,
    LteNode,
    MissingNode,
    NeNode,
    NinNode,
    RelatedToNode,
    StartswithNode,
    TypeIsNode,
    matches,
)


class EvalBaseType:
    pass


class EvalChildType(EvalBaseType):
    pass


class EvalGrandchildType(EvalChildType):
    pass


def test_matches_comparison_nodes() -> None:
    doc = {
        "exp": {
            "status": "success",
            "attempt": 3,
            "score": 0.91,
        }
    }

    assert matches(doc, EqNode(path="exp.status", value="success")) is True
    assert matches(doc, NeNode(path="exp.status", value="failed")) is True
    assert matches(doc, GtNode(path="exp.attempt", value=2)) is True
    assert matches(doc, GteNode(path="exp.score", value=0.91)) is True
    assert matches(doc, LtNode(path="exp.attempt", value=4)) is True
    assert matches(doc, LteNode(path="exp.attempt", value=3)) is True


def test_matches_between_inclusive_modes() -> None:
    doc = {"config": {"lr": 0.001}}

    assert (
        matches(
            doc,
            BetweenNode(path="config.lr", low=0.001, high=0.01, inclusive="both"),
        )
        is True
    )
    assert (
        matches(
            doc,
            BetweenNode(path="config.lr", low=0.001, high=0.01, inclusive="left"),
        )
        is True
    )
    assert (
        matches(
            doc,
            BetweenNode(path="config.lr", low=0.0, high=0.001, inclusive="right"),
        )
        is True
    )
    assert (
        matches(
            doc,
            BetweenNode(path="config.lr", low=0.001, high=0.01, inclusive="none"),
        )
        is False
    )


def test_matches_in_and_nin_nodes() -> None:
    doc = {"config": {"dataset": "mnist"}}

    assert (
        matches(
            doc,
            InNode(path="config.dataset", values=["imagenet", "mnist"]),
        )
        is True
    )
    assert (
        matches(doc, NinNode(path="config.dataset", values=["cifar10", "svhn"])) is True
    )


def test_matches_exists_and_missing_nodes() -> None:
    doc = {"config": {"dataset": {"name": "mnist"}}}

    assert matches(doc, ExistsNode(path="config.dataset.name")) is True
    assert matches(doc, MissingNode(path="config.dataset.version")) is True


def test_matches_string_predicates() -> None:
    doc = {"exp": {"namespace": "dashboard.pipelines.TrainModel"}}

    assert (
        matches(
            doc,
            ContainsNode(path="exp.namespace", value="pipelines", case_sensitive=True),
        )
        is True
    )
    assert (
        matches(
            doc,
            StartswithNode(
                path="exp.namespace", prefix="dashboard", case_sensitive=True
            ),
        )
        is True
    )
    assert (
        matches(
            doc,
            EndswithNode(
                path="exp.namespace", suffix="trainmodel", case_sensitive=False
            ),
        )
        is True
    )


def test_matches_type_relationship_nodes() -> None:
    base_type_name = f"{EvalBaseType.__module__}.{EvalBaseType.__qualname__}"
    child_type_name = f"{EvalChildType.__module__}.{EvalChildType.__qualname__}"
    grandchild_type_name = (
        f"{EvalGrandchildType.__module__}.{EvalGrandchildType.__qualname__}"
    )

    doc = {
        "config": {
            "obj": {
                "__class__": grandchild_type_name,
            }
        }
    }

    assert (
        matches(doc, TypeIsNode(path="config.obj", type=grandchild_type_name)) is True
    )
    assert matches(doc, IsANode(path="config.obj", type=child_type_name)) is True
    assert matches(doc, RelatedToNode(path="config.obj", type=base_type_name)) is True
