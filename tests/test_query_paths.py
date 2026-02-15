"""Tests for query path traversal helpers."""

from furu.query.paths import PATH_MISSING, get_path


def test_get_path_walks_dict_path() -> None:
    """Resolver should follow dot-separated keys through nested dictionaries."""

    doc = {"config": {"dataset": {"name": "mnist", "size": 128}}}

    assert get_path(doc, "config.dataset.name") == "mnist"
    assert get_path(doc, "config.dataset.size") == 128


def test_get_path_walks_list_path() -> None:
    """Resolver should index lists and tuples with numeric segments."""

    doc = {"deps": [{"name": "a"}, {"name": "b"}]}

    assert get_path(doc, "deps.1.name") == "b"
    assert get_path(doc, "deps.0") == {"name": "a"}


def test_get_path_returns_missing_for_absent_segments() -> None:
    """Resolver should return PATH_MISSING when keys are not present."""

    doc = {"config": {"seed": 42}}

    assert get_path(doc, "config.missing") is PATH_MISSING
    assert get_path(doc, "config.seed.unit") is PATH_MISSING


def test_get_path_returns_missing_for_invalid_index() -> None:
    """Resolver should return PATH_MISSING for invalid list indexing."""

    doc = {"values": [10, 20]}

    assert get_path(doc, "values.two") is PATH_MISSING
    assert get_path(doc, "values.10") is PATH_MISSING
