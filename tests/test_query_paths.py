from furu.query.paths import PATH_MISSING, get_path


def test_get_path_traverses_nested_dicts() -> None:
    doc = {
        "exp": {
            "result_status": "success",
            "attempt": {"number": 2},
        }
    }

    assert get_path(doc, "exp.result_status") == "success"
    assert get_path(doc, "exp.attempt.number") == 2


def test_get_path_traverses_list_indices() -> None:
    doc = {
        "config": {
            "deps": [
                {"name": "dataset"},
                {"name": "model"},
            ]
        }
    }

    assert get_path(doc, "config.deps.0.name") == "dataset"
    assert get_path(doc, "config.deps.1.name") == "model"


def test_get_path_returns_missing_for_invalid_paths() -> None:
    doc = {
        "config": {
            "deps": [
                {"name": "dataset"},
            ]
        }
    }

    assert get_path(doc, "config.missing") is PATH_MISSING
    assert get_path(doc, "config.deps.2.name") is PATH_MISSING
    assert get_path(doc, "config.deps.one.name") is PATH_MISSING
