from huldra.runtime import tracebacks


def test_rich_uncaught_tracebacks_enabled_by_default() -> None:
    assert tracebacks._RICH_UNCAUGHT_ENABLED is True
