from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import furu.utils as utils
from furu.result import load_result_bundle, save_result_bundle
from furu.result.codec import ResultCodec, ResultRegistry, resolve_result_codec
from furu.serialize import _from_json, to_json
from furu.utils import fully_qualified_name, resolve_qualified_name, set_main_module


@dataclass(frozen=True)
class _MainDataclass:
    value: int


class _MainCodecValue:
    pass


class _MainCodec(ResultCodec):
    @classmethod
    def matches(cls, value: object) -> bool:
        return isinstance(value, _MainCodecValue)

    @classmethod
    def dump(
        cls,
        value: object,
        *,
        artifact_dir: Path,
    ) -> None:
        artifact_dir.joinpath("value.txt").write_text("ok", encoding="utf-8")

    @classmethod
    def load(cls, *, artifact_dir: Path) -> object:
        artifact_dir.joinpath("value.txt").read_text(encoding="utf-8")
        return _MainCodecValue()


@pytest.fixture
def main_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    main = sys.modules["__main__"]
    monkeypatch.setattr(utils, "_MAIN_MODULE_OVERRIDE", None)
    monkeypatch.setattr(main, "__spec__", None, raising=False)
    return main


def test_fully_qualified_name_uses_main_spec_name(
    main_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        main_module,
        "__spec__",
        SimpleNamespace(name="my_lib.data"),
        raising=False,
    )
    cls = type("Adder", (), {"__module__": "__main__"})

    assert fully_qualified_name(cls) == "my_lib.data.Adder"


def test_fully_qualified_name_infers_package_module_from_main_file(
    tmp_path: Path,
    main_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_dir = tmp_path / "src" / "my_lib"
    package_dir.mkdir(parents=True)
    package_dir.joinpath("__init__.py").write_text("", encoding="utf-8")
    data_file = package_dir / "data.py"
    data_file.write_text("", encoding="utf-8")

    monkeypatch.setattr(main_module, "__file__", str(data_file), raising=False)
    cls = type("Adder", (), {"__module__": "__main__"})

    assert fully_qualified_name(cls) == "my_lib.data.Adder"


def test_fully_qualified_name_does_not_infer_from_non_package_main_file(
    tmp_path: Path,
    main_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_dir = tmp_path / "my_lib"
    script_dir.mkdir()
    data_file = script_dir / "data.py"
    data_file.write_text("", encoding="utf-8")

    monkeypatch.setattr(main_module, "__file__", str(data_file), raising=False)
    cls = type("Adder", (), {"__module__": "__main__"})

    with pytest.raises(ValueError, match="Cannot serialize objects from __main__"):
        fully_qualified_name(cls)


def test_set_main_module_overrides_main_resolution(
    main_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main_module, "__file__", None, raising=False)
    set_main_module("my_lib.data")
    cls = type("Adder", (), {"__module__": "__main__"})

    assert fully_qualified_name(cls) == "my_lib.data.Adder"


def test_resolve_qualified_name_returns_running_main_attribute(
    main_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        main_module,
        "__spec__",
        SimpleNamespace(name="my_lib.data"),
        raising=False,
    )
    cls = type("Adder", (), {"__module__": "__main__"})
    monkeypatch.setattr(main_module, "Adder", cls, raising=False)

    assert resolve_qualified_name("my_lib.data.Adder") is cls


def test_serialized_dataclass_from_main_loads_without_double_import(
    main_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        main_module,
        "__spec__",
        SimpleNamespace(name="my_lib.data"),
        raising=False,
    )
    monkeypatch.setattr(_MainDataclass, "__module__", "__main__")
    monkeypatch.setattr(main_module, "_MainDataclass", _MainDataclass, raising=False)

    payload = to_json(_MainDataclass(value=7))

    assert cast(dict[str, Any], payload)["|class"] == "my_lib.data._MainDataclass"
    loaded = _from_json(payload)
    assert type(loaded) is _MainDataclass
    assert loaded == _MainDataclass(value=7)


def test_result_dataclass_from_main_loads_without_double_import(
    tmp_path: Path,
    main_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        main_module,
        "__spec__",
        SimpleNamespace(name="my_lib.data"),
        raising=False,
    )
    monkeypatch.setattr(_MainDataclass, "__module__", "__main__")
    monkeypatch.setattr(main_module, "_MainDataclass", _MainDataclass, raising=False)

    bundle_dir = tmp_path / "bundle"
    save_result_bundle(
        _MainDataclass(value=9),
        bundle_dir,
        registry=ResultRegistry(),
    )

    loaded = load_result_bundle(bundle_dir)
    assert type(loaded) is _MainDataclass
    assert loaded == _MainDataclass(value=9)


def test_codec_from_main_resolves_without_double_import(
    main_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        main_module,
        "__spec__",
        SimpleNamespace(name="my_lib.data"),
        raising=False,
    )
    monkeypatch.setattr(_MainCodec, "__module__", "__main__")
    monkeypatch.setattr(main_module, "_MainCodec", _MainCodec, raising=False)

    assert _MainCodec._codec_id() == "my_lib.data._MainCodec"
    assert resolve_result_codec("my_lib.data._MainCodec") is _MainCodec
