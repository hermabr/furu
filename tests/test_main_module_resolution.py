from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from furu.utils import (
    _install_main_module_alias,
    fully_qualified_name,
    resolve_fully_qualified_name,
)


def _write_uv_project(path: Path) -> None:
    """Make path a locked uv project in a git repo so spawned furu scripts run."""
    (path / "pyproject.toml").write_text(
        '[project]\nname = "scratch"\nversion = "0"\nrequires-python = ">=3.12"\n',
        encoding="utf-8",
    )
    subprocess.run(["uv", "lock"], cwd=path, check=True, capture_output=True)
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-qm", "init"]):
        subprocess.run(
            ["git", "-c", "user.email=t@t.t", "-c", "user.name=t", *args],
            cwd=path,
            check=True,
            capture_output=True,
        )


@contextmanager
def _simulated_python_m_main(
    monkeypatch: pytest.MonkeyPatch, *, spec_name: str
) -> Iterator[ModuleType]:
    main = sys.modules["__main__"]
    parent_name = spec_name.rpartition(".")[0]
    parent = ModuleType(parent_name)

    monkeypatch.setattr(
        main,
        "__spec__",
        SimpleNamespace(name=spec_name),
        raising=False,
    )
    monkeypatch.setitem(sys.modules, parent_name, parent)
    try:
        yield parent
    finally:
        if sys.modules.get(spec_name) is main:
            del sys.modules[spec_name]


def test_install_main_module_alias_registers_spec_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main = sys.modules["__main__"]

    with _simulated_python_m_main(
        monkeypatch, spec_name="furu_alias_lib.data"
    ) as parent:
        _install_main_module_alias()

        assert sys.modules["furu_alias_lib.data"] is main
        assert parent.data is main
        assert importlib.import_module("furu_alias_lib.data") is main


def test_install_main_module_alias_keeps_existing_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = ModuleType("furu_alias_existing.data")

    with _simulated_python_m_main(monkeypatch, spec_name="furu_alias_existing.data"):
        monkeypatch.setitem(sys.modules, "furu_alias_existing.data", existing)
        _install_main_module_alias()

        assert sys.modules["furu_alias_existing.data"] is existing


def test_fully_qualified_name_round_trips_to_identical_main_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main = sys.modules["__main__"]
    main_type = type("MainThing", (), {"__module__": "__main__"})

    with _simulated_python_m_main(monkeypatch, spec_name="furu_alias_rt.data"):
        monkeypatch.setattr(main, "MainThing", main_type, raising=False)
        _install_main_module_alias()

        assert fully_qualified_name(main_type) == "furu_alias_rt.data.MainThing"
        assert resolve_fully_qualified_name("furu_alias_rt.data.MainThing") is main_type


def test_main_without_spec_name_is_not_serializable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main = sys.modules["__main__"]
    main_type = type("ScriptThing", (), {"__module__": "__main__"})

    monkeypatch.setattr(main, "__spec__", None, raising=False)

    with pytest.raises(ValueError, match=r"__main__\.ScriptThing.*python -m"):
        fully_qualified_name(main_type)


def test_direct_script_main_objects_work_in_debug_mode(tmp_path: Path) -> None:
    _write_uv_project(tmp_path)
    script = tmp_path / "debug_script.py"
    script.write_text(
        """
from __future__ import annotations

import json
from pathlib import Path

from furu import Metadata, Spec


class Adder(Spec[int]):
    a: int
    b: int

    def metadata(self) -> Metadata:
        return Metadata(storage=Path("store"))

    def create(self) -> int:
        return self.a + self.b


if __name__ == "__main__":
    obj = Adder(a=1, b=2)
    first = obj.create()
    second = obj.create()

    print(
        json.dumps(
            {
                "base_dir_uses_debug": str(obj._base_dir).startswith(
                    "furu-data/debug/objects/__main__/Adder/"
                ),
                "first": first,
                "fqn": obj._fully_qualified_name,
                "object_id_startswith_fqn": obj.object_id.startswith("__main__.Adder:"),
                "second": second,
            },
            sort_keys=True,
        )
    )
""".lstrip(),
        encoding="utf-8",
    )

    env = os.environ.copy()
    pythonpath = [str(Path(__file__).resolve().parents[1] / "src")]
    if existing := env.get("PYTHONPATH"):
        pythonpath.append(existing)
    env["FURU_DEBUG_MODE"] = "true"
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    # The scratch project is not a git repo, so snapshotting would refuse to run.
    env["FURU_PROVENANCE__SNAPSHOT"] = "false"

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    output = json.loads(result.stdout.strip().splitlines()[-1])
    assert output == {
        "base_dir_uses_debug": True,
        "first": 3,
        "fqn": "__main__.Adder",
        "object_id_startswith_fqn": True,
        "second": 3,
    }


def test_fully_qualified_name_rejects_local_classes() -> None:
    class LocalThing:
        pass

    with pytest.raises(ValueError, match="local classes"):
        fully_qualified_name(LocalThing)


def test_python_m_main_objects_resolve_to_main_identity(tmp_path: Path) -> None:
    _write_uv_project(tmp_path)
    package_dir = tmp_path / "my_lib"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "data.py").write_text(
        """
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from collections.abc import Mapping

from furu import Metadata, Spec
from furu.storage._layout import data_dir_in
from furu.result.bundle import load_result_bundle, _save_result_bundle
from furu.result.codec import Codec
from furu.serializer.artifact import _from_json


@dataclass(frozen=True)
class Payload:
    value: int


class MainCodec(Codec[bytes]):
    @classmethod
    def matches(cls, value: object) -> bool:
        return isinstance(value, bytes)

    @classmethod
    def save(cls, value: bytes, artifact_directory: Path) -> Mapping[str, object]:
        (artifact_directory / "data.bin").write_bytes(value)
        return {}

    @classmethod
    def load(
        cls, metadata: Mapping[str, object], artifact_directory: Path
    ) -> bytes:
        return (artifact_directory / "data.bin").read_bytes()


class Adder(Spec[Payload]):
    a: int
    b: int

    def metadata(self) -> Metadata:
        return Metadata(storage=Path("store"))

    def create(self) -> Payload:
        return Payload(self.a + self.b)


if __name__ == "__main__":
    import importlib
    import sys

    reimported = importlib.import_module("my_lib.data")

    obj = Adder(a=1, b=2)
    artifact = obj._artifact_data
    artifact_loaded = _from_json(artifact)
    first = obj.create()
    second = obj.create()

    payload_bundle = Path("payload-bundle")
    _save_result_bundle(
        Payload(9),
        payload_bundle,
        declared_type=Payload,
        result_codecs=(),
        data_dir=data_dir_in(payload_bundle.parent),
    )
    payload_loaded = load_result_bundle(
        payload_bundle,
        data_dir=data_dir_in(payload_bundle.parent),
        declared_type=Payload,
    )

    codec_bundle = Path("codec-bundle")
    _save_result_bundle(
        b"abc",
        codec_bundle,
        declared_type=bytes,
        result_codecs=(MainCodec,),
        data_dir=data_dir_in(codec_bundle.parent),
    )
    codec_loaded = load_result_bundle(
        codec_bundle,
        data_dir=data_dir_in(codec_bundle.parent),
        declared_type=bytes,
    )

    print(
        json.dumps(
            {
                "artifact_class": artifact["|class"],
                "artifact_loaded_same_class": type(artifact_loaded) is Adder,
                "codec_loaded": codec_loaded.decode(),
                "first_same_payload_class": type(first) is Payload,
                "fqn": obj._fully_qualified_name,
                "object_id_startswith_fqn": obj.object_id.startswith(
                    "my_lib.data.Adder:"
                ),
                "payload_loaded_same_class": type(payload_loaded) is Payload,
                "reimport_is_running_main": reimported is sys.modules["__main__"],
                "reimported_class_is_identical": reimported.Adder is Adder,
                "second_same_payload_class": type(second) is Payload,
            },
            sort_keys=True,
        )
    )
""".lstrip(),
        encoding="utf-8",
    )

    env = os.environ.copy()
    pythonpath = [
        str(Path(__file__).resolve().parents[1] / "src"),
        str(tmp_path),
    ]
    if existing := env.get("PYTHONPATH"):
        pythonpath.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    # The scratch project is not a git repo, so snapshotting would refuse to run.
    env["FURU_PROVENANCE__SNAPSHOT"] = "false"

    result = subprocess.run(
        [sys.executable, "-m", "my_lib.data"],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    output = json.loads(result.stdout.strip().splitlines()[-1])
    assert output == {
        "artifact_class": "my_lib.data.Adder",
        "artifact_loaded_same_class": True,
        "codec_loaded": "abc",
        "first_same_payload_class": True,
        "fqn": "my_lib.data.Adder",
        "object_id_startswith_fqn": True,
        "payload_loaded_same_class": True,
        "reimport_is_running_main": True,
        "reimported_class_is_identical": True,
        "second_same_payload_class": True,
    }
