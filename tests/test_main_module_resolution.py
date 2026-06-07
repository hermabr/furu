from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from furu.utils import fully_qualified_name, resolve_fully_qualified_name


def test_fully_qualified_name_uses_main_spec_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main = sys.modules["__main__"]
    main_type = type("MainThing", (), {"__module__": "__main__"})

    monkeypatch.setattr(
        main,
        "__spec__",
        SimpleNamespace(name="my_lib.data"),
        raising=False,
    )
    monkeypatch.setattr(main, "MainThing", main_type, raising=False)

    assert fully_qualified_name(main_type) == "my_lib.data.MainThing"
    assert resolve_fully_qualified_name("my_lib.data.MainThing") is main_type


def test_main_without_spec_name_is_not_serializable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main = sys.modules["__main__"]
    main_type = type("ScriptThing", (), {"__module__": "__main__"})

    monkeypatch.setattr(main, "__spec__", None, raising=False)

    with pytest.raises(ValueError, match="python -m"):
        fully_qualified_name(main_type)


def test_direct_script_main_objects_work_in_debug_mode(tmp_path: Path) -> None:
    script = tmp_path / "debug_script.py"
    script.write_text(
        """
from __future__ import annotations

import json
from pathlib import Path

from furu import Furu


class Adder(Furu[int]):
    a: int
    b: int

    @property
    def storage_root(self) -> Path:
        return Path("store")

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
                    "furu/debug/objects/__main__/Adder/"
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
    package_dir = tmp_path / "my_lib"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "data.py").write_text(
        """
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from furu import Furu
from furu.result import load_result_bundle, _save_result_bundle
from furu.result.codec import ResultCodec
from furu.serializer.artifact import _from_json


@dataclass(frozen=True)
class Payload:
    value: int


class MainCodec(ResultCodec[bytes]):
    @classmethod
    def matches(cls, value: object) -> bool:
        return isinstance(value, bytes)

    @classmethod
    def dump(cls, value: bytes, *, artifact_dir: Path) -> None:
        (artifact_dir / "data.bin").write_bytes(value)

    @classmethod
    def load(cls, *, artifact_dir: Path) -> bytes:
        return (artifact_dir / "data.bin").read_bytes()


class Adder(Furu[Payload]):
    a: int
    b: int

    @property
    def storage_root(self) -> Path:
        return Path("store")

    def create(self) -> Payload:
        return Payload(self.a + self.b)


if __name__ == "__main__":
    obj = Adder(a=1, b=2)
    artifact = obj._artifact_data
    artifact_loaded = _from_json(artifact)
    first = obj.create()
    second = obj.create()

    payload_bundle = Path("payload-bundle")
    _save_result_bundle(Payload(9), payload_bundle, result_codecs=())
    payload_loaded = load_result_bundle(payload_bundle)

    codec_bundle = Path("codec-bundle")
    _save_result_bundle(b"abc", codec_bundle, result_codecs=(MainCodec,))
    codec_loaded = load_result_bundle(codec_bundle)

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
        "second_same_payload_class": True,
    }
