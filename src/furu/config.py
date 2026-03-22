import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict


_PROJECT_MARKERS = ("pyproject.toml", ".git")
_PYPROJECT_PATH = "pyproject.toml"
_PYPROJECT_TOOL_SECTION = ("tool", "furu")


@dataclass(slots=True)
class _FuruDirectories:
    data: Path


class FuruSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    debug: bool = False
    data_dir: Path = Path(".furu/data")
    project_root: Path | None = None


def _discover_project_root(start_path: Path | None = None) -> Path | None:
    start = (start_path or Path.cwd()).resolve()
    if start.is_file():
        start = start.parent
    for candidate in (start, *start.parents):
        if any((candidate / marker).exists() for marker in _PROJECT_MARKERS):
            return candidate
    return start


def _load_pyproject_settings(project_root: Path | None) -> dict[str, object]:
    if project_root is None:
        return {}

    pyproject_path = project_root / _PYPROJECT_PATH
    if not pyproject_path.exists():
        return {}

    with pyproject_path.open("rb") as f:
        pyproject = tomllib.load(f)

    tool_settings: object = pyproject
    for key in _PYPROJECT_TOOL_SECTION:
        if not isinstance(tool_settings, dict):
            return {}
        tool_settings = tool_settings.get(key, {})

    if not isinstance(tool_settings, dict):
        return {}

    return dict(tool_settings)


def _load_env_settings() -> dict[str, object]:
    env_settings: dict[str, object] = {}

    if "FURU_DEBUG" in os.environ:
        env_settings["debug"] = os.environ["FURU_DEBUG"]
    if "FURU_DATA_DIR" in os.environ:
        env_settings["data_dir"] = os.environ["FURU_DATA_DIR"]
    if "FURU_PROJECT_ROOT" in os.environ:
        env_settings["project_root"] = os.environ["FURU_PROJECT_ROOT"]

    return env_settings


def _normalize_settings(settings: FuruSettings) -> FuruSettings:
    project_root = settings.project_root
    if project_root is None:
        project_root = _discover_project_root()
    else:
        project_root = project_root.resolve()

    data_dir = settings.data_dir
    if not data_dir.is_absolute():
        base_dir = project_root or Path.cwd().resolve()
        data_dir = (base_dir / data_dir).resolve()
    else:
        data_dir = data_dir.resolve()

    return settings.model_copy(
        update={
            "project_root": project_root,
            "data_dir": data_dir,
        }
    )


def load_settings(*, start_path: Path | None = None) -> FuruSettings:
    project_root = _discover_project_root(start_path)
    pyproject_settings = _load_pyproject_settings(project_root)
    env_settings = _load_env_settings()
    return _normalize_settings(
        FuruSettings.model_validate(
            {
                "project_root": project_root,
                **pyproject_settings,
                **env_settings,
            }
        )
    )


settings = load_settings()


def _replace_settings(new_settings: FuruSettings) -> FuruSettings:
    for field_name in type(settings).model_fields:
        setattr(settings, field_name, getattr(new_settings, field_name))
    return settings


def configure(**overrides: object) -> FuruSettings:
    return _replace_settings(
        _normalize_settings(
            FuruSettings.model_validate(
                {
                    **settings.model_dump(),
                    **overrides,
                }
            )
        )
    )


def reset_settings_for_testing(*, start_path: Path | None = None) -> FuruSettings:
    return _replace_settings(load_settings(start_path=start_path))


class _FuruConfig:
    @property
    def debug_mode(self) -> bool:
        return settings.debug

    @debug_mode.setter
    def debug_mode(self, value: bool) -> None:
        configure(debug=value)

    @property
    def directories(self) -> _FuruDirectories:
        return _FuruDirectories(data=settings.data_dir)

    @directories.setter
    def directories(self, value: _FuruDirectories) -> None:
        configure(data_dir=value.data)


config = _FuruConfig()
