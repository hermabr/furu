import json

from contextvars import Token

import pytest

import furu
from furu.execution.context import EXEC_CONTEXT, ExecContext
from furu.execution.slurm_spec import SlurmSpec, slurm_spec_key


DEFAULT_EXECUTOR_KEY = slurm_spec_key(SlurmSpec())
GPU_EXECUTOR_KEY = slurm_spec_key(
    SlurmSpec(partition="gpu", gpus=1, cpus=8, mem_gb=64, time_min=720)
)


class BasicTask(furu.Furu[int]):
    _create_calls: int = 0
    _load_calls: int = 0

    def _create(self) -> int:
        object.__setattr__(self, "_create_calls", self._create_calls + 1)
        (self.furu_dir / "value.json").write_text(json.dumps(1))
        return 1

    def _load(self) -> int:
        object.__setattr__(self, "_load_calls", self._load_calls + 1)
        return json.loads((self.furu_dir / "value.json").read_text())


class GpuTask(BasicTask):
    def _executor(self) -> SlurmSpec:
        return SlurmSpec(partition="gpu", gpus=1, cpus=8, mem_gb=64, time_min=720)


class MultiSpecTask(BasicTask):
    def _executor(self) -> SlurmSpec | tuple[SlurmSpec, ...]:
        return (
            SlurmSpec(),
            SlurmSpec(partition="gpu", gpus=1, cpus=8, mem_gb=64, time_min=720),
        )


class AnotherTask(BasicTask):
    """A distinct class to ensure a different furu hash from BasicTask."""


class ValidatingTask(BasicTask):
    def _validate(self) -> bool:
        return (self.furu_dir / "value.json").is_file()


class FailingTask(furu.Furu[int]):
    def _create(self) -> int:
        raise RuntimeError("boom")

    def _load(self) -> int:
        return 1


def _set_executor_context(
    spec_key: str, *, current_node_hash: str | None = "root"
) -> Token[ExecContext]:
    return EXEC_CONTEXT.set(
        ExecContext(
            mode="executor",
            spec_key=spec_key,
            backend="local",
            current_node_hash=current_node_hash,
        )
    )


def test_executor_get_requires_artifact(furu_tmp_root) -> None:
    obj = BasicTask()
    token = _set_executor_context(DEFAULT_EXECUTOR_KEY)
    try:
        with pytest.raises(furu.FuruMissingArtifact):
            obj.get()
    finally:
        EXEC_CONTEXT.reset(token)


def test_executor_get_loads_existing_artifact(furu_tmp_root) -> None:
    obj = BasicTask()
    obj.get()

    token = _set_executor_context(DEFAULT_EXECUTOR_KEY)
    try:
        assert obj.get() == 1
    finally:
        EXEC_CONTEXT.reset(token)

    assert obj._create_calls == 1
    assert obj._load_calls == 1


def test_executor_force_requires_matching_spec(furu_tmp_root) -> None:
    obj = GpuTask()
    token = _set_executor_context(DEFAULT_EXECUTOR_KEY, current_node_hash=obj.furu_hash)
    try:
        with pytest.raises(furu.FuruSpecMismatch):
            obj.get(force=True)
    finally:
        EXEC_CONTEXT.reset(token)


def test_executor_force_allows_matching_spec(furu_tmp_root) -> None:
    obj = BasicTask()
    token = _set_executor_context(DEFAULT_EXECUTOR_KEY, current_node_hash=obj.furu_hash)
    try:
        assert obj.get(force=True) == 1
    finally:
        EXEC_CONTEXT.reset(token)

    assert obj._create_calls == 1


def test_executor_force_allows_secondary_executor_choice(furu_tmp_root) -> None:
    obj = MultiSpecTask()
    token = _set_executor_context(GPU_EXECUTOR_KEY, current_node_hash=obj.furu_hash)
    try:
        assert obj.get(force=True) == 1
    finally:
        EXEC_CONTEXT.reset(token)

    assert obj._create_calls == 1


def test_executor_force_always_rerun_recomputes(furu_tmp_root, monkeypatch) -> None:
    obj = BasicTask()
    obj.get()

    qualname = f"{BasicTask.__module__}.{BasicTask.__qualname__}"
    monkeypatch.setattr(furu.FURU_CONFIG, "always_rerun_all", False)
    monkeypatch.setattr(furu.FURU_CONFIG, "always_rerun", {qualname})

    token = _set_executor_context(DEFAULT_EXECUTOR_KEY, current_node_hash=obj.furu_hash)
    try:
        assert obj.get(force=True) == 1
    finally:
        EXEC_CONTEXT.reset(token)

    assert obj._create_calls == 2


def test_worker_entry_always_rerun_recomputes(furu_tmp_root, monkeypatch) -> None:
    obj = BasicTask()
    obj.get()

    qualname = f"{BasicTask.__module__}.{BasicTask.__qualname__}"
    monkeypatch.setattr(furu.FURU_CONFIG, "always_rerun_all", False)
    monkeypatch.setattr(furu.FURU_CONFIG, "always_rerun", {qualname})
    monkeypatch.setattr(
        BasicTask, "_setup_signal_handlers", lambda *args, **kwargs: None
    )

    obj._worker_entry()

    assert obj._create_calls == 2


def test_executor_force_invalidates_on_failed_validate(furu_tmp_root) -> None:
    obj = ValidatingTask()
    obj.get()
    (obj.furu_dir / "value.json").unlink()

    token = _set_executor_context(DEFAULT_EXECUTOR_KEY, current_node_hash=obj.furu_hash)
    try:
        assert obj.get(force=True) == 1
    finally:
        EXEC_CONTEXT.reset(token)

    assert obj._create_calls == 2


def test_executor_force_disallows_non_current_node(furu_tmp_root) -> None:
    root = BasicTask()
    dep = AnotherTask()

    token = _set_executor_context(
        DEFAULT_EXECUTOR_KEY, current_node_hash=root.furu_hash
    )
    try:
        with pytest.raises(furu.FuruExecutionError):
            dep.get(force=True)
    finally:
        EXEC_CONTEXT.reset(token)


def test_worker_entry_invalidates_on_failed_validate(
    furu_tmp_root, monkeypatch
) -> None:
    obj = ValidatingTask()
    obj.get()
    (obj.furu_dir / "value.json").unlink()

    # Avoid installing signal handlers inside tests.
    monkeypatch.setattr(
        ValidatingTask, "_setup_signal_handlers", lambda *args, **kwargs: None
    )

    obj._worker_entry()

    assert obj._create_calls == 2


def test_worker_entry_failed_state_raises(furu_tmp_root, monkeypatch) -> None:
    obj = FailingTask()
    with pytest.raises(RuntimeError, match="boom"):
        obj.get()

    monkeypatch.setattr(furu.FURU_CONFIG, "retry_failed", False)

    with pytest.raises(furu.FuruComputeError) as exc_info:
        obj._worker_entry()

    message = str(exc_info.value)
    assert "already failed" in message
    assert obj.furu_hash in message
    assert str(furu.StateManager.get_state_path(obj._base_furu_dir())) in message
