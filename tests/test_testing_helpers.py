from pathlib import Path

import pytest

import furu
from furu.testing import furu_test_env, override_results


class _DependencyStub(furu.Furu[str]):
    def _create(self) -> str:
        raise RuntimeError("dependency should be overridden")

    def _load(self) -> str:
        raise RuntimeError("dependency should be overridden")


class _ParentPipeline(furu.Furu[str]):
    dependency: _DependencyStub = furu.chz.field(default_factory=_DependencyStub)

    def _create(self) -> str:
        return f"parent:{self.dependency.get()}"

    def _load(self) -> str:
        return "parent:loaded"


class _GrandparentPipeline(furu.Furu[str]):
    parent: _ParentPipeline = furu.chz.field(default_factory=_ParentPipeline)

    def _create(self) -> str:
        return f"grand:{self.parent.get()}"

    def _load(self) -> str:
        return "grand:loaded"


class _ListParentPipeline(furu.Furu[str]):
    deps: list[_DependencyStub] = furu.chz.field(
        default_factory=lambda: [_DependencyStub()],
    )

    def _create(self) -> str:
        return f"list:{self.deps[0].get()}"

    def _load(self) -> str:
        return "list:loaded"


def test_furu_test_env_sets_and_restores_config(tmp_path: Path) -> None:
    original_base_root = furu.FURU_CONFIG.base_root
    original_version_root = furu.FURU_CONFIG.version_controlled_root_override
    original_record_git = furu.FURU_CONFIG.record_git
    original_allow_no_git_origin = furu.FURU_CONFIG.allow_no_git_origin
    original_poll_interval = furu.FURU_CONFIG.poll_interval
    original_stale_timeout = furu.FURU_CONFIG.stale_timeout
    original_max_wait = furu.FURU_CONFIG.max_wait_time_sec
    original_lease_duration = furu.FURU_CONFIG.lease_duration_sec
    original_heartbeat = furu.FURU_CONFIG.heartbeat_interval_sec
    original_cancelled_is_preempted = furu.FURU_CONFIG.cancelled_is_preempted
    original_retry_failed = furu.FURU_CONFIG.retry_failed
    original_submitit_root = furu.FURU_CONFIG.submitit_root

    with furu_test_env(tmp_path) as root:
        assert root == tmp_path
        assert furu.FURU_CONFIG.base_root == tmp_path
        assert furu.FURU_CONFIG.version_controlled_root_override == (
            tmp_path / "furu-data" / "artifacts"
        )
        assert furu.FURU_CONFIG.record_git == "ignore"
        assert furu.FURU_CONFIG.allow_no_git_origin is False
        assert furu.FURU_CONFIG.poll_interval == 0.01
        assert furu.FURU_CONFIG.stale_timeout == 0.1
        assert furu.FURU_CONFIG.max_wait_time_sec is None
        assert furu.FURU_CONFIG.lease_duration_sec == 0.05
        assert furu.FURU_CONFIG.heartbeat_interval_sec == 0.01
        assert furu.FURU_CONFIG.cancelled_is_preempted is True
        assert furu.FURU_CONFIG.retry_failed is True
        assert furu.FURU_CONFIG.submitit_root == tmp_path / "submitit"
        assert furu.get_furu_root() == tmp_path / "data"
        assert furu.get_furu_root(version_controlled=True) == (
            tmp_path / "furu-data" / "artifacts"
        )
        assert furu.FURU_CONFIG.raw_dir == tmp_path / "raw"

    assert furu.FURU_CONFIG.base_root == original_base_root
    assert furu.FURU_CONFIG.version_controlled_root_override == original_version_root
    assert furu.FURU_CONFIG.record_git == original_record_git
    assert furu.FURU_CONFIG.allow_no_git_origin == original_allow_no_git_origin
    assert furu.FURU_CONFIG.poll_interval == original_poll_interval
    assert furu.FURU_CONFIG.stale_timeout == original_stale_timeout
    assert furu.FURU_CONFIG.max_wait_time_sec == original_max_wait
    assert furu.FURU_CONFIG.lease_duration_sec == original_lease_duration
    assert furu.FURU_CONFIG.heartbeat_interval_sec == original_heartbeat
    assert furu.FURU_CONFIG.cancelled_is_preempted == original_cancelled_is_preempted
    assert furu.FURU_CONFIG.retry_failed == original_retry_failed
    assert furu.FURU_CONFIG.submitit_root == original_submitit_root


def test_furu_tmp_root_fixture(furu_tmp_root: Path) -> None:
    assert furu.FURU_CONFIG.base_root == furu_tmp_root
    assert furu.get_furu_root() == furu_tmp_root / "data"


@pytest.mark.parametrize("use_hash_key", [False, True])
def test_override_results_skips_dependency_chain(
    furu_tmp_root: Path, use_hash_key: bool
) -> None:
    dependency = _DependencyStub()
    parent = _ParentPipeline(dependency=dependency)
    key = dependency.furu_hash if use_hash_key else dependency

    with override_results({key: "stubbed"}):
        assert dependency.exists() is True
        assert parent.get() == "parent:stubbed"

    with pytest.raises(RuntimeError, match="dependency should be overridden"):
        dependency.get()


def test_override_results_nested_path(furu_tmp_root: Path) -> None:
    grandparent = _GrandparentPipeline()

    with override_results({"parent.dependency": "stubbed"}, root=grandparent):
        assert grandparent.get() == "grand:parent:stubbed"


def test_override_results_list_index(furu_tmp_root: Path) -> None:
    root = _ListParentPipeline()

    with override_results({"deps.0": "stubbed"}, root=root):
        assert root.get() == "list:stubbed"
