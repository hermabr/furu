"""Instant stand-ins for the git, nvidia-smi and `uv lock --check` probes that
run on every computing create(). Tests marked real_probes keep the real ones."""

from pathlib import Path

from furu import provenance
from furu.execution import load_or_create


def _canned_git_identity(
    cls: type[provenance.GitIdentity], cwd: Path
) -> provenance.GitIdentity:
    return cls(
        commit="0" * 40,
        branch="main",
        remote=None,
        repo_root=str(cwd),
        dirty=False,
        diff_stats=None,
    )


CANNED_PROBES = {
    (provenance.GitIdentity, "capture"): classmethod(_canned_git_identity),
    (provenance, "_probe_accelerators"): lambda: (),
    (load_or_create, "_require_uv"): lambda: None,
}
