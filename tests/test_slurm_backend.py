from __future__ import annotations

import json
import os
import stat
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from furu.worker import cli
from furu.worker.backends.slurm import (
    SlurmResources,
    SlurmWorkerBackend,
    SlurmWorkerPool,
)


def test_worker_cli_calls_worker_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []

    def worker_loop(*, server_url: str, auth_token: str) -> None:
        calls.append((server_url, auth_token))

    monkeypatch.setattr(cli, "worker_loop", worker_loop)

    assert (
        cli.main(["--server-url", "http://manager.test", "--auth-token", "secret"]) == 0
    )

    assert calls == [("http://manager.test", "secret")]


def test_worker_cli_requires_connection_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []

    def worker_loop(*, server_url: str, auth_token: str) -> None:
        calls.append((server_url, auth_token))

    monkeypatch.setattr(cli, "worker_loop", worker_loop)

    with pytest.raises(SystemExit) as exc_info:
        cli.main([])

    assert exc_info.value.code == 2
    assert calls == []


def test_slurm_backend_submits_workers_with_required_sbatch_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record_file, _active_file = _install_fake_slurm(tmp_path, monkeypatch)
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    log_dir = tmp_path / "logs"

    backend = SlurmWorkerBackend(
        advertised_host="manager.cluster",
        n_workers=2,
        resources=SlurmResources(
            partition="debug",
            cpus_per_task=4,
            mem="8G",
            extra_sbatch_args=("--exclusive",),
        ),
        log_dir=log_dir,
        chdir=work_dir,
        python_executable="/venv/bin/python",
        poll_interval=0,
    )

    pool = backend.start_pool(
        server_url="http://127.0.0.1:1234",
        auth_token="secret-token",
    )

    assert pool.job_ids == ("100", "101")
    assert log_dir.is_dir()

    records = _read_records(record_file)
    sbatch_records = [record for record in records if record["executable"] == "sbatch"]
    assert len(sbatch_records) == 2

    for worker_index, record in enumerate(sbatch_records):
        argv = record["argv"]
        assert "--parsable" in argv
        assert f"--chdir={work_dir.resolve()}" in argv
        assert (
            f"--output={log_dir.resolve() / f'furu-worker-{worker_index}-%j.out'}"
            in argv
        )
        assert (
            f"--error={log_dir.resolve() / f'furu-worker-{worker_index}-%j.err'}"
            in argv
        )
        assert "--job-name=furu-worker" in argv
        assert not any(arg.startswith("--export") for arg in argv)
        assert "--partition=debug" in argv
        assert "--cpus-per-task=4" in argv
        assert "--mem=8G" in argv
        assert "--exclusive" in argv
        assert (
            "--wrap=/venv/bin/python -m furu.worker.cli "
            "--server-url http://manager.cluster:1234 --auth-token '<redacted>'"
        ) in argv
        assert not record["has_manager_environment"]

    assert "secret-token" not in record_file.read_text()


def test_slurm_worker_pool_health_tracks_squeue_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _record_file, active_file = _install_fake_slurm(tmp_path, monkeypatch)
    backend = SlurmWorkerBackend(
        advertised_host="127.0.0.1",
        n_workers=2,
        log_dir=tmp_path / "logs",
        chdir=tmp_path,
        poll_interval=0,
    )
    pool = backend.start_pool(
        server_url="http://127.0.0.1:1234",
        auth_token="secret-token",
    )

    assert pool.is_healthy()

    active_file.write_text("100\n")

    assert not pool.is_healthy()


def test_slurm_worker_pool_join_cancels_jobs_left_after_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record_file, active_file = _install_fake_slurm(tmp_path, monkeypatch)
    active_file.write_text("100\n101\n")
    pool = SlurmWorkerPool(job_ids=("100", "101"), poll_interval=0)

    pool.join(timeout=0)

    assert active_file.read_text() == ""
    records = _read_records(record_file)
    assert records[-1] == {"executable": "scancel", "argv": ["100", "101"]}


def test_slurm_backend_requires_advertised_host() -> None:
    with pytest.raises(TypeError):
        SlurmWorkerBackend()  # ty: ignore[missing-argument]

    assert SlurmWorkerBackend(advertised_host="manager.cluster").advertised_host == (
        "manager.cluster"
    )


def _install_fake_slurm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    record_file = tmp_path / "slurm-records.jsonl"
    active_file = tmp_path / "active-jobs.txt"
    counter_file = tmp_path / "next-job-id.txt"
    active_file.write_text("")
    counter_file.write_text("100")

    monkeypatch.delenv("FURU_MANAGER_SERVER_URL", raising=False)
    monkeypatch.delenv("FURU_MANAGER_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("FURU_FAKE_SLURM_RECORD_FILE", str(record_file))
    monkeypatch.setenv("FURU_FAKE_SLURM_ACTIVE_FILE", str(active_file))
    monkeypatch.setenv("FURU_FAKE_SLURM_COUNTER_FILE", str(counter_file))
    monkeypatch.setenv(
        "PATH",
        f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
    )

    _write_executable(
        bin_dir / "sbatch",
        """
        import json
        import os
        import shlex
        import sys

        record_file = os.environ["FURU_FAKE_SLURM_RECORD_FILE"]
        active_file = os.environ["FURU_FAKE_SLURM_ACTIVE_FILE"]
        counter_file = os.environ["FURU_FAKE_SLURM_COUNTER_FILE"]

        def redact_auth_token(argv):
            redacted = []
            for arg in argv:
                if not arg.startswith("--wrap="):
                    redacted.append(arg)
                    continue

                command = arg.removeprefix("--wrap=")
                command_parts = shlex.split(command)
                for index, value in enumerate(command_parts[:-1]):
                    if value == "--auth-token":
                        command_parts[index + 1] = "<redacted>"
                redacted.append("--wrap=" + shlex.join(command_parts))
            return redacted

        with open(record_file, "a", encoding="utf-8") as file:
            file.write(
                json.dumps(
                    {
                        "executable": "sbatch",
                        "argv": redact_auth_token(sys.argv[1:]),
                        "has_manager_environment": (
                            "FURU_MANAGER_SERVER_URL" in os.environ
                            or "FURU_MANAGER_AUTH_TOKEN" in os.environ
                        ),
                    }
                )
                + "\\n"
            )

        with open(counter_file, encoding="utf-8") as file:
            job_id = int(file.read())
        with open(counter_file, "w", encoding="utf-8") as file:
            file.write(str(job_id + 1))
        with open(active_file, "a", encoding="utf-8") as file:
            file.write(f"{job_id}\\n")

        print(f"{job_id};cluster")
        """,
    )
    _write_executable(
        bin_dir / "squeue",
        """
        import json
        import os
        import sys

        record_file = os.environ["FURU_FAKE_SLURM_RECORD_FILE"]
        active_file = os.environ["FURU_FAKE_SLURM_ACTIVE_FILE"]

        with open(record_file, "a", encoding="utf-8") as file:
            file.write(json.dumps({"executable": "squeue", "argv": sys.argv[1:]}) + "\\n")

        requested_jobs = set()
        for index, arg in enumerate(sys.argv[1:]):
            if arg == "--jobs":
                requested_jobs.update(sys.argv[index + 2].split(","))
            elif arg.startswith("--jobs="):
                requested_jobs.update(arg.removeprefix("--jobs=").split(","))

        with open(active_file, encoding="utf-8") as file:
            active_jobs = set(file.read().split())

        for job_id in sorted(requested_jobs & active_jobs):
            print(job_id)
        """,
    )
    _write_executable(
        bin_dir / "scancel",
        """
        import json
        import os
        import sys

        record_file = os.environ["FURU_FAKE_SLURM_RECORD_FILE"]
        active_file = os.environ["FURU_FAKE_SLURM_ACTIVE_FILE"]

        with open(record_file, "a", encoding="utf-8") as file:
            file.write(json.dumps({"executable": "scancel", "argv": sys.argv[1:]}) + "\\n")

        cancelled_jobs = set(sys.argv[1:])
        with open(active_file, encoding="utf-8") as file:
            active_jobs = set(file.read().split())
        active_jobs.difference_update(cancelled_jobs)
        with open(active_file, "w", encoding="utf-8") as file:
            file.write("".join(f"{job_id}\\n" for job_id in sorted(active_jobs)))
        """,
    )

    return record_file, active_file


def _write_executable(path: Path, source: str) -> None:
    path.write_text(f"#!{sys.executable}\n{textwrap.dedent(source).lstrip()}")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _read_records(record_file: Path) -> list[dict[str, Any]]:
    if not record_file.exists():
        return []
    return [json.loads(line) for line in record_file.read_text().splitlines()]
