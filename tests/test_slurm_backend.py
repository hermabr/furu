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


def test_worker_cli_reads_auth_token_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []
    token_file = tmp_path / "worker.token"
    token_file.write_text("secret\n\n")

    def worker_loop(*, server_url: str, auth_token: str) -> None:
        calls.append((server_url, auth_token))

    monkeypatch.setattr(cli, "worker_loop", worker_loop)

    assert (
        cli.main(
            [
                "--server-url",
                "http://manager.test",
                "--auth-token-file",
                str(token_file),
            ]
        )
        == 0
    )

    assert calls == [("http://manager.test", "secret")]
    assert token_file.exists()


def test_worker_cli_requires_auth_token_file(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []

    def worker_loop(*, server_url: str, auth_token: str) -> None:
        calls.append((server_url, auth_token))

    monkeypatch.setattr(cli, "worker_loop", worker_loop)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--server-url", "http://manager.test"])

    assert exc_info.value.code == 2
    assert calls == []


def test_worker_cli_rejects_auth_token_argument(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []
    token_file = tmp_path / "worker.token"
    token_file.write_text("secret")

    def worker_loop(*, server_url: str, auth_token: str) -> None:
        calls.append((server_url, auth_token))

    monkeypatch.setattr(cli, "worker_loop", worker_loop)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                "--server-url",
                "http://manager.test",
                "--auth-token-file",
                str(token_file),
                "--auth-token",
                "secret",
            ]
        )

    assert exc_info.value.code == 2
    assert calls == []


def test_slurm_backend_submits_workers_with_required_sbatch_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record_file, _active_file = _install_fake_slurm(tmp_path, monkeypatch)
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    executor_dir = tmp_path / "furu" / "executions" / "executor-1"
    worker_dir = executor_dir / "workers"
    log_dir = worker_dir / "logs"
    monkeypatch.chdir(work_dir)

    backend = SlurmWorkerBackend(
        n_workers=2,
        resources=SlurmResources(
            partition="debug",
            cpus_per_task=4,
            mem="8G",
            extra_sbatch_args=("--exclusive",),
        ),
        poll_interval=1.5,
    )

    pool = backend.start_pool(
        server_url="http://manager.cluster:1234",
        auth_token="secret-token",
        executor_dir=executor_dir,
    )

    assert pool.array_job_id == "100"
    assert pool.n_workers == 2
    assert pool.health_check_interval == 1.5
    assert log_dir.is_dir()

    records = _read_records(record_file)
    sbatch_records = [record for record in records if record["executable"] == "sbatch"]
    assert len(sbatch_records) == 1

    argv = sbatch_records[0]["argv"]
    assert "--parsable" in argv
    assert f"--chdir={work_dir.resolve()}" in argv
    assert f"--output={log_dir.resolve() / 'furu-worker-%A-%a.out'}" in argv
    assert f"--error={log_dir.resolve() / 'furu-worker-%A-%a.err'}" in argv
    assert "--job-name=furu-worker" in argv
    assert "--array=0-1" in argv
    assert "--export=NIL" in argv
    assert not any(arg.startswith("--wrap") for arg in argv)
    assert "--partition=debug" in argv
    assert "--cpus-per-task=4" in argv
    assert "--mem=8G" in argv
    assert "--exclusive" in argv
    assert "secret-token" not in " ".join(argv)

    script_path = Path(argv[-1])
    script = script_path.read_text()
    assert "--auth-token-file" in script
    assert "--auth-token " not in script
    assert "secret-token" not in script
    assert f"exec {sys.executable} -m furu.worker.cli" in script
    assert "--server-url http://manager.cluster:1234" in script

    assert not (worker_dir / "secrets").exists()
    token_files = sorted(worker_dir.glob("worker-*.token"))
    assert len(token_files) == 1
    for token_file in token_files:
        assert _mode(token_file) == 0o600
        assert token_file.read_text() == "secret-token"
        assert str(token_file) in script

    assert not sbatch_records[0]["has_manager_environment"]

    assert "secret-token" not in record_file.read_text()

    assert all(token_file.exists() for token_file in token_files)


def test_slurm_worker_pool_health_tracks_sacct_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record_file, active_file = _install_fake_slurm(tmp_path, monkeypatch)
    backend = SlurmWorkerBackend(
        n_workers=2,
        resources=SlurmResources(),
        poll_interval=0,
    )
    pool = backend.start_pool(
        server_url="http://127.0.0.1:1234",
        auth_token="secret-token",
        executor_dir=tmp_path / "executor",
    )

    assert pool.is_healthy()

    active_file.write_text("100_0\n100_1 FAILED\n")

    assert not pool.is_healthy()
    sacct_records = [
        record
        for record in _read_records(record_file)
        if record["executable"] == "sacct"
    ]
    assert sacct_records[-1]["argv"] == [
        "-o",
        "JobID,State,NodeList",
        "--parsable2",
        "-j",
        "100",
    ]


def test_slurm_backend_requires_explicit_executor_dir() -> None:
    backend = SlurmWorkerBackend(
        n_workers=1,
        resources=SlurmResources(),
    )

    with pytest.raises(TypeError, match="executor_dir"):
        backend.start_pool(
            server_url="http://127.0.0.1:1234",
            auth_token="secret-token",
        )  # ty: ignore[missing-argument]


def test_slurm_worker_pool_join_cancels_jobs_left_after_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record_file, active_file = _install_fake_slurm(tmp_path, monkeypatch)
    active_file.write_text("100_0\n100_1\n")
    pool = SlurmWorkerPool(array_job_id="100", n_workers=2, poll_interval=0)

    pool.join(timeout=0)

    assert active_file.read_text() == ""
    records = _read_records(record_file)
    assert records[-1] == {"executable": "scancel", "argv": ["100"]}


def test_slurm_backend_uses_default_poll_interval() -> None:
    with pytest.raises(TypeError):
        SlurmWorkerBackend()  # ty: ignore[missing-argument]

    backend = SlurmWorkerBackend(
        n_workers=1,
        resources=SlurmResources(),
    )

    assert backend.poll_interval == 10.0


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
        import sys

        record_file = os.environ["FURU_FAKE_SLURM_RECORD_FILE"]
        active_file = os.environ["FURU_FAKE_SLURM_ACTIVE_FILE"]
        counter_file = os.environ["FURU_FAKE_SLURM_COUNTER_FILE"]

        with open(record_file, "a", encoding="utf-8") as file:
            file.write(
                json.dumps(
                    {
                        "executable": "sbatch",
                        "argv": sys.argv[1:],
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
        array_arg = next(
            (arg.removeprefix("--array=") for arg in sys.argv[1:] if arg.startswith("--array=")),
            "0",
        )
        start_text, separator, end_text = array_arg.partition("-")
        start = int(start_text)
        end = int(end_text if separator else start_text)

        with open(active_file, "a", encoding="utf-8") as file:
            for task_id in range(start, end + 1):
                file.write(f"{job_id}_{task_id}\\n")

        print(f"{job_id};cluster")
        """,
    )
    _write_executable(
        bin_dir / "sacct",
        """
        import json
        import os
        import sys

        record_file = os.environ["FURU_FAKE_SLURM_RECORD_FILE"]
        active_file = os.environ["FURU_FAKE_SLURM_ACTIVE_FILE"]

        with open(record_file, "a", encoding="utf-8") as file:
            file.write(json.dumps({"executable": "sacct", "argv": sys.argv[1:]}) + "\\n")

        requested_jobs = set()
        for index, arg in enumerate(sys.argv[1:]):
            if arg == "-j":
                requested_jobs.update(sys.argv[index + 2].split(","))
            elif arg.startswith("-j="):
                requested_jobs.update(arg.removeprefix("-j=").split(","))

        print("JobID|State|NodeList")
        with open(active_file, encoding="utf-8") as file:
            active_jobs = file.read().splitlines()

        for active_job in sorted(active_jobs):
            job_id, _, state = active_job.partition(" ")
            if job_id.partition("_")[0] not in requested_jobs:
                continue
            print(f"{job_id}|{state or 'RUNNING'}|node-a")
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

        show_array_tasks = "--array" in sys.argv[1:]

        with open(active_file, encoding="utf-8") as file:
            active_jobs = set(file.read().split())

        for active_job in sorted(active_jobs):
            job_id, separator, task_id = active_job.partition("_")
            if job_id not in requested_jobs:
                continue
            if show_array_tasks and separator:
                print(f"{job_id} {task_id}")
            else:
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
        active_jobs = {
            active_job
            for active_job in active_jobs
            if active_job not in cancelled_jobs
            and active_job.partition("_")[0] not in cancelled_jobs
        }
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


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)
