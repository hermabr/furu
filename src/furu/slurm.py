"""Running Slurm's commands: whether a job is gone, and the steps of a job."""

from __future__ import annotations

import os
import subprocess
import time
from contextlib import suppress
from pathlib import Path

# The states of a job whose processes Slurm saw end. Not NODE_FAIL or
# BOOT_FAIL: a node that stopped answering can still be running them.
_GONE = frozenset(
    {
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "REVOKED",
        "TIMEOUT",
    }
)

# A job in the queue: its id, its state and how many times it was started again.
_SQUEUE = (
    *("squeue", "--all", "--array", "--noheader"),
    *("-O", "JobID:20,State:20,RestartCnt:10"),
)

# How long a killed step may take to leave the queue, and how often it is
# asked: Slurm gives a step KillWait, 30 s by default, to end.
_STEP_END_TIMEOUT_S = 120.0
_STEP_POLL_S = 0.5

# The cluster the controller named, once it has: asked at most once a process.
_cluster: str | None = None


def run(*argv: str) -> str:
    """Run a Slurm command and return its output.

    Raises CalledProcessError if it fails, TimeoutExpired if it hangs and
    FileNotFoundError on a machine without Slurm. The variables that change
    what squeue and sacct print, or which cluster a command asks, are not
    passed on: the output is read by code.
    """
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("SQUEUE_", "SACCT_")) and name != "SLURM_CLUSTERS"
    }
    return subprocess.run(
        argv,
        check=True,
        capture_output=True,
        text=True,
        timeout=60.0,
        env=environment,
    ).stdout


def cluster() -> str | None:
    """The Slurm cluster this machine belongs to, or None without Slurm.

    A job has it in its environment; any other machine asks the controller.
    """
    global _cluster
    if name := os.environ.get("SLURM_CLUSTER_NAME"):
        return name
    if _cluster is None:
        try:
            config = run("scontrol", "show", "config")
        except FileNotFoundError:
            return None
        for line in config.splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "ClusterName" and value.strip():
                _cluster = value.strip()
                break
        else:
            raise ValueError("scontrol show config did not report ClusterName")
    return _cluster


def gone(cluster_name: str, job: str, restart: int) -> bool:
    """Whether Slurm says that run ``restart`` of ``job`` ended and left no
    process behind.

    True when the queue no longer runs the job and accounting has a clean end
    for it. Also when the queue has started the job again since, as it does
    with a job it preempted, and accounting has a clean end for that run: a
    job that runs again has ``SLURM_RESTART_COUNT`` one higher. Raises if a
    Slurm command fails, times out, or prints unexpected output.
    """
    if cluster_name != cluster():
        return False
    # Asked by id, the queue also lists a job that ended, for as long as the
    # controller remembers it. Not -O JobIDRaw, which Slurm 25.11.5 refuses.
    try:
        queued = run(*_SQUEUE, "-j", job)
    except subprocess.CalledProcessError as exc:
        # How the queue answers for a job it no longer remembers.
        if (exc.stdout or "").strip() or "Invalid job id" not in (exc.stderr or ""):
            raise
        queued = ""
    again = False
    for line in queued.splitlines():
        match line.split():
            case [job_id, state, restarts] if restarts.isdigit():
                # The other tasks of its array are listed with it.
                if job_id != job or state in _GONE:
                    continue
                # COMPLETING is a job whose processes are still being killed.
                if state not in {"PENDING", "RUNNING"} or int(restarts) <= restart:
                    return False
                again = True
            case _:
                raise ValueError(f"unexpected squeue output: {line!r}")
    # -D: a requeued job has a row for each time it ran.
    accounted = run(
        *("sacct", "-D", "-X", "--noheader", "--parsable2"),
        *("-o", "JobIDRaw,State,Reason", "-j", job),
    )
    states = []
    for line in accounted.splitlines():
        match line.split("|", 2):
            case [raw_id, state, reason]:
                if raw_id != job:
                    continue
                # Accounting ended the job while a process could not be killed.
                if "kill task failed" in reason.lower():
                    return False
                states.append(state.partition(" ")[0])  # "CANCELLED by <uid>"
            case _:
                raise ValueError(f"unexpected sacct output: {line!r}")
    clean = _GONE | {"REQUEUED"}
    if again:
        # A row for each run that ended, the first run's first, and one for
        # the run the queue has now. None may be of a node that stopped
        # answering, whichever run that was.
        return restart < sum(state in clean for state in states) and all(
            state in clean | {"PENDING", "RUNNING"} for state in states
        )
    # A row that says the job was requeued is no end: accounting can have it
    # before the row of the run that followed. Without any row it is behind
    # the queue.
    return all(state in clean for state in states) and any(
        state in _GONE for state in states
    )


def step(name: str, *, taken: int, slots: int) -> list[str]:
    """The start of the command that runs a program as a step of this job,
    named ``name``, on ``taken`` of the ``slots`` even parts of the job.

    Slurm keeps the step to its part: it sees only its GPUs, runs only on its
    CPUs and is killed if it uses more than its memory, as is everything it
    starts, in a session of its own or not. A step that asks for what other
    steps hold waits for them to end.
    """
    cpus = int(os.environ["SLURM_CPUS_ON_NODE"])
    gpus = int(os.environ.get("SLURM_GPUS_ON_NODE") or 0)
    # Slurm gives a step whole cores where it gives a job whole cores, so a
    # step that asked for one of a core's two CPUs would take the other too,
    # and the steps of a full job would wait for each other.
    threads = _threads_per_core()
    if not (cores := taken * (cpus // threads) // slots):
        raise RuntimeError(
            f"this Slurm job has {cpus // threads} cores for {slots} slots, and "
            f"Slurm gives a job whole cores of {threads} CPUs: ask for at least "
            f"{slots * threads} CPUs"
        )
    argv = ["srun", "--exact", "--ntasks=1", f"--cpus-per-task={cores * threads}"]
    # A step that does not say takes every GPU of the job, and its memory.
    if gpus:
        argv.append(f"--gpus={taken * gpus // slots}")
    # In MiB. A job that asked for memory by the CPU or GPU has what all of
    # them come to; a job has the first of these that is set.
    for per, count in (("CPU", cpus), ("GPU", gpus), ("NODE", 1)):
        if count and (memory := os.environ.get(f"SLURM_MEM_PER_{per}")):
            argv.append(f"--mem={taken * int(memory) * count // slots}M")
            break
    # --export: the step has this process's environment even where the job
    # was submitted with --export=NIL, which a step would otherwise follow.
    return [*argv, "--export=ALL", f"--job-name={name}"]


def _threads_per_core() -> int:
    """How many of the CPUs this process may run on share a core."""
    cpus = os.sched_getaffinity(0)
    siblings = "/sys/devices/system/cpu/cpu{}/topology/thread_siblings_list"
    return len(cpus) // len({Path(siblings.format(cpu)).read_text() for cpu in cpus})


def end_step(name: str) -> None:
    """Kill the steps of this job named ``name``, and return once Slurm lists
    none: every process of a step is dead by then, whatever started it.

    Raises if a Slurm command fails or the steps outlive
    ``_STEP_END_TIMEOUT_S``.
    """
    deadline = time.monotonic() + _STEP_END_TIMEOUT_S
    while True:
        listed = run(
            *("squeue", "--steps", "--noheader", "--format=%i %j"),
            *("-j", os.environ["SLURM_JOB_ID"]),
        )
        steps = [
            line.split()[0]
            for line in listed.splitlines()
            if line.split()[1:] == [name]
        ]
        if not steps:
            return
        if time.monotonic() > deadline:
            raise TimeoutError(f"Slurm still lists the killed steps {steps}")
        with suppress(subprocess.CalledProcessError):  # e.g. it just ended
            run("scancel", "--signal=KILL", *steps)
        time.sleep(_STEP_POLL_S)
