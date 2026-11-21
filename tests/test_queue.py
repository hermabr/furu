"""The manager's queue, fed plan lines written by hand: no specs, sockets or threads.

Scripts and workers are plain names. A spec id is ``tests.<class>:c:<name>``,
so a spec's run is named ``<name>``.
"""

import logging
from collections.abc import Sequence
from pathlib import Path

import pytest

import furu.manager.queue as queue_module
from furu.config import get_config
from furu.constants import KINDMARKER, VALUEMARKER
from furu.locking import lock
from furu.manager.queue import Queue
from furu.protocol import events, plan_text
from furu.protocol.layout import attempt_dir_in, lock_path_in, run_name
from furu.protocol.messages import (
    Batch,
    CodeLocation,
    GaveUpMessage,
    Group,
    JobCompletedResult,
    JobFailedResult,
    JobResult,
    JobThrottle,
    Launch,
    ProcessSettings,
    RerunMessage,
    ResultMessage,
    SubmitMessage,
    WarningMessage,
)

CPU = "cpu"
CLASS_DIR = Path("/nowhere/tests.Leaf")
DONE = JobCompletedResult()
FAILED = JobFailedResult(error="boom")


def _specs(names: str, cls: str = "Leaf") -> list[str]:
    return [f"tests.{cls}:c:{name}" for name in names.split()]


def _group(
    specs: Sequence[str],
    *,
    batch: str = "Leaf",
    cap: int = 1,
    pools: dict[str, Batch] | None = None,
    throttle: int | None = None,
    class_dir: Path = CLASS_DIR,
) -> Group:
    """``specs`` on the cpu pool, each alone unless ``cap`` lets them share a call."""
    return Group(
        specs=list(specs),
        class_dir=class_dir,
        process=ProcessSettings(
            environment={}, required_environment_variables=(), reuse="never"
        ),
        pools=pools or {CPU: Batch(batch=batch, cap=cap)},
        throttle=None
        if throttle is None
        else JobThrottle(key=batch, max_running=throttle),
    )


def _plan(name: str) -> Path:
    return get_config().directories.anchored().plans / f"{name}.plan"


def _submit(
    queue: Queue[str, str],
    script: str,
    *groups: Group,
    plan: str = "e1",
    holds: dict[str, list[str]] | None = None,
) -> None:
    """Send ``groups`` as ``script``'s launch; ``holds`` says which specs a spec's fields hold."""
    holds = holds or {}
    lines = [
        plan_text.dump(
            {
                "spec": spec,
                "fields": {
                    f"f{i}": {KINDMARKER: "spec", VALUEMARKER: held}
                    for i, held in enumerate(holds.get(spec, []))
                },
            }
        )
        for group in groups
        for spec in group.specs
    ]
    nowhere = Path("/nowhere")
    launch = Launch(
        plan=_plan(plan),
        execution_log=nowhere / "execution.log",
        config=get_config(),
        code=CodeLocation(nowhere, nowhere, nowhere, nowhere, nowhere),
    )
    queue.submit(
        script, SubmitMessage(pools={}, launch=launch, lines=lines, groups=list(groups))
    )


def _out(queue: Queue[str, str], worker: str, pool: str = CPU) -> list[str]:
    """The specs of the job ``worker`` is handed; none if there is none for it."""
    job = queue.hand_out(worker, pool)
    return [] if job is None else job.specs


def _told(queue: Queue[str, str]) -> list[tuple[str, object]]:
    """What the queue has to tell scripts, emptied, with every job taking no time."""
    outbox, queue.outbox = queue.outbox, []
    return [
        (
            script,
            message.model_copy(update={"seconds": 0.0})
            if isinstance(message, ResultMessage)
            else message,
        )
        for script, message in outbox
    ]


def _result(specs: Sequence[str], result: JobResult) -> ResultMessage:
    return ResultMessage(object_ids=list(specs), result=result, seconds=0.0)


def _gave_up(specs: Sequence[str]) -> GaveUpMessage:
    return GaveUpMessage(
        specs=list(specs),
        error="boom",
        log=Path("/nowhere/log"),
        attempts=dict.fromkeys(specs, 1),
    )


def test_demand_counts_busy_workers_and_jobs_that_could_start_up_to_max_workers() -> (
    None
):
    queue: Queue[str, str] = Queue()
    _submit(queue, "s", _group(_specs("a b c")))
    assert _out(queue, "w1") != []

    assert [queue.demand(CPU, n) for n in (10, 2, 0)] == [3, 2, 0]
    assert queue.demand("gpu", 10) == 0


def test_throttle_limits_leases_and_demand() -> None:
    queue: Queue[str, str] = Queue()
    limited, free = _specs("l1 l2 l3", "Limited"), _specs("f")
    _submit(queue, "s", _group(limited, batch="Limited", throttle=2), _group(free))

    assert queue.demand(CPU, 10) == 3
    assert [_out(queue, w) for w in ("w1", "w2", "w3")] == [
        limited[:1],
        limited[1:2],
        free,
    ]
    # Busy workers still count; none more can start.
    assert queue.demand(CPU, 10) == 3
    assert _out(queue, "w4") == []

    queue.finish("w1", DONE)
    assert _out(queue, "w1") == limited[2:]


def test_specs_with_one_batch_key_are_one_job_up_to_the_cap() -> None:
    queue: Queue[str, str] = Queue()
    xs, ys = _specs("x1 x2 x3"), _specs("y1")
    _submit(queue, "s", _group(xs, batch="x", cap=2), _group(ys, batch="y", cap=2))

    assert queue.demand(CPU, 10) == 3
    assert [_out(queue, w) for w in ("w1", "w2", "w3", "w4")] == [
        xs[:2],
        xs[2:],
        ys,
        [],
    ]


def test_a_batch_holds_specs_from_one_plan_and_a_throttle_counts_batches(
    caplog: pytest.LogCaptureFixture,
) -> None:
    queue: Queue[str, str] = Queue()
    for index, names in enumerate(("a b", "c d", "e")):
        _submit(
            queue,
            f"s{index}",
            _group(_specs(names), batch="T", cap=3, throttle=2),
            plan=f"e{index}",
        )

    with caplog.at_level(logging.WARNING, logger="furu"):
        first, second = queue.hand_out("w1", CPU), queue.hand_out("w2", CPU)

    assert first is not None and second is not None
    assert len(first.specs) == len(second.specs) == 2
    assert [first.plan, second.plan] == [_plan("e0"), _plan("e1")]
    # A worker could take three. It is said once for each pair of plans, to
    # the script and in the manager's own log.
    told = _told(queue)
    assert [script for script, _ in told] == ["s0", "s0", "s1"]
    reasons = [m.reason for _, m in told if isinstance(m, WarningMessage)]
    assert len(reasons) == 3 and all("smaller batch" in reason for reason in reasons)
    assert caplog.messages == reasons
    # The throttle counts the two running batches across all scripts.
    assert queue.demand(CPU, 10) == 2
    assert _out(queue, "w3") == []
    queue.finish("w1", DONE)
    assert _out(queue, "w1") == _specs("e")


def test_a_spec_that_two_launches_ask_for_is_queued_once_under_the_first() -> None:
    queue: Queue[str, str] = Queue()
    leaf, other = _specs("leaf other")
    (parent,) = _specs("parent", "Parent")
    _submit(queue, "s1", _group([leaf, other]))
    _submit(
        queue,
        "s2",
        _group([leaf, other]),
        _group([parent], batch="Parent"),
        plan="e2",
        holds={parent: [leaf]},
    )

    job = queue.hand_out("w1", CPU)
    assert job is not None and job.specs == [leaf] and job.plan == _plan("e1")
    queue.finish("w1", DONE)
    assert _told(queue) == [
        ("s1", _result([leaf], DONE)),
        ("s2", _result([leaf], DONE)),
    ]

    # The first launch leaves: what only it asked for goes with it.
    queue.leave("s1")
    assert sorted([_out(queue, "w1"), _out(queue, "w2")]) == [[other], [parent]]
    # Nobody asks for the two that run any more. They finish all the same.
    queue.leave("s2")
    assert not queue.placed and sorted(queue.unwanted()) == ["w1", "w2"]
    queue.finish("w1", DONE)
    queue.finish("w2", DONE)
    assert not queue.dag.nodes and _told(queue) == []


def test_an_attempt_that_did_not_complete_waits_for_its_script_to_say_what_happens() -> (
    None
):
    queue: Queue[str, str] = Queue()
    (leaf,), (parent,) = _specs("leaf"), _specs("parent", "Parent")
    _submit(
        queue,
        "s",
        _group([leaf]),
        _group([parent], batch="Parent"),
        holds={parent: [leaf]},
    )

    assert _out(queue, "w1") == [leaf]
    queue.finish("w1", FAILED)
    assert _told(queue) == [("s", _result([leaf], FAILED))]
    assert _out(queue, "w1") == []
    queue.rerun("s", RerunMessage(specs=[leaf]))
    assert _out(queue, "w1") == [leaf]
    queue.finish("w1", FAILED)
    queue.gave_up("s", _gave_up([leaf]))
    assert _told(queue)[1:] == [
        ("s", _gave_up([leaf]).model_copy(update={"blocked": [parent]}))
    ]
    assert _out(queue, "w1") == [] and queue.demand(CPU, 10) == 0

    # Another launch that asks for the spec gives it a fresh go. What waited
    # for it does not run for the script that was told it would not.
    _submit(queue, "s2", _group([leaf]), plan="e2")
    assert _out(queue, "w1") == [leaf]
    queue.finish("w1", DONE)
    assert _out(queue, "w1") == [] and not queue.placed


def test_each_script_answers_for_its_own_specs_of_a_batch_that_did_not_complete() -> (
    None
):
    queue: Queue[str, str] = Queue()
    specs = _specs("a b")
    _submit(queue, "first", _group(specs, batch="B", cap=10))
    for script, spec in zip(("s2", "s3"), specs, strict=True):
        _submit(queue, script, _group([spec], batch="B", cap=10), plan=script)
    queue.leave("first")

    assert _out(queue, "w1") == specs
    queue.finish("w1", FAILED)
    assert [script for script, _ in _told(queue)] == ["s2", "s3"]
    queue.rerun("s2", RerunMessage(specs=specs[:1]))
    assert _out(queue, "w1") == specs[:1]
    # The other spec waits for the script that answers for it.
    assert _out(queue, "w2") == []
    ((script, warning),) = _told(queue)
    assert script == "s2" and isinstance(warning, WarningMessage)
    queue.gave_up("s3", _gave_up(specs[1:]))
    assert _told(queue) == [("s3", _gave_up(specs[1:]))]


# --- stopped batches --------------------------------------------------------------

TINY, SMALL, LARGE = "tiny", "small", "large"
SWEEP = {
    pool: Batch(batch=f"sweep:{cap}", cap=cap)
    for pool, cap in ((TINY, 2), (SMALL, 5), (LARGE, 20))
}


def _stopped_sweep(count: int) -> tuple[Queue[str, str], list[str]]:
    """A launch of ``count`` specs whose first 5 were a batch that stopped."""
    queue: Queue[str, str] = Queue()
    specs = _specs(" ".join(str(n) for n in range(1, count + 1)))
    _submit(queue, "s", _group(specs, pools=SWEEP))
    assert _out(queue, "w0", SMALL) == specs[:5]
    queue.finish("w0", JobFailedResult(error="worker w0 disconnected"))
    queue.rerun("s", RerunMessage(specs=specs[:5]))
    _told(queue)
    return queue, specs


@pytest.mark.parametrize(
    ("count", "pool", "handed", "warns", "then"),
    [
        # 25 are unstarted: a worker of 20 gets 6 to 25, and the stopped batch
        # stays whole.
        (30, LARGE, range(6, 26), False, range(26, 31)),
        # Only 3 are: it gets those and the stopped 5 as one batch of 8.
        (8, LARGE, [6, 7, 8, 1, 2, 3, 4, 5], True, []),
        # 18 are: it gets those and specs 1 and 2, and 3 to 5 are unstarted.
        (23, LARGE, [*range(6, 24), 1, 2], True, [3, 4, 5]),
        # Alone it goes out as it was, and continues in its folder.
        (5, LARGE, range(1, 6), False, []),
        # Too large for the worker that asks, it is cut.
        (5, TINY, [1, 2], True, [3, 4, 5]),
    ],
)
def test_a_worker_is_filled_with_unstarted_specs_then_stopped_batches(
    count: int, pool: str, handed: Sequence[int], warns: bool, then: Sequence[int]
) -> None:
    queue, specs = _stopped_sweep(count)

    assert _out(queue, "w1", pool) == [specs[n - 1] for n in handed]

    old_folder = attempt_dir_in(CLASS_DIR, run_name([str(n) for n in range(1, 6)]))
    warnings = _told(queue)
    assert len(warnings) == warns
    for _, warning in warnings:
        assert isinstance(warning, WarningMessage)
        assert warning.reason.endswith(f"stays in attempts/: {old_folder}")
    # What a worker of 5 gets next: unstarted specs.
    assert _out(queue, "w2", SMALL) == [specs[n - 1] for n in then]
    assert _told(queue) == []


def test_a_new_queue_takes_up_the_working_folders_on_disk_once_their_lock_is_free(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(queue_module, "CHECK_INTERVAL_S", 0.0)
    specs = _specs("0 1 2 3")
    # A batch that a worker of another manager opened, in its order.
    started = [specs[2], specs[0]]
    run = run_name(["2", "0"])
    folder = attempt_dir_in(tmp_path, run)
    folder.mkdir(parents=True)
    events.start(folder, ["2", "0"], _plan("e1"), _plan("e1"))

    queue: Queue[str, str] = Queue()
    with lock(lock_path_in(tmp_path, run)):
        _submit(queue, "s", _group(specs, pools=SWEEP, class_dir=tmp_path))
        assert _out(queue, "w1", SMALL) == [specs[1], specs[3]]
        # Its worker is alive: nothing to hand out.
        assert _out(queue, "w2", SMALL) == []
    assert _out(queue, "w2", SMALL) == started
    assert _told(queue) == []
