from __future__ import annotations

import argparse
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import furu
import furu.execution.local as local_executor
from furu.core import Furu
from furu.execution import run_local
from furu.execution.plan import DependencyPlan, build_plan
from my_project.benchmark_tasks import TinyLeaf, TinyMerge


@dataclass(frozen=True)
class GraphShape:
    leaf_nodes: int
    level1_nodes: int
    level2_nodes: int
    root_nodes: int

    @property
    def total_nodes(self) -> int:
        return self.leaf_nodes + self.level1_nodes + self.level2_nodes + self.root_nodes

    def validate(self) -> None:
        if self.leaf_nodes < 1:
            raise ValueError("leaf_nodes must be >= 1")
        if self.level1_nodes < 1:
            raise ValueError("level1_nodes must be >= 1")
        if self.level2_nodes < 1:
            raise ValueError("level2_nodes must be >= 1")
        if self.root_nodes < 1:
            raise ValueError("root_nodes must be >= 1")
        if self.level1_nodes > self.leaf_nodes:
            raise ValueError("level1_nodes cannot exceed leaf_nodes")
        if self.level2_nodes > self.level1_nodes:
            raise ValueError("level2_nodes cannot exceed level1_nodes")
        if self.root_nodes > self.level2_nodes:
            raise ValueError("root_nodes cannot exceed level2_nodes")


@dataclass(frozen=True)
class GraphBuildResult:
    roots: list[Furu[int]]
    leaves: int
    level1: int
    level2: int

    @property
    def total_nodes(self) -> int:
        return self.leaves + self.level1 + self.level2 + len(self.roots)


@dataclass
class PlannerProfile:
    calls: int = 0
    total_sec: float = 0.0
    max_sec: float = 0.0

    def record(self, elapsed_sec: float) -> None:
        self.calls += 1
        self.total_sec += elapsed_sec
        if elapsed_sec > self.max_sec:
            self.max_sec = elapsed_sec


def _partition_round_robin(
    nodes: list[Furu[int]],
    bucket_count: int,
) -> list[list[Furu[int]]]:
    buckets: list[list[Furu[int]]] = [[] for _ in range(bucket_count)]
    for index, node in enumerate(nodes):
        buckets[index % bucket_count].append(node)
    return buckets


def build_benchmark_graph(shape: GraphShape) -> GraphBuildResult:
    shape.validate()

    leaves = [TinyLeaf(node_id=index) for index in range(shape.leaf_nodes)]

    level1: list[Furu[int]] = []
    for index, deps in enumerate(_partition_round_robin(leaves, shape.level1_nodes)):
        level1.append(TinyMerge(node_id=100_000 + index, deps=deps))

    level2: list[Furu[int]] = []
    for index, deps in enumerate(_partition_round_robin(level1, shape.level2_nodes)):
        level2.append(TinyMerge(node_id=200_000 + index, deps=deps))

    roots: list[Furu[int]] = []
    for index, deps in enumerate(_partition_round_robin(level2, shape.root_nodes)):
        roots.append(TinyMerge(node_id=300_000 + index, deps=deps))

    return GraphBuildResult(
        roots=roots,
        leaves=len(leaves),
        level1=len(level1),
        level2=len(level2),
    )


def measure_plan_latency(
    roots: list[Furu[int]],
    *,
    repeats: int,
) -> list[float]:
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        _ = build_plan(roots)
        samples.append(time.perf_counter() - started)
    return samples


def run_local_with_plan_profile(
    roots: list[Furu[int]],
    *,
    max_workers: int,
    poll_interval_sec: float,
) -> tuple[float, PlannerProfile]:
    profile = PlannerProfile()
    original_build_plan = local_executor.build_plan

    def timed_build_plan(
        plan_roots: list[Furu],
        *,
        completed_hashes: set[str] | None = None,
    ) -> DependencyPlan:
        started = time.perf_counter()
        plan = original_build_plan(plan_roots, completed_hashes=completed_hashes)
        profile.record(time.perf_counter() - started)
        return plan

    started = time.perf_counter()
    with patch.object(local_executor, "build_plan", timed_build_plan):
        run_local(
            roots,
            max_workers=max_workers,
            window_size="bfs",
            poll_interval_sec=poll_interval_sec,
        )
    elapsed = time.perf_counter() - started
    return elapsed, profile


def _percentile(samples: list[float], quantile: float) -> float:
    if not samples:
        raise ValueError("samples must be non-empty")
    if quantile <= 0.0:
        return min(samples)
    if quantile >= 1.0:
        return max(samples)
    ordered = sorted(samples)
    scaled = quantile * (len(ordered) - 1)
    lower = int(scaled)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    fraction = scaled - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _format_ms(seconds: float) -> str:
    return f"{seconds * 1000.0:,.2f} ms"


def _print_plan_summary(label: str, samples: list[float]) -> None:
    print(label)
    print(f"  mean:   {_format_ms(statistics.mean(samples))}")
    print(f"  median: {_format_ms(statistics.median(samples))}")
    print(f"  p95:    {_format_ms(_percentile(samples, 0.95))}")
    print(f"  min:    {_format_ms(min(samples))}")
    print(f"  max:    {_format_ms(max(samples))}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark local executor planning and end-to-end runtime on "
            "a large nested DAG."
        )
    )
    parser.add_argument("--leaf-nodes", type=int, default=9_700)
    parser.add_argument("--level1-nodes", type=int, default=250)
    parser.add_argument("--level2-nodes", type=int, default=40)
    parser.add_argument("--root-nodes", type=int, default=10)
    parser.add_argument(
        "--plan-runs",
        type=int,
        default=7,
        help="How many build_plan() samples to record before/after execution.",
    )
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--poll-interval-sec", type=float, default=0.25)
    parser.add_argument(
        "--log-level",
        type=str,
        default="ERROR",
        help="Console log level for furu runtime logs during benchmark.",
    )
    parser.add_argument(
        "--furu-root",
        type=Path,
        default=None,
        help="Optional persistent root. Default uses a temporary directory.",
    )
    return parser.parse_args()


def run_benchmark(args: argparse.Namespace, root: Path) -> None:
    shape = GraphShape(
        leaf_nodes=args.leaf_nodes,
        level1_nodes=args.level1_nodes,
        level2_nodes=args.level2_nodes,
        root_nodes=args.root_nodes,
    )
    shape.validate()

    os.environ["FURU_LOG_LEVEL"] = args.log_level.upper()
    furu.set_furu_root(root)
    furu.FURU_CONFIG.record_git = "ignore"

    build_started = time.perf_counter()
    graph = build_benchmark_graph(shape)
    graph_build_sec = time.perf_counter() - build_started

    print("Local executor benchmark")
    print(f"furu root: {root}")
    print(
        "graph:",
        f"leaves={graph.leaves:,},",
        f"level1={graph.level1:,},",
        f"level2={graph.level2:,},",
        f"roots={len(graph.roots):,},",
        f"total={graph.total_nodes:,}",
    )
    print(
        "run config:",
        f"max_workers={args.max_workers},",
        f"poll_interval_sec={args.poll_interval_sec},",
        f"log_level={args.log_level.upper()}",
    )
    print(f"graph construction: {_format_ms(graph_build_sec)}")

    pre_plan_samples = measure_plan_latency(graph.roots, repeats=args.plan_runs)
    _print_plan_summary(
        f"plan latency before execution ({args.plan_runs} samples):",
        pre_plan_samples,
    )

    cold_run_sec, cold_profile = run_local_with_plan_profile(
        graph.roots,
        max_workers=args.max_workers,
        poll_interval_sec=args.poll_interval_sec,
    )

    planner_share_cold = (
        100.0 * cold_profile.total_sec / cold_run_sec if cold_run_sec > 0 else 0.0
    )
    non_planner_cold = max(0.0, cold_run_sec - cold_profile.total_sec)
    print("run_local cold run:")
    print(f"  elapsed: {_format_ms(cold_run_sec)}")
    print(
        f"  planner: {_format_ms(cold_profile.total_sec)} "
        f"across {cold_profile.calls} build_plan() calls "
        f"({planner_share_cold:.1f}% of run)"
    )
    print(f"  non-planner runtime: {_format_ms(non_planner_cold)}")

    post_plan_samples = measure_plan_latency(graph.roots, repeats=args.plan_runs)
    _print_plan_summary(
        f"plan latency after execution ({args.plan_runs} samples):",
        post_plan_samples,
    )

    warm_run_sec, warm_profile = run_local_with_plan_profile(
        graph.roots,
        max_workers=args.max_workers,
        poll_interval_sec=args.poll_interval_sec,
    )

    planner_share_warm = (
        100.0 * warm_profile.total_sec / warm_run_sec if warm_run_sec > 0 else 0.0
    )
    non_planner_warm = max(0.0, warm_run_sec - warm_profile.total_sec)
    print("run_local warm run (everything cached):")
    print(f"  elapsed: {_format_ms(warm_run_sec)}")
    print(
        f"  planner: {_format_ms(warm_profile.total_sec)} "
        f"across {warm_profile.calls} build_plan() calls "
        f"({planner_share_warm:.1f}% of run)"
    )
    print(f"  non-planner runtime: {_format_ms(non_planner_warm)}")


def main() -> None:
    args = _parse_args()
    if args.furu_root is not None:
        run_benchmark(args, args.furu_root.resolve())
        return

    with TemporaryDirectory(prefix="furu-local-bench-") as temp_dir:
        run_benchmark(args, Path(temp_dir).resolve())


if __name__ == "__main__":
    main()
