"""Spec classes for subprocess-execution tests.

These live in their own module (instead of the test file) so the worker's
child process can import them by fully qualified name; the tests put this
directory on the child's PYTHONPATH. Storage is a spec field so parent and
child agree on directories without config plumbing.
"""

from __future__ import annotations

import os
import signal
import sys
from pathlib import Path
from typing import Literal, Self, TypeAlias

from furu import Metadata, Spec, Subprocess, batched

Reuse: TypeAlias = Literal["never", "same_environment", "same_environment_same_spec"]


def _pid_and_variable(name: str) -> str:
    return f"{os.getpid()}:{os.environ.get(name)}"


class SubprocessEnvLeaf(Spec[str]):
    storage_root: str
    variable_name: str
    variable_value: str | None
    reuse: Reuse = "same_environment"
    required_environment: tuple[str, ...] = ()
    marker: int = 0

    def metadata(self) -> Metadata:
        return Metadata(
            storage=Path(self.storage_root),
            execution=Subprocess(
                environment={self.variable_name: self.variable_value},
                reuse=self.reuse,
                required_environment=self.required_environment,
            ),
        )

    def create(self) -> str:
        return _pid_and_variable(self.variable_name)


class OtherSubprocessEnvLeaf(Spec[str]):
    storage_root: str
    variable_name: str
    variable_value: str | None
    reuse: Reuse = "same_environment"
    marker: int = 0

    def metadata(self) -> Metadata:
        return Metadata(
            storage=Path(self.storage_root),
            execution=Subprocess(
                environment={self.variable_name: self.variable_value},
                reuse=self.reuse,
            ),
        )

    def create(self) -> str:
        return _pid_and_variable(self.variable_name)


class SubprocessBatchLeaf(Spec[str]):
    storage_root: str
    value: int

    def metadata(self) -> Metadata:
        return Metadata(storage=Path(self.storage_root), execution=Subprocess())

    @batched(lambda _: (None, 8))
    def create(objs: list[Self]) -> list[str]:
        return [f"{os.getpid()}:{obj.value}" for obj in objs]


class SubprocessCrashLeaf(Spec[str]):
    storage_root: str
    marker: int = 0

    def metadata(self) -> Metadata:
        return Metadata(
            storage=Path(self.storage_root),
            execution=Subprocess(),
        )

    def create(self) -> str:
        print("crash-leaf about to die", file=sys.stderr, flush=True)
        os.kill(os.getpid(), signal.SIGKILL)
        return "unreachable"


class SubprocessDependencyLeaf(Spec[str]):
    storage_root: str
    marker: int = 0

    def create(self) -> str:
        return "dependency"


class SubprocessBlockedParent(Spec[str]):
    storage_root: str
    marker: int = 0

    def metadata(self) -> Metadata:
        return Metadata(
            storage=Path(self.storage_root),
            execution=Subprocess(),
        )

    def create(self) -> str:
        return SubprocessDependencyLeaf(
            storage_root=self.storage_root, marker=self.marker
        ).create()
