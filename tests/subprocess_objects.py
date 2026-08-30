"""Spec classes for subprocess-execution tests.

These live in their own module (instead of the test file) so the worker's
child process can import them by fully qualified name; the tests put this
directory on the child's PYTHONPATH.
"""

from __future__ import annotations

import os
import signal
import sys
from pathlib import Path
from typing import Literal

from furu import Metadata, Spec, batched

type Reuse = Literal["never", "same_environment", "same_environment_same_spec"]


def _pid_and_variable(name: str) -> str:
    return f"{os.getpid()}:{os.environ.get(name)}"


class SubprocessEnvLeaf(Spec[str]):
    variable_name: str
    variable_value: str | None
    reuse: Reuse = "same_environment"
    required_environment_variables: tuple[str, ...] = ()
    marker: int = 0

    def metadata(self) -> Metadata:
        return Metadata(
            environment={self.variable_name: self.variable_value},
            reuse=self.reuse,
            required_environment_variables=self.required_environment_variables,
        )

    def create(self) -> str:
        return _pid_and_variable(self.variable_name)


class OtherSubprocessEnvLeaf(Spec[str]):
    variable_name: str
    variable_value: str | None
    reuse: Reuse = "same_environment"
    marker: int = 0

    def metadata(self) -> Metadata:
        return Metadata(
            environment={self.variable_name: self.variable_value},
            reuse=self.reuse,
        )

    def create(self) -> str:
        return _pid_and_variable(self.variable_name)


class SubprocessBatchLeaf(Spec[str]):
    value: int

    @batched(lambda _: (None, 8))
    def create(objs: list[SubprocessBatchLeaf]) -> list[str]:
        return [f"{os.getpid()}:{obj.value}" for obj in objs]


class SubprocessCwdLeaf(Spec[str]):
    marker: int = 0

    def create(self) -> str:
        return f"{os.getpid()}:{Path.cwd()}"


class SubprocessCrashLeaf(Spec[str]):
    marker: int = 0

    def create(self) -> str:
        print("crash-leaf about to die", file=sys.stderr, flush=True)
        os.kill(os.getpid(), signal.SIGKILL)
        return "unreachable"


class SubprocessDependencyLeaf(Spec[str]):
    marker: int = 0

    def create(self) -> str:
        return "dependency"


class SubprocessBlockedParent(Spec[str]):
    marker: int = 0

    def create(self) -> str:
        return SubprocessDependencyLeaf(marker=self.marker).create()
