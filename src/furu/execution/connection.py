from __future__ import annotations

import re
import shutil
import subprocess
import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, field
from queue import Empty, SimpleQueue
from typing import IO, Protocol


class ManagerConnection(Protocol):
    def connect(self, *, local_url: str) -> AbstractContextManager[str]: ...


@dataclass(frozen=True, slots=True)
class DirectManagerConnection:
    def connect(self, *, local_url: str) -> AbstractContextManager[str]:
        return nullcontext(local_url)


@dataclass(frozen=True, slots=True)
class CloudflareQuickTunnel:
    command: tuple[str, ...] = ("cloudflared",)
    startup_timeout: float = 30.0
    extra_args: tuple[str, ...] = field(default_factory=tuple)

    def connect(self, *, local_url: str) -> AbstractContextManager[str]:
        return _cloudflare_quick_tunnel(
            command=self.command,
            startup_timeout=self.startup_timeout,
            extra_args=self.extra_args,
            local_url=local_url,
        )


_TRYCLOUDFLARE_URL_RE = re.compile(r"https://[-a-zA-Z0-9.]+\.trycloudflare\.com")


@contextmanager
def _cloudflare_quick_tunnel(
    *,
    command: tuple[str, ...],
    startup_timeout: float,
    extra_args: tuple[str, ...],
    local_url: str,
) -> Iterator[str]:
    if not command:
        raise ValueError("cloudflared command must not be empty")

    executable = command[0]
    if shutil.which(executable) is None:
        raise RuntimeError(
            f"could not find {executable!r} on PATH; install cloudflared or configure "
            "the manager to use a different connection method"
        )

    args = [
        *command,
        "tunnel",
        *extra_args,
        "--url",
        local_url,
    ]
    process = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    output = _CapturedOutput()
    output_reader = _start_output_reader(process.stdout, output)

    try:
        public_url = _wait_for_trycloudflare_url(
            process=process,
            output=output,
            output_reader=output_reader,
            startup_timeout=startup_timeout,
        )
    except BaseException:
        _terminate_process(process, output_reader=output_reader, timeout=5.0)
        raise

    try:
        yield public_url
    finally:
        _terminate_process(process, output_reader=output_reader, timeout=5.0)


class _CapturedOutput:
    def __init__(self) -> None:
        self._lines: deque[str] = deque(maxlen=50)
        self._queue: SimpleQueue[str] = SimpleQueue()
        self._lock = threading.Lock()

    def append(self, line: str) -> None:
        with self._lock:
            self._lines.append(line)
        self._queue.put(line)

    def get(self, *, timeout: float) -> str:
        return self._queue.get(timeout=timeout)

    def get_nowait(self) -> str:
        return self._queue.get_nowait()

    def recent_text(self) -> str:
        with self._lock:
            text = "".join(self._lines).strip()
        if text:
            return text
        return "<no output>"


def _start_output_reader(
    stream: IO[str] | None,
    output: _CapturedOutput,
) -> threading.Thread:
    if stream is None:
        raise RuntimeError("cloudflared stdout pipe was not created")

    def read_output() -> None:
        with stream:
            for line in stream:
                output.append(line)

    thread = threading.Thread(
        target=read_output,
        name="furu-cloudflared-output-reader",
        daemon=True,
    )
    thread.start()
    return thread


def _wait_for_trycloudflare_url(
    *,
    process: subprocess.Popen[str],
    output: _CapturedOutput,
    output_reader: threading.Thread,
    startup_timeout: float,
) -> str:
    deadline = time.monotonic() + startup_timeout

    while True:
        while True:
            try:
                line = output.get_nowait()
            except Empty:
                break
            if match := _TRYCLOUDFLARE_URL_RE.search(line):
                return match.group(0).rstrip("/")

        returncode = process.poll()
        if returncode is not None:
            output_reader.join(timeout=1.0)
            while True:
                try:
                    line = output.get_nowait()
                except Empty:
                    break
                if match := _TRYCLOUDFLARE_URL_RE.search(line):
                    return match.group(0).rstrip("/")
            raise RuntimeError(
                "cloudflared exited before printing a trycloudflare URL: "
                f"{output.recent_text()}"
            )

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                "cloudflared did not print a trycloudflare URL within "
                f"{startup_timeout:g} seconds; recent output: {output.recent_text()}"
            )

        try:
            line = output.get(timeout=min(0.05, remaining))
        except Empty:
            continue
        if match := _TRYCLOUDFLARE_URL_RE.search(line):
            return match.group(0).rstrip("/")


def _terminate_process(
    process: subprocess.Popen[str],
    *,
    output_reader: threading.Thread,
    timeout: float,
) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    output_reader.join(timeout=1.0)
