import signal
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from types import FrameType
from typing import cast

DEFAULT_TERMINATION_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)

SignalHandler = int | Callable[[int, FrameType | None], None] | None


class _TerminationSignal(BaseException):
    def __init__(self, signum: int, frame: FrameType | None):
        self.signum = signum
        self.frame = frame


def _raise_for_signal(signum: int) -> None:
    if signum == signal.SIGINT:
        raise KeyboardInterrupt
    raise SystemExit(128 + signum)


def _restore_signal_handler(*, signum: int, handler: SignalHandler) -> None:
    signal.signal(signum, handler if handler is not None else signal.SIG_DFL)


def _reraise_signal(
    *, signum: int, frame: FrameType | None, handler: SignalHandler
) -> None:
    if callable(handler):
        cast(Callable[[int, FrameType | None], None], handler)(signum, frame)
        return
    if handler == signal.SIG_IGN:
        return
    _raise_for_signal(signum)


@contextmanager
def handle_termination_signals(
    *, signals: Sequence[int] = DEFAULT_TERMINATION_SIGNALS
) -> Iterator[None]:
    previous_handlers: dict[int, SignalHandler] = {}
    received_signal: _TerminationSignal | None = None

    def _handler(signum: int, frame: FrameType | None) -> None:
        raise _TerminationSignal(signum, frame)

    for signum in signals:
        previous_handlers[signum] = signal.signal(signum, _handler)

    try:
        yield
    except _TerminationSignal as exc:
        received_signal = exc
    finally:
        for signum, handler in previous_handlers.items():
            _restore_signal_handler(signum=signum, handler=handler)

    if received_signal is not None:
        _reraise_signal(
            signum=received_signal.signum,
            frame=received_signal.frame,
            handler=previous_handlers[received_signal.signum],
        )
