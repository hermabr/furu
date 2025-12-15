import traceback
from pathlib import Path
from typing import Optional, Self


class _HuldraMissing:
    """Sentinel value for missing fields."""

    __slots__ = ()

    def __repr__(self: Self) -> str:
        return "Huldra.MISSING"


MISSING = _HuldraMissing()


class HuldraError(Exception):
    """Base exception for Huldra errors."""

    pass


class HuldraWaitTimeout(HuldraError):
    """Raised when waiting for a result exceeds _max_wait_time_sec."""

    pass


class HuldraComputeError(HuldraError):
    """Raised when computation fails."""

    def __init__(
        self,
        message: str,
        state_path: Path,
        original_error: Optional[Exception] = None,
    ):
        self.state_path = state_path
        self.original_error = original_error
        super().__init__(message)

    def __str__(self: Self) -> str:
        msg = super().__str__()  # ty: ignore[invalid-super-argument]
        if self.original_error:
            msg += f"\n\nOriginal error: {self.original_error}"
            if hasattr(self.original_error, "__traceback__"):
                tb = "".join(
                    traceback.format_exception(
                        type(self.original_error),
                        self.original_error,
                        self.original_error.__traceback__,
                    )
                )
                msg += f"\n\nTraceback:\n{tb}"
        msg += f"\n\nState file: {self.state_path}"
        return msg

