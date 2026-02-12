import traceback
from collections.abc import Sequence
from pathlib import Path


class _FuruMissing:
    """Sentinel value for missing fields."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "Furu.MISSING"


MISSING = _FuruMissing()


class FuruError(Exception):
    """Base exception for Furu errors."""

    def __init__(self, message: str, *, hints: Sequence[str] | None = None):
        super().__init__(message)
        self.hints = list(hints or [])

    def _format_hints(self) -> str:
        if not self.hints:
            return ""
        lines = ["", "Hints:"]
        lines.extend([f"  - {hint}" for hint in self.hints])
        return "\n".join(lines)


class FuruExecutionError(FuruError):
    """Raised when executor wiring or scheduling fails."""


class FuruValidationError(FuruError):
    """Raised by `_validate()` to indicate an invalid or missing artifact."""


class FuruWaitTimeout(FuruError):
    """Raised when waiting for a result exceeds _max_wait_time_sec."""

    def __str__(self) -> str:
        msg = super().__str__()
        msg += self._format_hints()
        return msg


class FuruLockNotAcquired(FuruError):
    """Raised when a compute lock cannot be acquired (someone else holds it)."""

    pass


class FuruComputeError(FuruError):
    """Raised when computation fails."""

    def __init__(
        self,
        message: str,
        state_path: Path,
        original_error: Exception | None = None,
        *,
        recorded_error_type: str | None = None,
        recorded_error_message: str | None = None,
        recorded_traceback: str | None = None,
        hints: Sequence[str] | None = None,
    ):
        super().__init__(message, hints=hints)
        self.state_path = state_path
        self.original_error = original_error
        self.recorded_error_type = recorded_error_type
        self.recorded_error_message = recorded_error_message
        self.recorded_traceback = recorded_traceback

    def __str__(self) -> str:
        msg = super().__str__()
        internal_dir = self.state_path.parent
        furu_dir = internal_dir.parent

        msg += f"\n\nFuru dir: {furu_dir}"

        if self.recorded_error_type or self.recorded_error_message:
            msg += "\n\nRecorded error (from state.json):"
            if self.recorded_error_type:
                msg += f"\n  Type: {self.recorded_error_type}"
            if self.recorded_error_message:
                msg += f"\n  Message: {self.recorded_error_message}"

        if self.recorded_traceback:
            msg += f"\n\nRecorded traceback:\n{self.recorded_traceback}"

        if self.original_error:
            msg += f"\n\nOriginal error: {self.original_error}"
            if (
                hasattr(self.original_error, "__traceback__")
                and self.original_error.__traceback__ is not None
            ):
                tb = "".join(
                    traceback.format_exception(
                        type(self.original_error),
                        self.original_error,
                        self.original_error.__traceback__,
                    )
                )
                msg += f"\n\nTraceback:\n{tb}"
        msg += self._format_hints()
        return msg


class FuruMigrationRequired(FuruError):
    """Raised when a migrated object requires explicit migration."""

    def __init__(self, message: str, *, state_path: Path | None = None):
        self.state_path = state_path
        super().__init__(message)

    def __str__(self) -> str:
        msg = super().__str__()
        if self.state_path is not None:
            msg += f"\n\nState file: {self.state_path}"
        return msg


class FuruMissingArtifact(FuruError):
    """Raised when a dependency is missing in executor mode."""


class FuruSpecMismatch(FuruError):
    """Raised when executor spec keys do not match."""
