from __future__ import annotations

from dataclasses import dataclass

from furu.results.paths import LogicalPath


@dataclass(slots=True)
class ResultPersistenceError(RuntimeError):
    message: str
    logical_path: LogicalPath | None = None

    def __post_init__(self) -> None:
        prefix = self.message
        if self.logical_path is not None:
            prefix = f"{prefix} at {self.logical_path.format()}"
        RuntimeError.__init__(self, prefix)


class ResultSerializationError(ResultPersistenceError):
    pass


class ResultDeserializationError(ResultPersistenceError):
    pass


class UnknownResultCodecError(ResultDeserializationError):
    def __init__(self, codec_id: str, logical_path: LogicalPath | None = None) -> None:
        super().__init__(
            (
                f"Unknown result codec {codec_id!r}. Register it in _result_config() "
                "with registry.register_codec(...) or registry.register_type(...)."
            ),
            logical_path=logical_path,
        )


class ResultPathCollisionError(ResultSerializationError):
    def __init__(
        self,
        artifact_dir: str,
        logical_path: LogicalPath,
        other_path: LogicalPath,
    ) -> None:
        super().__init__(
            (
                f"Result artifact path collision for {artifact_dir!r}; "
                f"{logical_path.format()} and {other_path.format()} encode to the same path"
            ),
            logical_path=logical_path,
        )
