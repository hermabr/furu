from __future__ import annotations


class ResultError(Exception):
    pass


class ResultSerializationError(ResultError):
    pass


class ResultDeserializationError(ResultError):
    pass


class ResultCodecError(ResultError):
    pass


class UnknownResultCodecError(ResultCodecError):
    pass
