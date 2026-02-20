from __future__ import annotations

import re

_SUPPORTED_FLAGS = {
    "i": re.IGNORECASE,
    "m": re.MULTILINE,
    "s": re.DOTALL,
    "x": re.VERBOSE,
}


def parse_regex_flags(flags: str) -> int:
    parsed_flags = 0
    for flag in flags:
        regex_flag = _SUPPORTED_FLAGS.get(flag)
        if regex_flag is None:
            raise ValueError(
                f"unsupported regex flag {flag!r}; supported flags are 'imsx'"
            )
        parsed_flags |= regex_flag
    return parsed_flags


def compile_regex(*, pattern: str, flags: str, path: str) -> re.Pattern[str]:
    parsed_flags = parse_regex_flags(flags)
    try:
        return re.compile(pattern, parsed_flags)
    except re.error as exc:
        raise ValueError(
            f"invalid regex pattern {pattern!r} for path {path!r}: {exc}"
        ) from exc
