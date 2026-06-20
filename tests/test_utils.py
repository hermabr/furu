import pytest

from furu.utils import format_duration


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0.0, "0ms"),
        (0.85, "850ms"),
        (0.999, "999ms"),
        (1.0, "1.0s"),
        (3.2, "3.2s"),
        (59.9, "59.9s"),
        (60.0, "1m00s"),
        (125.0, "2m05s"),
        (3661.0, "61m01s"),  # no hours bucket: minutes keep accumulating
    ],
)
def test_format_duration_buckets(seconds: float, expected: str) -> None:
    assert format_duration(seconds) == expected
