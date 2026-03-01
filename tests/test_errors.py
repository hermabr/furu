import pytest

from furu import Furu


class SubtractPositive(Furu[int]):
    a: int
    b: int

    def _create(self) -> int:
        res = self.a - self.b
        assert res > 0
        return res


def test_subtract_positive():
    assert SubtractPositive(a=5, b=3).load_or_create() == 2


def test_subtract_negative():
    with pytest.raises(AssertionError):
        assert SubtractPositive(a=-5, b=3).load_or_create()
