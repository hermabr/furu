import pytest

from furu import Spec


class SubtractPositive(Spec[int]):
    a: int
    b: int

    def create(self) -> int:
        res = self.a - self.b
        assert res > 0
        return res


def test_subtract_positive():
    assert SubtractPositive(a=5, b=3).create() == 2


def test_subtract_negative():
    with pytest.raises(AssertionError):
        SubtractPositive(a=-5, b=3).create()
