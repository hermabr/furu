from typing import (
    Generator,
    Generic,
    Iterator,
    Literal,
    TypeVar,
    cast,
    overload,
)

from .furu import Furu

_H = TypeVar("_H", bound=Furu, covariant=True)


class _FuruListMeta(type):
    """Metaclass that provides collection methods for FuruList subclasses."""

    def _entries(cls: "type[FuruList[_H]]") -> list[_H]:
        """Collect all Furu instances from class attributes."""
        items: list[_H] = []
        seen: set[str] = set()

        def maybe_add(obj: object) -> None:
            if not isinstance(obj, Furu):
                raise TypeError(f"{obj!r} is not a Furu instance")

            digest = obj.furu_hash
            if digest not in seen:
                seen.add(digest)
                items.append(cast(_H, obj))

        for name, value in cls.__dict__.items():
            if name.startswith("_") or callable(value):
                continue

            if isinstance(value, dict):
                for v in value.values():
                    maybe_add(v)
            elif isinstance(value, list):
                for v in value:
                    maybe_add(v)
            else:
                maybe_add(value)

        return items

    def __iter__(cls: "type[FuruList[_H]]") -> Iterator[_H]:
        """Iterate over all Furu instances."""
        return iter(cls._entries())

    def all(cls: "type[FuruList[_H]]") -> list[_H]:
        """Get all Furu instances as a list."""
        return cls._entries()

    def items_iter(
        cls: "type[FuruList[_H]]",
    ) -> Generator[tuple[str, _H], None, None]:
        """Iterate over (name, instance) pairs."""
        for name, value in cls.__dict__.items():
            if name.startswith("_") or callable(value):
                continue
            if not isinstance(value, dict):
                yield name, cast(_H, value)

    def items(cls: "type[FuruList[_H]]") -> list[tuple[str, _H]]:
        """Get all (name, instance) pairs as a list."""
        return list(cls.items_iter())

    @overload
    def by_name(
        cls: "type[FuruList[_H]]", name: str, *, strict: Literal[True] = True
    ) -> _H: ...

    @overload
    def by_name(
        cls: "type[FuruList[_H]]", name: str, *, strict: Literal[False]
    ) -> _H | None: ...

    def by_name(cls: "type[FuruList[_H]]", name: str, *, strict: bool = True):
        """Get Furu instance by name."""
        attr = cls.__dict__.get(name)
        if attr and not callable(attr) and not name.startswith("_"):
            return cast(_H, attr)

        # Check nested dicts
        for value in cls.__dict__.values():
            if isinstance(value, dict) and name in value:
                return cast(_H, value[name])

        if strict:
            raise KeyError(f"{cls.__name__} has no entry named '{name}'")
        return None


class FuruList(Generic[_H], metaclass=_FuruListMeta):
    """
    Base class for typed Furu collections.

    Example:
        class MyComputation(Furu[str]):
            value: int

            def _create(self) -> str:
                result = f"Result: {self.value}"
                (self.furu_dir / "result.txt").write_text(result)
                return result

            def _load(self) -> str:
                return (self.furu_dir / "result.txt").read_text()

        class MyExperiments(FuruList[MyComputation]):
            exp1 = MyComputation(value=1)
            exp2 = MyComputation(value=2)
            exp3 = MyComputation(value=3)

        # Use the collection
        for exp in MyExperiments:
            result = exp.get()
            print(result)
    """

    pass
