from typing import (
    Any,
    Generator,
    Generic,
    Iterator,
    List,
    Literal,
    Optional,
    TypeVar,
    cast,
    overload,
)

from .huldra import Huldra

_H = TypeVar("_H", bound=Huldra, covariant=True)


class _HuldraListMeta(type):
    """Metaclass that provides collection methods for HuldraList subclasses."""

    def _entries(cls: "type[HuldraList[_H]]") -> List[_H]:
        """Collect all Huldra instances from class attributes."""
        items: List[_H] = []
        seen: set[str] = set()

        def maybe_add(obj: Any) -> None:
            if not isinstance(obj, Huldra):
                raise TypeError(f"{obj!r} is not a Huldra instance")

            digest = obj.hexdigest
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

    def __iter__(cls: "type[HuldraList[_H]]") -> Iterator[_H]:
        """Iterate over all Huldra instances."""
        return iter(cls._entries())

    def all(cls: "type[HuldraList[_H]]") -> List[_H]:
        """Get all Huldra instances as a list."""
        return cls._entries()

    def items_iter(
        cls: "type[HuldraList[_H]]",
    ) -> Generator[tuple[str, _H], None, None]:
        """Iterate over (name, instance) pairs."""
        for name, value in cls.__dict__.items():
            if name.startswith("_") or callable(value):
                continue
            if not isinstance(value, dict):
                yield name, cast(_H, value)

    def items(cls: "type[HuldraList[_H]]") -> List[tuple[str, _H]]:
        """Get all (name, instance) pairs as a list."""
        return list(cls.items_iter())

    @overload
    def by_name(
        cls: "type[HuldraList[_H]]", name: str, *, strict: Literal[True] = True
    ) -> _H: ...

    @overload
    def by_name(
        cls: "type[HuldraList[_H]]", name: str, *, strict: Literal[False]
    ) -> Optional[_H]: ...

    def by_name(cls: "type[HuldraList[_H]]", name: str, *, strict: bool = True):
        """Get Huldra instance by name."""
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


class HuldraList(Generic[_H], metaclass=_HuldraListMeta):
    """
    Base class for typed Huldra collections.

    Example:
        class MyComputation(Huldra[str]):
            value: int

            def _create(self) -> str:
                result = f"Result: {self.value}"
                (self.huldra_dir / "result.txt").write_text(result)
                return result

            def _load(self) -> str:
                return (self.huldra_dir / "result.txt").read_text()

        class MyExperiments(HuldraList[MyComputation]):
            exp1 = MyComputation(value=1)
            exp2 = MyComputation(value=2)
            exp3 = MyComputation(value=3)

        # Use the collection
        for exp in MyExperiments:
            result = exp.load_or_create()
            print(result)
    """

    pass
