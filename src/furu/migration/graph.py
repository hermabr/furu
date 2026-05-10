from __future__ import annotations

from collections.abc import Iterable

from furu.migration.types import Migration, MigrationNode


class _MigrationGraph:
    def __init__(self, migrations: Iterable[Migration]) -> None:
        self._by_old: dict[MigrationNode, list[Migration]] = {}
        for migration in migrations:
            self._by_old.setdefault(
                (migration.old_fully_qualified_name, migration.old_schema_hash),
                [],
            ).append(migration)

    def paths(
        self, start: MigrationNode, end: MigrationNode
    ) -> Iterable[tuple[Migration, ...]]:
        yield from self._paths(start, end, visited={start}, path=())

    def _paths(
        self,
        current: MigrationNode,
        end: MigrationNode,
        *,
        visited: set[MigrationNode],
        path: tuple[Migration, ...],
    ) -> Iterable[tuple[Migration, ...]]:
        if current == end:
            yield path
            return

        for migration in self._by_old.get(current, ()):
            next_node = (
                migration.new_fully_qualified_name,
                migration.new_schema_hash,
            )
            if next_node in visited:
                continue
            yield from self._paths(
                next_node,
                end,
                visited=visited | {next_node},
                path=(*path, migration),
            )
