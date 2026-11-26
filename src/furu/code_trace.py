"""Traced code versions: which repo code a create() ran, and whether it changed.

A traced class publishes its runs under one folder per code version. The
folder's trace.json says which of the class's own functions ran, with a
fingerprint of each repo file they reached, and names the code-version folders
of the results create() loaded: what it was built on, not a copy of their
traces. The folder's name is a hash of both. A version is live while the files
still have those fingerprints and every version it names is live, so an edit
upstream reaches the results built on it one name at a time. A process decides
this once per version: it runs the code it imported, whatever is edited later.

A file's fingerprint is its AST with docstrings dropped, keeping only the
functions that ran, the module-level bindings that kept code uses (a closure
that follows imports between repo files), every other module-level statement
except an ``if __name__ == "__main__":`` block, and every class that is used or
has a method that ran (minus the methods that did not run). A function that
kept code calls at import (``X = f()``, ``@f``, a default) also counts as ran.
The hash also covers the Python minor version, since ast.dump output differs
between them.

Only .py files count: a YAML config, CSV or style file that create() reads is
not traced, and neither are environment variables or sys.argv, so changing one
keeps the old result. Hashing them would rerun results on edits and environment
changes that do not matter. Instead furu warns when create() reads a repo file
other than .py, or a function that ran mentions os.environ, os.getenv or
sys.argv. Put what they decide in a spec field.
"""

from __future__ import annotations

import ast
import copy
import functools
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import sysconfig
import threading
import time
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import CodeType
from typing import Literal

from pydantic import BaseModel, ConfigDict

from furu.config import get_config
from furu.logging import get_logger
from furu.storage._layout import FIXED, trace_path_in, version_dirs_in
from furu.utils import _hash_dict_deterministically

_PROCESS_START_NS = time.time_ns()
_PYTHON = f"{sys.version_info.major}.{sys.version_info.minor}"
_REPO_ROOT_ENV_VAR = "_FURU_REPO_ROOT"
_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef)
_IMPORTS = (ast.Import, ast.ImportFrom)
_KEEPS_OLD_RESULT = (
    "changing it keeps the old result. Put what it decides in a spec field instead."
)

type Uses = frozenset[str] | Literal["all"]


class StaleCodeError(RuntimeError):
    """A traced file changed after this process imported it."""


# --- repo files --------------------------------------------------------------


def repo_root() -> Path:
    """The tree trace paths are relative to: the git worktree, or the snapshot."""
    if root := os.environ.get(_REPO_ROOT_ENV_VAR):
        return Path(_realpath(root))
    return _git_toplevel(os.getcwd())


@functools.cache
def _git_toplevel(cwd: str) -> Path:
    try:
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"tracing code needs a git repository; {cwd} is not in one"
        ) from exc
    return Path(_realpath(top))


@functools.cache
def _realpath(path: str) -> str:
    return os.path.realpath(path)


@functools.cache
def _excluded() -> tuple[str, ...]:
    # The venv often lives inside the repo; furu's own code is never user code.
    paths = {sys.prefix, sys.base_prefix, sys.exec_prefix, str(Path(__file__).parent)}
    paths |= set(sysconfig.get_paths().values())
    return tuple(_realpath(path) + os.sep for path in paths)


@functools.cache
def _rel(filename: str, root: str) -> str | None:
    """Repo-relative path of a traced source file; root ends with os.sep."""
    if not filename.endswith(".py") or not os.path.isabs(filename):
        return None
    real = _realpath(filename)
    if not real.startswith(root) or real.startswith(_excluded()):
        return None
    return real[len(root) :]


_SOURCES: dict[tuple[str, int, int], str] = {}


def _read(path: Path) -> str | None:
    """Source text, memoized per (path, mtime_ns, size)."""
    try:
        stat = path.stat()
    except OSError:
        return None
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if (text := _SOURCES.get(key)) is None:
        try:
            text = _SOURCES[key] = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None
    return text


def _content_hash(text: str | None) -> str | None:
    if text is None:
        return None
    return hashlib.blake2s(text.encode(), digest_size=10).hexdigest()


# Each repo file's content hash as this process imported it; None if unknown.
_IMPORTED: dict[str, str | None] = {}


def _imported_hash(rel: str, path: Path, *, importing: bool) -> str | None:
    """The hash of rel's code this process runs, recorded the first time we look.

    A file imported before tracing started is trusted only if it is unchanged
    since the process started; content, not mtime, decides after that.
    """
    if importing or rel not in _IMPORTED:
        try:
            fresh = importing or path.stat().st_mtime_ns <= _PROCESS_START_NS
        except OSError:
            fresh = False
        _IMPORTED[rel] = _content_hash(_read(path)) if fresh else None
    return _IMPORTED[rel]


# --- pruning and the uses closure -------------------------------------------


@dataclass(frozen=True, slots=True)
class _Stmt:
    node: ast.stmt
    kind: Literal["class", "def", "bind", "other"]
    name: str = ""
    binds: frozenset[str] = frozenset()
    header: frozenset[str] = frozenset()  # class bases, keywords, decorators


@dataclass(frozen=True, slots=True)
class _Parsed:
    stmts: tuple[_Stmt, ...]
    bound: frozenset[str]
    # (name, import, alias) for imported names nothing else mentions.
    bare_imports: tuple[tuple[str, ast.Import | ast.ImportFrom, ast.alias], ...]


@dataclass(frozen=True, slots=True)
class _Facts:
    mentions: frozenset[str]
    imports: tuple[ast.Import | ast.ImportFrom, ...]
    dynamic: tuple[str, int] | None  # (what, line) of the first dynamic lookup
    untraced: tuple[str, int] | None  # (what, line) of the first env or argv read


def _strip_docstring(
    node: ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
) -> None:
    match node.body:
        case [ast.Expr(value=ast.Constant(value=str())), *rest]:
            node.body = rest
        case _:
            pass


def _mentions(nodes: Sequence[ast.AST]) -> frozenset[str]:
    names: set[str] = set()
    for node in nodes:
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                names.add(child.id)
            elif isinstance(child, ast.Attribute):
                names.add(child.attr)
    return frozenset(names)


def _binds(stmt: ast.stmt) -> frozenset[str] | None:
    """Names a plain binding statement binds; None for anything else."""
    match stmt:
        case ast.Assign(targets=targets) if all(
            isinstance(target, ast.Name) for target in targets
        ):
            return frozenset(
                target.id for target in targets if isinstance(target, ast.Name)
            )
        case (
            ast.AnnAssign(target=ast.Name(id=name))
            | ast.AugAssign(target=ast.Name(id=name))
            | ast.TypeAlias(name=ast.Name(id=name))
        ):
            return frozenset({name})
        case ast.Import(names=aliases) | ast.ImportFrom(names=aliases) if all(
            alias.name != "*" for alias in aliases
        ):
            return frozenset(
                (alias.asname or alias.name).split(".")[0] for alias in aliases
            )
    return None


_PARSED: dict[str, _Parsed | None] = {}


def _parse(text: str) -> _Parsed | None:
    if text in _PARSED:
        return _PARSED[text]
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        _PARSED[text] = None
        return None
    _strip_docstring(tree)
    stmts: list[_Stmt] = []
    for stmt in tree.body:
        match stmt:
            case ast.If(
                test=ast.Compare(
                    left=ast.Name(id="__name__"),
                    comparators=[ast.Constant(value="__main__")],
                )
            ):
                continue  # a worker imports the module, so this never runs
        if isinstance(stmt, ast.ClassDef):
            _strip_docstring(stmt)
            header = _mentions([*stmt.bases, *stmt.keywords, *stmt.decorator_list])
            stmts.append(_Stmt(stmt, "class", stmt.name, header=header))
        elif isinstance(stmt, _DEFS):
            stmts.append(_Stmt(stmt, "def", stmt.name))
        elif (binds := _binds(stmt)) is not None:
            stmts.append(_Stmt(stmt, "bind", binds=binds))
        else:
            stmts.append(_Stmt(stmt, "other"))
    # Names that pruning by use can drop; a def is kept only if it ran.
    bound = frozenset(
        name
        for stmt in stmts
        if stmt.kind != "def"
        for name in (stmt.binds or {stmt.name})
        if name
    )
    mentioned = _mentions([s.node for s in stmts if not isinstance(s.node, _IMPORTS)])
    bare = tuple(
        (name, s.node, alias)
        for s in stmts
        if isinstance(s.node, _IMPORTS)
        for alias in s.node.names
        if (name := (alias.asname or alias.name).split(".")[0]) not in mentioned
        and alias.name != "*"
    )
    parsed = _PARSED[text] = _Parsed(tuple(stmts), bound, bare)
    return parsed


def _dynamic(node: ast.AST, mentions: set[str]) -> str | None:
    """Name a lookup the analysis cannot follow; literal getattr names count as used."""
    match node:
        case ast.Call(
            func=ast.Name(id="getattr" | "hasattr" | "setattr" | "delattr" as fn),
            args=[_, name, *_],
        ):
            if isinstance(name, ast.Constant) and isinstance(name.value, str):
                mentions.add(name.value)
                return None
            return f"{fn} with a computed name"
        case ast.Call(
            func=ast.Name(id="globals" | "eval" | "exec" | "__import__" as fn)
        ):
            return f"{fn}()"
        case ast.Call(func=ast.Name(id="vars"), args=[]):
            return "vars()"
        case ast.Call(
            func=ast.Name(id="import_module") | ast.Attribute(attr="import_module")
        ):
            return "import_module()"
        case ast.Attribute(value=ast.Name(id="sys"), attr="modules"):
            return "sys.modules"
    return None


def _untraced(node: ast.AST) -> str | None:
    """Name a read of an input furu does not trace: the environment or argv."""
    match node:
        case ast.Attribute(value=ast.Name(id="os"), attr="environ" | "getenv" as attr):
            return f"os.{attr}"
        case ast.Attribute(value=ast.Name(id="sys"), attr="argv"):
            return "sys.argv"
    return None


def _calls(node: ast.AST, calls: set[str]) -> None:
    """Add the names running node calls: decorators and defaults, not def bodies."""
    if isinstance(node, (ast.Call, *_DEFS, ast.ClassDef)):
        for func in [node.func] if isinstance(node, ast.Call) else node.decorator_list:
            match func:
                case ast.Name(id=name) | ast.Attribute(attr=name):
                    calls.add(name)
    children: Iterable[ast.AST] = ast.iter_child_nodes(node)
    if isinstance(node, (*_DEFS, ast.Lambda)):
        args = node.args
        children = [
            *getattr(node, "decorator_list", ()),
            *args.defaults,
            *filter(None, args.kw_defaults),
        ]
    for child in children:
        _calls(child, calls)


_FACTS: dict[int, _Facts] = {}  # keyed by id(node); _PARSED keeps the nodes alive


def _facts(node: ast.AST) -> _Facts:
    """Mentions, imports and dynamic lookups of kept code, computed once per node."""
    if (facts := _FACTS.get(id(node))) is None:
        mentions: set[str] = set()
        imports: list[ast.Import | ast.ImportFrom] = []
        dynamic: tuple[str, int] | None = None
        untraced: tuple[str, int] | None = None
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                mentions.add(child.id)
            elif isinstance(child, ast.Attribute):
                mentions.add(child.attr)
            elif isinstance(child, (ast.Import, ast.ImportFrom)):
                imports.append(child)
            elif isinstance(child, (*_DEFS, ast.ClassDef)):
                _strip_docstring(child)
            if (what := _dynamic(child, mentions)) and dynamic is None:
                dynamic = (what, getattr(child, "lineno", 0))
            if (what := _untraced(child)) and untraced is None:
                untraced = (what, getattr(child, "lineno", 0))
        facts = _FACTS[id(node)] = _Facts(
            frozenset(mentions), tuple(imports), dynamic, untraced
        )
    return facts


type _Kept = tuple[ast.stmt, frozenset[str], list[tuple[str | None, _Facts]]]


def _kept(parsed: _Parsed, ran: frozenset[str], uses: Uses) -> Iterator[_Kept]:
    """(node, class header, [(qualname if it ran, facts)]) for each kept statement."""
    every = uses == "all"
    for stmt in parsed.stmts:
        if stmt.kind == "class":
            prefix = stmt.name + "."
            if not (
                every or stmt.name in uses or any(q.startswith(prefix) for q in ran)
            ):
                continue
            assert isinstance(stmt.node, ast.ClassDef)
            members = [
                member
                for member in stmt.node.body
                if not isinstance(member, _DEFS) or prefix + member.name in ran
            ]
            pruned = copy.copy(stmt.node)
            pruned.body = members
            yield (
                pruned,
                stmt.header,
                [
                    (prefix + m.name if isinstance(m, _DEFS) else None, _facts(m))
                    for m in members
                ],
            )
        elif stmt.kind == "def":
            if stmt.name in ran:
                yield stmt.node, frozenset(), [(stmt.name, _facts(stmt.node))]
        elif stmt.kind == "bind" and not every and not (stmt.binds & uses):
            continue
        else:
            yield stmt.node, frozenset(), [(None, _facts(stmt.node))]


_FINGERPRINTS: dict[tuple[str, frozenset[str], Uses], str] = {}


def _fingerprint(text: str | None, ran: frozenset[str], uses: Uses) -> str | None:
    if text is None or (parsed := _parse(text)) is None:
        return None
    key = (text, ran, uses)
    if (fingerprint := _FINGERPRINTS.get(key)) is None:
        body = [node for node, _, _ in _kept(parsed, ran, uses)]
        dump = ast.dump(ast.Module(body=body, type_ignores=[]))
        digest = hashlib.blake2s(dump.encode(), digest_size=10).hexdigest()
        fingerprint = _FINGERPRINTS[key] = f"blake2s:{digest}"
    return fingerprint


class FileTrace(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ran: tuple[str, ...]
    uses: tuple[str, ...] | Literal["all"]
    reason: str | None = None
    fingerprint: str | None

    @property
    def uses_key(self) -> Uses:
        return "all" if self.uses == "all" else frozenset(self.uses)


_MODULES: dict[str, tuple[tuple[tuple[str, object], ...], dict]] = {}


def _repo_modules(root: str) -> dict[str, tuple[str, str | None]]:
    """The file and package of each loaded module that is repo code, by name.

    Kept while sys.modules holds the same module objects under the same names.
    """
    loaded = tuple(sys.modules.items())
    if (seen := _MODULES.get(root)) is None or seen[0] != loaded:
        seen = _MODULES[root] = (
            loaded,
            {
                name: (rel, getattr(module, "__package__", None))
                for name, module in loaded
                if isinstance(file := getattr(module, "__file__", None), str)
                and (rel := _rel(file, root)) is not None
            },
        )
    return seen[1]


def _code(
    ran: Mapping[str, frozenset[str]],
    source: Callable[[str], str | None],
    modules: Mapping[str, tuple[str, str | None]],
) -> tuple[dict[str, FileTrace], list[str]]:
    """Fingerprint every repo file reached from the functions that ran.

    ``modules`` is what _repo_modules gives. Also returns warnings about what
    the fingerprints cannot see.
    """
    packages = dict(modules.values())

    def module_file(name: str) -> str | None:
        return modules[name][0] if name in modules else None

    def imports_repo_code(
        imp: ast.Import | ast.ImportFrom, alias: ast.alias, rel: str
    ) -> bool:
        if isinstance(imp, ast.Import):
            return module_file(alias.name) is not None
        try:
            base = importlib.util.resolve_name(
                "." * imp.level + (imp.module or ""), packages.get(rel) or ""
            )
        except (ImportError, ValueError):
            return False
        return any(module_file(name) for name in (base, f"{base}.{alias.name}"))

    uses: dict[str, set[str]] = {rel: set() for rel in ran}
    called: dict[str, set[str]] = {}  # names kept code calls at import
    everything: dict[str, str] = {}  # files that keep every name, and why
    warnings: list[str] = []
    scanned: set[tuple[str, str]] = set()
    changed = True

    def reach(
        target: str | None,
        names: set[str] | frozenset[str] | None,
        why: str,
        calls: set[str] | frozenset[str] = frozenset(),
    ) -> None:
        nonlocal changed
        if target is None:
            return
        if not calls <= called.setdefault(target, set()):
            called[target] |= calls
            changed = True
        if names is None:  # every name
            if target not in everything:
                everything[target] = why
                uses.setdefault(target, set())
                changed = True
        elif not names <= uses.setdefault(target, set()):
            uses[target] |= names
            changed = True

    while changed:
        changed = False
        for rel in list(uses):
            if (text := source(rel)) is None or (parsed := _parse(text)) is None:
                continue
            at_import = called.setdefault(rel, set())
            ran_here = frozenset(ran.get(rel, ())) | at_import
            every = everything.get(rel)
            items = list(
                _kept(parsed, ran_here, "all" if every else frozenset(uses[rel]))
            )
            for _, _, members in items:
                for qualname, facts in members:
                    if qualname is None or (rel, qualname) in scanned:
                        continue
                    scanned.add((rel, qualname))
                    if facts.dynamic is not None:
                        what, line = facts.dynamic
                        warnings.append(
                            f"{qualname} ({rel}:{line}) uses {what}, so every "
                            f"module-level name in {rel} counts toward its cache "
                            "key. Editing any of them will rerun it."
                        )
                        reach(rel, None, f"{what} in {qualname}")
                    if facts.untraced is not None:
                        what, line = facts.untraced
                        warnings.append(
                            f"{qualname} ({rel}:{line}) reads {what}, which furu "
                            f"does not trace: {_KEEPS_OLD_RESULT}"
                        )
            used = set().union(
                *(header for _, header, _ in items),
                *(facts.mentions for _, _, members in items for _, facts in members),
            )
            for name, imp, alias in parsed.bare_imports:
                if imports_repo_code(imp, alias, rel):  # kept for its side effects
                    used.add(name)
            calls: set[str] = set()
            for node, _, _ in items:
                _calls(node, calls)
                if isinstance(node, _DEFS) and node.name in at_import:
                    for stmt in node.body:  # it runs at import, so does what it calls
                        _calls(stmt, calls)
            for _, _, members in items:
                for _, facts in members:
                    for imp in facts.imports:
                        if isinstance(imp, ast.Import):
                            for alias in imp.names:
                                if (alias.asname or alias.name.split(".")[0]) in used:
                                    reach(
                                        module_file(alias.name),
                                        None if every else used,
                                        every or "",
                                        calls,
                                    )
                            continue
                        base = imp.module or ""
                        if imp.level:
                            try:
                                base = importlib.util.resolve_name(
                                    "." * imp.level + base, packages.get(rel) or ""
                                )
                            except (ImportError, ValueError):
                                continue
                        for alias in imp.names:
                            if alias.name == "*":
                                reach(
                                    module_file(base),
                                    None,
                                    f"from {base} import * in {rel}",
                                    calls,
                                )
                            elif (alias.asname or alias.name) not in used:
                                continue
                            elif (
                                sub := module_file(f"{base}.{alias.name}")
                            ) is not None:
                                reach(sub, None if every else used, every or "", calls)
                            else:
                                bound_as = alias.asname or alias.name
                                reach(
                                    module_file(base),
                                    {alias.name},
                                    "",
                                    {alias.name} if bound_as in calls else frozenset(),
                                )
            if not used <= uses[rel]:
                uses[rel] |= used
                changed = True
            if not calls <= at_import:
                at_import |= calls
                changed = True

    code: dict[str, FileTrace] = {}
    for rel in sorted(uses):
        text = source(rel)
        parsed = _parse(text) if text is not None else None
        bound = parsed.bound if parsed is not None else frozenset()
        stored: Uses = "all" if rel in everything else frozenset(uses[rel] & bound)
        defs = {s.name for s in parsed.stmts if s.kind == "def"} if parsed else set()
        ran_here = frozenset(ran.get(rel, ())) | (called.get(rel, set()) & defs)
        code[rel] = FileTrace(
            ran=tuple(sorted(ran_here)),
            uses="all" if stored == "all" else tuple(sorted(stored)),
            reason=everything.get(rel),
            fingerprint=_fingerprint(text, ran_here, stored),
        )
    return code, warnings


# --- trace.json and code versions ---------------------------------------------


class Trace(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    python: str  # the minor version
    code: dict[str, FileTrace]
    # The code-version folders of the results create() loaded, relative to the
    # storage root. "fixed" ones are always live, so they are left out.
    built_on: tuple[str, ...] = ()

    @property
    def version(self) -> str:
        """The name of the code-version folder this trace is published under."""
        fingerprints = {rel: entry.fingerprint for rel, entry in self.code.items()}
        return _hash_dict_deterministically(
            {"python": self.python, "code": fingerprints, "built_on": self.built_on}
        )


# What this process found in each code-version folder: its trace, and why it
# is outdated (None when live). A published trace.json never changes, and the
# process keeps running the code it imported.
_CHECKED: dict[str, tuple[Trace, str | None]] = {}
_MANY_LIVE_VERSIONS = 100
_WARNED_MANY: set[Path] = set()


def _changed(trace: Trace) -> str | None:
    """Why the code a trace ran differs from the code on disk, if it does."""
    if trace.python != _PYTHON:
        return f"Python {trace.python} → {_PYTHON}"
    root = repo_root()
    for rel, entry in trace.code.items():
        now = _fingerprint(_read(root / rel), frozenset(entry.ran), entry.uses_key)
        if now != entry.fingerprint:
            why = f"{rel} changed" if now else f"{rel} is missing"
            if entry.uses == "all":
                why += f" (all names kept: {entry.reason})"
            return why
    return None


def _check(path: Path, storage_root: Path) -> tuple[Trace, str | None] | None:
    """The trace in the file ``path`` and why it is outdated (None if live).

    ``storage_root`` is what the code versions it names are relative to. None
    when the file cannot be read. It may only not be visible on this machine
    yet, so that answer is not kept, nor one that rests on it. A code version
    has one name in an overlay and under it, so a file the overlay does not
    have is read under it.
    """
    key = os.path.normpath(path)
    if (checked := _CHECKED.get(key)) is not None:
        return checked
    for layer in get_config().layers(Path(key)):
        try:
            trace = Trace.model_validate_json(layer.read_bytes())
            break
        except (OSError, ValueError):
            continue
    else:
        return None
    why = _changed(trace)
    if why is None:
        for name in trace.built_on:
            upstream = storage_root / name
            if why_outdated(upstream) is not None:
                why = f"it was built on {name}, which is outdated"
                if os.path.normpath(trace_path_in(upstream)) not in _CHECKED:
                    return trace, why
                break
    _CHECKED[key] = trace, why
    return trace, why


def why_trace_outdated(path: Path, storage_root: Path) -> str | None:
    """None if the trace in the file ``path`` is live, otherwise why not.

    It is live when the code it ran still matches the code on disk and every
    code version it names, relative to ``storage_root``, is live.
    """
    checked = _check(path, storage_root)
    return checked[1] if checked is not None else f"its {path.name} is missing"


def why_outdated(version: Path) -> str | None:
    """None if the code version at ``version`` is live, otherwise why not."""
    if version.name == FIXED:
        return None
    return why_trace_outdated(trace_path_in(version), version.parents[2])


def versions(class_dir: Path) -> list[tuple[Path, str | None]]:
    """A traced class's code versions, each with why it is outdated (None if live)."""
    found = [
        (version, checked[1])
        for version in version_dirs_in(class_dir)
        if version.name != FIXED
        and (checked := _check(trace_path_in(version), class_dir.parents[1]))
        is not None
    ]
    live = sum(why is None for _, why in found)
    if live > _MANY_LIVE_VERSIONS and class_dir not in _WARNED_MANY:
        _WARNED_MANY.add(class_dir)
        get_logger().warning(
            "%s has %d live code versions, and every lookup checks each of them",
            class_dir,
            live,
        )
    return found


# --- tracing create() ---------------------------------------------------------


class _TraceLog:
    """trace.log in a working folder: an append-only record of what a traced create() ran.

    One JSON array per line: ["source", file, text] when a file first runs,
    ["hash", file, digest] for the code of a file imported earlier,
    ["ran", file, qualname], ["built_on", version] and ["stale", file] when
    what the process imported is unknown.
    """

    def __init__(self, path: Path | None) -> None:
        # Without a path nothing is kept: a call that is never resumed.
        self._fd = (
            None
            if path is None
            else os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        )
        self._path = path
        self.broken = False

    def write(self, record: list[str]) -> None:
        if self._fd is None or self.broken:
            return
        try:
            os.write(self._fd, (json.dumps(record) + "\n").encode())
        except OSError:
            self.abandon()

    def abandon(self) -> None:
        """A log missing entries could resume wrongly; without one, we restart."""
        self.broken = True
        if self._path is not None:
            self._path.unlink(missing_ok=True)

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)


@dataclass(slots=True)
class LogState:
    """What earlier attempts recorded in trace.log."""

    ran: dict[str, set[str]] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)
    hashes: dict[str, str] = field(default_factory=dict)
    built_on: set[str] = field(default_factory=set)
    stale: set[str] = field(default_factory=set)

    @classmethod
    def read(cls, path: Path) -> LogState | None:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return None
        state = cls()
        for line in lines:
            try:
                kind, key, *value = json.loads(line)
            except (TypeError, ValueError):
                continue  # a line cut short by a crash
            match kind, value:
                case "source", [text]:
                    state.sources.setdefault(key, text)
                case "hash", [digest]:
                    state.hashes.setdefault(key, digest)
                case "ran", [qualname]:
                    state.ran.setdefault(key, set()).add(qualname)
                case "built_on", []:
                    state.built_on.add(key)
                case "stale", []:
                    state.stale.add(key)
        return state


class _Collector:
    def __init__(self, root: str, log: _TraceLog, state: LogState) -> None:
        self.root = root
        self.log = log
        # furu's own data directories may live in the repo; reading them is fine.
        # A snapshot worker's root lies inside one, so skip those.
        config = get_config()
        dirs = [anchored for _, anchored in config.directories.anchored()]
        if config.overlay is not None:
            dirs.append(config.overlay)
        self.data_dirs = tuple(
            path
            for anchored in dirs
            if not root.startswith(path := _realpath(str(anchored)) + os.sep)
        )
        self.opened: set[str] = set()  # repo files besides .py that create() read
        self.ran = {rel: set(names) for rel, names in state.ran.items()}
        # Content hashes of the code each file ran: earlier attempts' as
        # trace.log recorded them, and this process's own.
        self.resumed = state.hashes | {
            rel: _content_hash(text) for rel, text in state.sources.items()
        }
        self.hashes: dict[str, str | None] = {}
        self.thread = threading.get_ident()
        self._lock = threading.Lock()

    def record(self, rel: str, *, importing: bool) -> None:
        path = Path(self.root, rel)
        text = _read(path)
        imported = self.hashes[rel] = _imported_hash(rel, path, importing=importing)
        if imported is None:
            self.log.write(["stale", rel])
        elif rel in self.resumed:
            pass  # trace.log keeps the first record of each file
        elif text is not None and imported == _content_hash(text):
            self.log.write(["source", rel, text])
        else:
            self.log.write(["hash", rel, imported])

    def changed(self, rel: str) -> bool:
        """Whether rel on disk differs from the code that ran."""
        ran = self.hashes[rel] if rel in self.hashes else self.resumed.get(rel)
        return ran is None or ran != _content_hash(_read(Path(self.root, rel)))

    def open(self, path: object, flags: int) -> None:
        if not isinstance(path, str | bytes) or flags & os.O_ACCMODE == os.O_WRONLY:
            return  # an fd, or a write
        name = os.path.abspath(os.fsdecode(path))
        if name.endswith((".py", ".pyc")) or not name.startswith(self.root):
            return
        real = os.path.realpath(name)  # not _realpath: its cache would keep every file
        if (
            real.startswith(self.root)
            and not real.startswith((*_excluded(), *self.data_dirs))
            and os.path.isfile(real)
        ):
            self.opened.add(real[len(self.root) :])

    def see(self, code: CodeType) -> None:
        if (rel := _rel(code.co_filename, self.root)) is None:
            return
        qualname = code.co_qualname.split(".<locals>", 1)[0]
        with self._lock:
            if rel not in self.hashes:
                self.record(rel, importing=qualname == "<module>")
            if qualname.startswith("<"):  # module code, module-level lambdas
                return
            names = self.ran.setdefault(rel, set())
            if qualname not in names:
                names.add(qualname)
                self.log.write(["ran", rel, qualname])


_active: _Collector | None = None
_tool_id: int | None = None
# Held by the thread that records, which may record inside its own recording.
# A recording another thread starts waits, so that neither gets the other's code.
_RECORDING = threading.RLock()


def _on_py_start(code: CodeType, offset: int) -> object:
    if (collector := _active) is None:
        return None  # paused: seen at its next call, if create() makes one
    try:
        collector.see(code)
    except Exception:  # noqa: BLE001 -- never break the code being traced
        collector.log.abandon()
    return sys.monitoring.DISABLE


def _on_audit(event: str, args: tuple[object, ...]) -> None:
    # Python runs this for every audit event in the process, forever.
    if event == "open" and (collector := _active) is not None:
        try:
            path, _, flags = args
            assert isinstance(flags, int)
            collector.open(path, flags)
        except Exception:  # noqa: BLE001, S110 -- never break the code being traced
            pass


def _trace_into(collector: _Collector | None) -> None:
    """Send PY_START events to collector; None stops tracing."""
    global _active, _tool_id
    monitoring = sys.monitoring
    if _tool_id is None:
        # A free tool id, so coverage.py and debuggers keep working.
        _tool_id = next((t for t in (3, 4) if monitoring.get_tool(t) is None), None)
        if _tool_id is None:
            raise RuntimeError(
                "sys.monitoring tool ids 3 and 4 are taken; furu needs one"
            )
        monitoring.use_tool_id(_tool_id, "furu")
        monitoring.register_callback(_tool_id, monitoring.events.PY_START, _on_py_start)
        sys.addaudithook(_on_audit)  # cannot be removed, so it checks _active
    _active = collector
    monitoring.set_events(_tool_id, monitoring.events.PY_START if collector else 0)
    # _on_py_start disables each code object it sees; re-arm them all.
    monitoring.restart_events()


def tracing() -> bool:
    """Whether a traced call is running, and its tracing is not paused."""
    return _active is not None


@contextmanager
def paused() -> Generator[None]:
    """Run code that is no part of the call being traced.

    Only the thread that records pauses it: a call on another thread would
    take what this one runs meanwhile out of its trace.
    """
    global _active
    collector = _active
    if collector is None or collector.thread != threading.get_ident():
        yield
        return
    _active = None
    try:
        yield
    finally:
        _active = collector


@dataclass(slots=True)
class Recording:
    """What one traced call ran and loaded: a create(), or a remembered dependency method."""

    storage_root: Path
    built_on: set[str] = field(default_factory=set)
    _log: Callable[[list[str]], None] | None = None
    _collector: _Collector | None = None
    checked: set[str] = field(default_factory=set)  # untraced checks, to warn about

    @property
    def ran(self) -> dict[str, frozenset[str]]:
        if self._collector is None:
            return {}
        return {rel: frozenset(names) for rel, names in self._collector.ran.items()}

    def loaded(self, run: Path) -> None:
        """Note the code version of a published run that the call loaded."""
        version = run.parent.parent
        if version.name == FIXED:
            return
        # Between the paths both stand for, so a version has one name whether
        # its run is in an overlay or under it.
        layers = get_config().layers
        name = os.path.relpath(layers(version)[-1], layers(self.storage_root)[-1])
        if name not in self.built_on:
            self.built_on.add(name)
            if self._log is not None:
                self._log(["built_on", name])


@contextmanager
def recording(
    log_path: Path | None,
    *,
    storage_root: Path,
    resumed: LogState,
) -> Generator[Recording]:
    """Trace the repo code a call runs, logging it to ``log_path`` if there is one.

    A create never runs another create inline (a missing dependency aborts it).
    A remembered dependency method can run one, which is traced on its own:
    the method's trace goes on when the create returns.
    """
    root = str(repo_root()) + os.sep
    log = _TraceLog(log_path)
    collector = _Collector(root, log, resumed)
    for rel, _ in _repo_modules(root).values():
        if rel not in collector.hashes:
            collector.record(rel, importing=False)
    with _RECORDING:
        outer = _active
        _trace_into(collector)
        try:
            yield Recording(storage_root, set(resumed.built_on), log.write, collector)
        finally:
            _trace_into(outer)
            log.close()


def why_not_resumable(state: LogState | None, *, storage_root: Path) -> str | None:
    """None if the code an earlier attempt ran is unchanged on disk."""
    if state is None:
        return "it has no trace.log"
    if state.stale:
        return f"{min(state.stale)} changed while it ran"
    root = repo_root()
    prefix = str(root) + os.sep
    ran = {rel: frozenset(names) for rel, names in state.ran.items()}
    modules = _repo_modules(prefix)

    class Changed(Exception):
        pass

    def logged(rel: str) -> str | None:
        if rel in state.sources:
            return state.sources[rel]
        text = _read(root / rel)
        if rel in state.hashes and _content_hash(text) != state.hashes[rel]:
            raise Changed(rel)
        return text

    try:
        before, _ = _code(ran, logged, modules)
    except Changed as exc:
        return f"{exc} changed"
    now, _ = _code(ran, lambda rel: _read(root / rel), modules)
    for rel in sorted(before.keys() | now.keys()):
        if (
            (b := before.get(rel)) is None
            or (n := now.get(rel)) is None
            or b.fingerprint != n.fingerprint
        ):
            return f"{rel} changed"
    for name in sorted(state.built_on):
        if why_outdated(storage_root / name) is not None:
            return f"it was built on {name}, which is outdated"
    return None


_WARNED: set[str] = set()
# What _code gave, by what it was given but the source, with each text it read.
type _Built = tuple[dict[str, str | None], dict[str, FileTrace], list[str]]
_CODE: dict[object, _Built] = {}


def build_trace(run: Recording, *, label: str, warn: Callable[[str], None]) -> Trace:
    """The trace of a finished call; raises StaleCodeError if unsafe."""
    root = repo_root()
    modules = _repo_modules(str(root) + os.sep)
    key = (root, frozenset(run.ran.items()), frozenset(modules.items()))
    texts, code, found = _CODE.get(key, (None, {}, []))
    if texts is None or any(_read(root / rel) != text for rel, text in texts.items()):
        read: dict[str, str | None] = {}
        code, found = _code(
            run.ran, lambda rel: read.setdefault(rel, _read(root / rel)), modules
        )
        _CODE[key] = read, code, found
    warnings = list(found)
    assert run._collector is not None
    if opened := sorted(run._collector.opened):
        more = f" and {len(opened) - 3} more" if len(opened) > 3 else ""
        warnings.append(
            f"read {', '.join(opened[:3])}{more}, which furu does not "
            f"trace: {_KEEPS_OLD_RESULT}"
        )
    if run.checked:
        warnings.append(
            f"{' and '.join(sorted(run.checked))}, which furu does not trace: the "
            "method is not called again when that changes"
        )
    for warning in warnings:
        if warning not in _WARNED:  # once per process
            _WARNED.add(warning)
            warn(f"{label}: {warning}")
    if stale := sorted(rel for rel in code if run._collector.changed(rel)):
        raise StaleCodeError(
            f"{label}: {', '.join(stale)} changed after this process imported it, "
            "so the code that ran may not match the file on disk. Not publishing "
            "the result; rerun in a fresh process."
        )
    return Trace(python=_PYTHON, code=code, built_on=tuple(sorted(run.built_on)))
