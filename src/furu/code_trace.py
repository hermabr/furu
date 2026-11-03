"""Traced code versions: which repo code a create() ran, and whether it changed.

A traced spec publishes its result to ``v-<hash>/``, where the hash covers a
fingerprint of every repo file its create() reached and the versions of the
dependencies it loaded. A lookup re-fingerprints the files named in the
version's trace.json and accepts it only if the hash still equals its name.

A file's fingerprint is its AST with docstrings dropped, keeping only the
functions that ran, the module-level bindings that kept code uses (a closure
that follows imports between repo files), every other module-level statement
except an ``if __name__ == "__main__":`` block, and every class that is used or
has a method that ran (minus the methods that did not run). A function that
kept code calls at import (``X = f()``, ``@f``, a default) also counts as ran.
The hash also covers the Python minor version, since ast.dump output differs
between them.

Only .py files count: a YAML config, CSV or style file that create() reads is
not traced, so editing one keeps the old result. Put what such a file decides
in a spec field instead.
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

from furu.config import CodeVersion
from furu.storage._layout import trace_path_in
from furu.utils import _hash_dict_deterministically, spec_label

_PROCESS_START_NS = time.time_ns()
_PYTHON = f"{sys.version_info.major}.{sys.version_info.minor}"
_REPO_ROOT_ENV_VAR = "_FURU_REPO_ROOT"
_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef)
_IMPORTS = (ast.Import, ast.ImportFrom)

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
            f'code_version="traced" needs a git repository; {cwd} is not in one'
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
        facts = _FACTS[id(node)] = _Facts(frozenset(mentions), tuple(imports), dynamic)
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


@dataclass(frozen=True, slots=True)
class _Fallback:
    file: str
    qualname: str
    what: str
    line: int


class FileTrace(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ran: tuple[str, ...]
    uses: tuple[str, ...] | Literal["all"]
    reason: str | None = None
    fingerprint: str | None

    @property
    def uses_key(self) -> Uses:
        return "all" if self.uses == "all" else frozenset(self.uses)


def _code(
    ran: Mapping[str, frozenset[str]],
    source: Callable[[str], str | None],
    root: str,
) -> tuple[dict[str, FileTrace], list[_Fallback]]:
    """Fingerprint every repo file reached from the functions that ran."""
    packages = {
        rel: getattr(module, "__package__", None)
        for module in list(sys.modules.values())
        if isinstance(file := getattr(module, "__file__", None), str)
        and (rel := _rel(file, root)) is not None
    }

    def module_file(name: str) -> str | None:
        file = getattr(sys.modules.get(name), "__file__", None)
        return _rel(file, root) if isinstance(file, str) else None

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
    fallbacks: list[_Fallback] = []
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
                    if qualname is None or facts.dynamic is None:
                        continue
                    if (rel, qualname) not in scanned:
                        scanned.add((rel, qualname))
                        what, line = facts.dynamic
                        fallbacks.append(_Fallback(rel, qualname, what, line))
                        reach(rel, None, f"{what} in {qualname}")
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
    return code, fallbacks


# --- trace.json and lookup --------------------------------------------------


class Trace(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code_version: CodeVersion
    python: str | None = None  # the minor version, for traced specs
    code: dict[str, FileTrace] | None = None
    # object_id -> its loaded version directory, relative to the storage root
    dependencies: dict[str, str] = {}

    def hashed_dependencies(self) -> dict[str, str]:
        """A fixed spec's version covers only dependencies that were traced."""
        if self.code is not None:
            return self.dependencies
        return {k: v for k, v in self.dependencies.items() if Path(v).name != "v-fixed"}

    def version_name(self, fingerprints: Mapping[str, str | None] | None = None) -> str:
        dependencies = self.hashed_dependencies()
        if self.code is None:
            if not dependencies:
                return "v-fixed"
            return "v-fixed-" + _hash_dict_deterministically(
                {"dependencies": dependencies}
            )
        if fingerprints is None:
            fingerprints = {rel: entry.fingerprint for rel, entry in self.code.items()}
        return "v-" + _hash_dict_deterministically(
            {
                "python": self.python,
                "code": dict(fingerprints),
                "dependencies": dependencies,
            }
        )


_TRACES: dict[Path, Trace] = {}  # published versions never change


def _read_trace(version: Path) -> Trace | None:
    if (trace := _TRACES.get(version)) is None:
        try:
            trace = Trace.model_validate_json(trace_path_in(version).read_bytes())
        except (OSError, ValueError):
            return None
        _TRACES[version] = trace
    return trace


def _check(
    version: Path, storage_root: Path, memo: dict[Path, str | None]
) -> str | None:
    """None if version is valid for the code on disk now, otherwise why not."""
    if version in memo:
        return memo[version]
    memo[version] = why = _check_uncached(version, storage_root, memo)
    return why


def _check_uncached(
    version: Path, storage_root: Path, memo: dict[Path, str | None]
) -> str | None:
    if version.name == "v-fixed":
        return None if version.is_dir() else f"{version} is missing"
    if (trace := _read_trace(version)) is None:
        return f"{version} has no readable trace.json"
    fingerprints = None
    if trace.code is not None:
        if trace.python != _PYTHON:
            return f"Python {trace.python} → {_PYTHON}"
        root = repo_root()
        fingerprints = {
            rel: _fingerprint(_read(root / rel), frozenset(entry.ran), entry.uses_key)
            for rel, entry in trace.code.items()
        }
        for rel, entry in trace.code.items():
            if fingerprints[rel] != entry.fingerprint:
                why = f"{rel} changed" if fingerprints[rel] else f"{rel} is missing"
                if entry.uses == "all":
                    why += f" (all names kept: {entry.reason})"
                return why
    for object_id, dependency in trace.hashed_dependencies().items():
        if _check(storage_root / dependency, storage_root, memo) is not None:
            return f"its dependency {spec_label(*object_id.split(':'))} changed"
    if trace.version_name(fingerprints) != version.name:
        return f"{version.name} does not match its trace.json"
    return None


def _candidates(identity_dir: Path, code_version: CodeVersion | None) -> list[Path]:
    try:
        entries = [
            entry
            for entry in os.scandir(identity_dir)
            if entry.name.startswith("v-")
            and entry.name != "v-fixed"
            and entry.is_dir()
        ]
    except FileNotFoundError:
        return []
    if code_version is not None:
        fixed = code_version == "fixed"
        entries = [e for e in entries if e.name.startswith("v-fixed-") == fixed]
    entries.sort(key=lambda entry: entry.stat().st_mtime_ns, reverse=True)
    return [Path(entry.path) for entry in entries]


def valid_version(
    identity_dir: Path, code_version: CodeVersion | None, *, storage_root: Path
) -> Path | None:
    """The published version still valid for the code on disk; None = any kind."""
    if code_version != "traced" and (fixed := identity_dir / "v-fixed").is_dir():
        return fixed  # one stat, the common case
    memo: dict[Path, str | None] = {}
    for version in _candidates(identity_dir, code_version):
        if _check(version, storage_root, memo) is None:
            return version
    return None


def is_valid(version: Path, *, storage_root: Path) -> bool:
    return _check(version, storage_root, {}) is None


def why_outdated(
    identity_dir: Path, code_version: CodeVersion, *, storage_root: Path
) -> str | None:
    """Why the newest stored version of this kind is no longer valid, if any."""
    candidates = _candidates(identity_dir, code_version)
    return _check(candidates[0], storage_root, {}) if candidates else None


# --- tracing create() ---------------------------------------------------------


class _TraceLog:
    """attempt/trace.log: an append-only record of what a traced create() ran.

    One JSON array per line: ["source", file, text] when a file first runs,
    ["hash", file, digest] for the code of a file imported earlier,
    ["ran", file, qualname], ["dependency", object_id, version] and
    ["stale", file] when what the process imported is unknown.
    """

    def __init__(self, paths: Sequence[Path]) -> None:
        self._fds = [
            os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            for path in paths
        ]
        self._paths = paths
        self.broken = False

    def write(self, record: list[str]) -> None:
        if self.broken:
            return
        line = (json.dumps(record) + "\n").encode()
        try:
            for fd in self._fds:
                os.write(fd, line)
        except OSError:
            self.abandon()

    def abandon(self) -> None:
        """A log missing entries could resume wrongly; without one, we restart."""
        self.broken = True
        for path in self._paths:
            path.unlink(missing_ok=True)

    def close(self) -> None:
        for fd in self._fds:
            os.close(fd)


@dataclass(slots=True)
class LogState:
    """What earlier attempts recorded in trace.log."""

    ran: dict[str, set[str]] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)
    hashes: dict[str, str] = field(default_factory=dict)
    dependencies: dict[str, str] = field(default_factory=dict)
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
                case "dependency", [version]:
                    state.dependencies[key] = version
                case "stale", []:
                    state.stale.add(key)
        return state

    def merge(self, other: LogState) -> None:
        for rel, names in other.ran.items():
            self.ran.setdefault(rel, set()).update(names)
        self.sources |= other.sources
        self.hashes |= other.hashes
        self.dependencies |= other.dependencies
        self.stale |= other.stale


class _Collector:
    def __init__(self, root: str, log: _TraceLog, state: LogState) -> None:
        self.root = root
        self.log = log
        self.ran = {rel: set(names) for rel, names in state.ran.items()}
        # Content hashes of the code each file ran: earlier attempts' as
        # trace.log recorded them, and this process's own.
        self.resumed = state.hashes | {
            rel: _content_hash(text) for rel, text in state.sources.items()
        }
        self.hashes: dict[str, str | None] = {}
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


def _on_py_start(code: CodeType, offset: int) -> object:
    if (collector := _active) is not None:
        try:
            collector.see(code)
        except Exception:  # noqa: BLE001 -- never break the code being traced
            collector.log.abandon()
    return sys.monitoring.DISABLE


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
    _active = collector
    monitoring.set_events(_tool_id, monitoring.events.PY_START if collector else 0)
    # _on_py_start disables each code object it sees; re-arm them all.
    monitoring.restart_events()


@dataclass(slots=True)
class Recording:
    """What one create() call (one spec, or one batch) ran and loaded."""

    code_version: CodeVersion
    dependencies: dict[str, str]
    log: Callable[[list[str]], None] | None = None
    _collector: _Collector | None = None

    @property
    def ran(self) -> dict[str, frozenset[str]]:
        if self._collector is None:
            return {}
        return {rel: frozenset(names) for rel, names in self._collector.ran.items()}


@contextmanager
def recording(
    log_paths: Sequence[Path], *, code_version: CodeVersion, resumed: LogState
) -> Generator[Recording]:
    """Trace the repo code a create() runs.

    A create never runs another create inline (a missing dependency aborts it),
    so at most one create is traced at a time.
    """
    if code_version == "fixed":
        yield Recording(code_version, {})
        return
    if _active is not None:
        raise RuntimeError("a traced create() is already running in this process")
    root = str(repo_root()) + os.sep
    log = _TraceLog(log_paths)
    collector = _Collector(root, log, resumed)
    for module in list(sys.modules.values()):
        file = getattr(module, "__file__", None)
        if (
            isinstance(file, str)
            and (rel := _rel(file, root)) is not None
            and rel not in collector.hashes
        ):
            collector.record(rel, importing=False)
    _trace_into(collector)
    try:
        yield Recording(code_version, dict(resumed.dependencies), log.write, collector)
    finally:
        _trace_into(None)
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
        before, _ = _code(ran, logged, prefix)
    except Changed as exc:
        return f"{exc} changed"
    now, _ = _code(ran, lambda rel: _read(root / rel), prefix)
    for rel in sorted(before.keys() | now.keys()):
        if (
            (b := before.get(rel)) is None
            or (n := now.get(rel)) is None
            or b.fingerprint != n.fingerprint
        ):
            return f"{rel} changed"
    memo: dict[Path, str | None] = {}
    for object_id, dependency in state.dependencies.items():
        if _check(storage_root / dependency, storage_root, memo) is not None:
            return f"its dependency {spec_label(*object_id.split(':'))} changed"
    return None


_WARNED: set[tuple[str, str]] = set()


def build_trace(run: Recording, *, label: str, warn: Callable[[str], None]) -> Trace:
    """The trace.json for a finished create(); raises StaleCodeError if unsafe."""
    if run.code_version == "fixed":
        return Trace(
            code_version="fixed", dependencies=dict(sorted(run.dependencies.items()))
        )
    root = repo_root()
    code, fallbacks = _code(run.ran, lambda rel: _read(root / rel), str(root) + os.sep)
    for fallback in fallbacks:
        if (fallback.file, fallback.qualname) not in _WARNED:
            _WARNED.add((fallback.file, fallback.qualname))
            warn(
                f"{label}: {fallback.qualname} ({fallback.file}:{fallback.line}) uses "
                f"{fallback.what}, so every module-level name in {fallback.file} "
                "counts toward its cache key. Editing any of them will rerun it."
            )
    assert run._collector is not None
    if stale := sorted(rel for rel in code if run._collector.changed(rel)):
        raise StaleCodeError(
            f"{label}: {', '.join(stale)} changed after this process imported it, "
            "so the code that ran may not match the file on disk. Not publishing "
            "the result; rerun in a fresh process."
        )
    return Trace(
        code_version="traced",
        python=_PYTHON,
        code=code,
        dependencies=dict(sorted(run.dependencies.items())),
    )
