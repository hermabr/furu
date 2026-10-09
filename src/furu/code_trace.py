"""Traced code versions: which repo code a create() ran, and whether it changed.

A traced spec publishes its result to ``v-<hash>/``, where the hash covers a
fingerprint of every repo file its create() reached and the versions of the
dependencies it loaded. A lookup re-fingerprints the files named in the
version's trace.json and accepts it only if the hash still equals its name.

A file's fingerprint is its AST with docstrings dropped, keeping only the
functions that ran, the module-level bindings that kept code uses (a closure
that follows imports between repo files), every other module-level statement,
and every class that is used or has a method that ran (minus the methods that
did not run).
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
from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
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
_REPO_ROOT_ENV_VAR = "_FURU_REPO_ROOT"
_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef)

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


def _modified_since_start(path: Path) -> bool:
    try:
        return path.stat().st_mtime_ns > _PROCESS_START_NS
    except OSError:
        return True


def _content_hash(text: str) -> str:
    return hashlib.blake2s(text.encode(), digest_size=10).hexdigest()


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
    parsed = _PARSED[text] = _Parsed(tuple(stmts), bound)
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

    uses: dict[str, set[str]] = {rel: set() for rel in ran}
    everything: dict[str, str] = {}  # files that keep every name, and why
    fallbacks: list[_Fallback] = []
    scanned: set[str] = set()
    changed = True

    def reach(
        target: str | None, names: set[str] | frozenset[str] | None, why: str
    ) -> None:
        nonlocal changed
        if target is None:
            return
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
            ran_here = ran.get(rel, frozenset())
            if rel not in scanned:  # functions that ran are kept whatever the uses
                scanned.add(rel)
                for _, _, members in _kept(parsed, ran_here, frozenset()):
                    for qualname, facts in members:
                        if qualname is not None and facts.dynamic is not None:
                            what, line = facts.dynamic
                            fallbacks.append(_Fallback(rel, qualname, what, line))
                            everything.setdefault(rel, f"{what} in {qualname}")
            every = everything.get(rel)
            items = list(
                _kept(parsed, ran_here, "all" if every else frozenset(uses[rel]))
            )
            used = set().union(
                *(header for _, header, _ in items),
                *(facts.mentions for _, _, members in items for _, facts in members),
            )
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
                                )
                            elif (alias.asname or alias.name) not in used:
                                continue
                            elif (
                                sub := module_file(f"{base}.{alias.name}")
                            ) is not None:
                                reach(sub, None if every else used, every or "")
                            else:
                                reach(module_file(base), {alias.name}, "")
            if not used <= uses[rel]:
                uses[rel] |= used
                changed = True

    code: dict[str, FileTrace] = {}
    for rel in sorted(uses):
        text = source(rel)
        parsed = _parse(text) if text is not None else None
        bound = parsed.bound if parsed is not None else frozenset()
        stored: Uses = "all" if rel in everything else frozenset(uses[rel] & bound)
        ran_here = ran.get(rel, frozenset())
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
            {"code": dict(fingerprints), "dependencies": dependencies}
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
    ["hash", file, digest] for repo modules already imported at the start,
    ["ran", file, qualname], ["dependency", object_id, version] and
    ["stale", file] when a file changed after the process imported it.
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
        self._sources = set(state.sources)
        self._lock = threading.Lock()

    def see(self, code: CodeType) -> None:
        if (rel := _rel(code.co_filename, self.root)) is None:
            return
        qualname = code.co_qualname.split(".<locals>", 1)[0]
        with self._lock:
            if rel not in self._sources:
                self._sources.add(rel)
                path = Path(self.root, rel)
                if _modified_since_start(path):
                    self.log.write(["stale", rel])
                if (text := _read(path)) is not None:
                    self.log.write(["source", rel, text])
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
    for module in list(sys.modules.values()):
        file = getattr(module, "__file__", None)
        if (
            isinstance(file, str)
            and (rel := _rel(file, root)) is not None
            and rel not in resumed.hashes
            and (text := _read(Path(root, rel))) is not None
        ):
            log.write(["hash", rel, _content_hash(text)])
    collector = _Collector(root, log, resumed)
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
        if rel in state.hashes and (
            text is None or _content_hash(text) != state.hashes[rel]
        ):
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
    if stale := sorted(rel for rel in code if _modified_since_start(root / rel)):
        raise StaleCodeError(
            f"{label}: {', '.join(stale)} changed after this process imported it, "
            "so the code that ran may not match the file on disk. Not publishing "
            "the result; rerun in a fresh process."
        )
    return Trace(
        code_version="traced",
        code=code,
        dependencies=dict(sorted(run.dependencies.items())),
    )
