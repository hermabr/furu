"""Which parts of furu may import which.

The manager and workers never import a Spec: they know only what messages,
plan lines and folders on disk tell them. This checks each module's own
imports; importing any of them still runs ``furu/__init__.py``. Everything not
listed here is the script side, which may import anything.
"""

import ast
from pathlib import Path

import furu

# Shared by scripts, the manager and workers; never imports the script side.
SHARED = (
    "furu.config",
    "furu.constants",
    "furu.dag",
    "furu.locking",
    "furu.logging",
    "furu.protocol",
    "furu.resources",
    "furu.spec_metadata",
    "furu.utils",
)

# The most specific prefix that names a module says what it may import.
ALLOWED = {
    **dict.fromkeys(SHARED, SHARED),
    "furu.locking": ("furu.locking", "furu.logging", "furu.utils"),
    "furu.worker": (*SHARED, "furu.worker"),
    "furu.manager": (*SHARED, "furu.manager", "furu.worker"),
    # What runs next is decided without sockets, threads or pools.
    "furu.manager.queue": SHARED,
}


def _within(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name == p or name.startswith(f"{p}.") for p in prefixes)


def _imports(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module == "furu":
            names |= {f"furu.{alias.name}" for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return {name for name in names if _within(name, ("furu",))}


def test_manager_and_workers_never_import_the_script_side() -> None:
    root = Path(furu.__file__).parent
    broken = []
    for path in sorted(root.rglob("*.py")):
        module = ".".join(("furu", *path.relative_to(root).with_suffix("").parts))
        module = module.removesuffix(".__init__")
        rules = [p for p in ALLOWED if _within(module, (p,))]
        if not rules:
            continue
        allowed = ALLOWED[max(rules, key=len)]
        broken += [
            f"{module} imports {name}"
            for name in sorted(_imports(path))
            if not _within(name, allowed)
        ]
    assert broken == []
