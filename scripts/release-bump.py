import argparse
import re
import subprocess
from pathlib import Path

PYPROJECT_PATH = Path("pyproject.toml")
CHANGELOG_PATH = Path("CHANGELOG.md")


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def read_text(path: Path) -> str:
    if not path.exists():
        raise RuntimeError(f"Missing required file: {path}")
    return path.read_text(encoding="utf-8")


def get_current_version() -> str:
    text = read_text(PYPROJECT_PATH)
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
    if match is None:
        raise RuntimeError(f"Could not find version in {PYPROJECT_PATH}")
    return match.group(1)


def bump_version(version: str, bump: str) -> str:
    parts = version.split(".")
    if len(parts) != 3:
        raise RuntimeError(f"Expected version in X.Y.Z format, got: {version}")

    major, minor, patch = (int(part) for part in parts)

    if bump == "patch":
        return f"{major}.{minor}.{patch + 1}"
    if bump == "minor":
        return f"{major}.{minor + 1}.0"
    if bump == "major":
        return f"{major + 1}.0.0"

    raise RuntimeError(f"Invalid bump type: {bump}")


def render_pyproject_with_version(new_version: str) -> str:
    text = read_text(PYPROJECT_PATH)
    new_text, count = re.subn(
        r'(^version\s*=\s*")([^"]+)(")',
        rf"\g<1>{new_version}\g<3>",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise RuntimeError(f"Failed to update version in {PYPROJECT_PATH}")
    return new_text


def render_changelog_for_release(new_version: str) -> str:
    lines = read_text(CHANGELOG_PATH).splitlines()

    if len(lines) < 3:
        raise RuntimeError(
            f"{CHANGELOG_PATH} must have at least 3 lines, found {len(lines)}"
        )

    if lines[2] != "## Unreleased":
        raise RuntimeError(
            f"{CHANGELOG_PATH} line 3 must be '## Unreleased', but found {lines[2]!r}"
        )

    lines[2] = f"## v{new_version}"
    return "\n".join(lines) + "\n"


def confirm(current_version: str, new_version: str, *, yes: bool) -> None:
    print(f"Current version: {current_version}")
    print(f"New version: {new_version}")

    if yes:
        return

    response = input("Continue? [y/N] ").strip().lower()
    if response != "y":
        raise SystemExit(1)


def create_release_pr(new_version: str) -> None:
    branch_name = f"release/v{new_version}"
    pr_title = f"Release v{new_version}"

    run(["git", "checkout", "-b", branch_name])
    run(["uv", "sync"])
    run(["git", "add", "pyproject.toml", "uv.lock", "CHANGELOG.md"])
    run(["git", "commit", "-m", f"bump version to v{new_version}"])
    run(["git", "push", "--set-upstream", "origin", branch_name])
    run(
        [
            "gh",
            "pr",
            "create",
            "--title",
            pr_title,
            "--body",
            pr_title,
            "--base",
            "main",
        ]
    )

    print()
    print(f"Release PR created for v{new_version}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bump", choices=["major", "minor", "patch"])
    parser.add_argument("-y", "--yes", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    current_version = get_current_version()
    new_version = bump_version(current_version, args.bump)

    confirm(current_version, new_version, yes=args.yes)

    original_pyproject_text = read_text(PYPROJECT_PATH)
    original_changelog_text = read_text(CHANGELOG_PATH)

    new_pyproject_text = render_pyproject_with_version(new_version)
    new_changelog_text = render_changelog_for_release(new_version)

    if args.dry_run:
        return

    try:
        PYPROJECT_PATH.write_text(new_pyproject_text, encoding="utf-8")
        CHANGELOG_PATH.write_text(new_changelog_text, encoding="utf-8")
        create_release_pr(new_version)
    except Exception:
        PYPROJECT_PATH.write_text(original_pyproject_text, encoding="utf-8")
        CHANGELOG_PATH.write_text(original_changelog_text, encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
