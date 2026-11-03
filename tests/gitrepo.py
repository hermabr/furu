import subprocess
from pathlib import Path


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()


def commit_and_push(repo: Path, message: str = "update") -> str:
    """Commit everything in repo, push it to origin, and return the commit."""
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "--allow-empty", "-m", message)
    git(repo, "push", "-q", "origin", "HEAD")
    return git(repo, "rev-parse", "HEAD")


def init_repo(repo: Path) -> Path:
    """Make repo a git repo on main whose files are committed and pushed to a
    bare origin beside it."""
    origin = repo.with_name(f"{repo.name}-origin.git")
    git(repo.parent, "init", "-q", "--bare", str(origin))
    git(repo, "init", "-q", "-b", "main")
    git(repo, "remote", "add", "origin", str(origin))
    commit_and_push(repo, "init")
    return repo
