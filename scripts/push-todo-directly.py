#!/usr/bin/env python3
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


def format_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def run(
    cmd: list[str], *, capture_output: bool = False, input_text: str | None = None
) -> str:
    print(f"+ {format_cmd(cmd)}")
    completed = subprocess.run(
        cmd,
        check=True,
        text=True,
        input=input_text,
        capture_output=capture_output,
    )
    return completed.stdout if capture_output else ""


def git_output(*args: str) -> str:
    return run(["git", *args], capture_output=True).strip()


def gh_api(path: str, *, method: str = "GET", payload: dict | None = None) -> dict:
    cmd = ["gh", "api"]
    if method != "GET":
        cmd.extend(["-X", method])
    cmd.append(path)

    input_text = None
    if payload is not None:
        cmd.extend(["--input", "-"])
        input_text = json.dumps(payload)

    output = run(cmd, capture_output=True, input_text=input_text)
    return json.loads(output)


def actor_names(items: list[dict] | None) -> list[str]:
    names: list[str] = []
    for item in items or []:
        for key in ("slug", "login", "name"):
            value = item.get(key)
            if value:
                names.append(value)
                break
        else:
            raise RuntimeError(f"Could not normalize actor: {item}")
    return names


def normalize_required_status_checks(protection: dict) -> dict | None:
    status_checks = protection.get("required_status_checks")
    if status_checks is None:
        return None

    normalized: dict = {"strict": status_checks.get("strict", False)}
    checks = status_checks.get("checks")
    if checks is not None:
        normalized["checks"] = [
            {"context": check["context"], "app_id": check.get("app_id")}
            for check in checks
        ]
    else:
        normalized["contexts"] = status_checks.get("contexts", [])
    return normalized


def normalize_pull_request_reviews(protection: dict) -> dict | None:
    reviews = protection.get("required_pull_request_reviews")
    if reviews is None:
        return None

    normalized: dict = {
        "dismiss_stale_reviews": reviews.get("dismiss_stale_reviews", False),
        "require_code_owner_reviews": reviews.get("require_code_owner_reviews", False),
        "require_last_push_approval": reviews.get("require_last_push_approval", False),
        "required_approving_review_count": reviews.get(
            "required_approving_review_count", 0
        ),
    }

    dismissal_restrictions = reviews.get("dismissal_restrictions")
    if dismissal_restrictions:
        normalized["dismissal_restrictions"] = {
            "users": actor_names(dismissal_restrictions.get("users")),
            "teams": actor_names(dismissal_restrictions.get("teams")),
        }

    bypass_allowances = reviews.get("bypass_pull_request_allowances")
    if bypass_allowances:
        normalized["bypass_pull_request_allowances"] = {
            "users": actor_names(bypass_allowances.get("users")),
            "teams": actor_names(bypass_allowances.get("teams")),
            "apps": actor_names(bypass_allowances.get("apps")),
        }

    return normalized


def normalize_restrictions(protection: dict) -> dict | None:
    restrictions = protection.get("restrictions")
    if restrictions is None:
        return None

    return {
        "users": actor_names(restrictions.get("users")),
        "teams": actor_names(restrictions.get("teams")),
        "apps": actor_names(restrictions.get("apps")),
    }


def protection_payload(protection: dict) -> dict:
    return {
        "required_status_checks": normalize_required_status_checks(protection),
        "enforce_admins": protection.get("enforce_admins", {}).get("enabled", False),
        "required_pull_request_reviews": normalize_pull_request_reviews(protection),
        "restrictions": normalize_restrictions(protection),
        "required_linear_history": protection.get("required_linear_history", {}).get(
            "enabled", False
        ),
        "allow_force_pushes": protection.get("allow_force_pushes", {}).get(
            "enabled", False
        ),
        "allow_deletions": protection.get("allow_deletions", {}).get("enabled", False),
        "block_creations": protection.get("block_creations", {}).get("enabled", False),
        "required_conversation_resolution": protection.get(
            "required_conversation_resolution", {}
        ).get("enabled", False),
        "lock_branch": protection.get("lock_branch", {}).get("enabled", False),
        "allow_fork_syncing": protection.get("allow_fork_syncing", {}).get(
            "enabled", False
        ),
    }


def ensure_clean_worktree() -> None:
    status = git_output("status", "--short")
    if status:
        raise RuntimeError("Working tree must be clean before using this script.")


def ensure_branch(expected_branch: str) -> None:
    current_branch = git_output("rev-parse", "--abbrev-ref", "HEAD")
    if current_branch != expected_branch:
        raise RuntimeError(
            f"Current branch is {current_branch!r}; expected {expected_branch!r}."
        )


def ensure_only_allowed_paths(remote: str, branch: str, allowed_path: str) -> None:
    diff_output = git_output("diff", "--name-only", f"{remote}/{branch}..HEAD")
    paths = [path for path in diff_output.splitlines() if path]
    if not paths:
        raise RuntimeError(f"No commits are ahead of {remote}/{branch}.")

    disallowed_paths = sorted({path for path in paths if path != allowed_path})
    if disallowed_paths:
        disallowed = ", ".join(disallowed_paths)
        raise RuntimeError(
            f"This script only allows {allowed_path!r}. Found other paths: {disallowed}."
        )


def verify_restored(protection: dict) -> None:
    if protection.get("required_pull_request_reviews") is None:
        raise RuntimeError("Pull request protection was not restored.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--path", default="TODO.md")
    args = parser.parse_args()

    allowed_path = Path(args.path).as_posix()
    repo = json.loads(
        run(["gh", "repo", "view", "--json", "nameWithOwner"], capture_output=True)
    )["nameWithOwner"]
    protection_path = f"repos/{repo}/branches/{args.branch}/protection"

    ensure_clean_worktree()
    ensure_branch(args.branch)
    run(["git", "fetch", args.remote, args.branch])
    ensure_only_allowed_paths(args.remote, args.branch, allowed_path)

    current_protection = gh_api(protection_path)
    restore_payload = protection_payload(current_protection)
    temporary_payload = dict(restore_payload)
    temporary_payload["required_pull_request_reviews"] = None
    temporary_payload["allow_force_pushes"] = False

    print("Temporarily allowing direct pushes to main...")
    gh_api(protection_path, method="PUT", payload=temporary_payload)

    try:
        run(["git", "push", args.remote, f"HEAD:{args.branch}"])
    finally:
        print("Restoring branch protection...")
        gh_api(protection_path, method="PUT", payload=restore_payload)

    restored_protection = gh_api(protection_path)
    verify_restored(restored_protection)
    print("Done. Branch protection is back in PR-only mode.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(
            f"Command failed with exit code {exc.returncode}: {format_cmd(exc.cmd)}",
            file=sys.stderr,
        )
        raise SystemExit(exc.returncode) from exc
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
