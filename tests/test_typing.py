import subprocess
import sys
from pathlib import Path
from textwrap import dedent


def _run_ty_check(
    tmp_path: Path, name: str, source: str
) -> subprocess.CompletedProcess[str]:
    snippet_path = tmp_path / name
    snippet_path.write_text(dedent(source))
    repo_root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        [sys.executable, "-m", "ty", "check", str(snippet_path)],
        capture_output=True,
        check=False,
        cwd=repo_root,
        text=True,
    )


def test_create_batched_self_override_type_checks(tmp_path: Path) -> None:
    result = _run_ty_check(
        tmp_path,
        "create_batched_self_ok.py",
        """
        from typing import Self

        from furu import Furu

        class AdderBatchedCorrectTyping(Furu[int]):
            a: int
            b: int

            @classmethod
            def _create_batched(cls, objs: list[Self]) -> list[int]:
                return [obj.a + obj.b for obj in objs]
        """,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output


def test_create_batched_base_type_override_does_not_type_check(
    tmp_path: Path,
) -> None:
    result = _run_ty_check(
        tmp_path,
        "create_batched_base_type_error.py",
        """
        from furu import Furu

        class AdderBatchedIncorrectTyping(Furu[int]):
            a: int
            b: int

            @classmethod
            def _create_batched(cls, objs: list[Furu[int]]) -> list[int]:
                return [obj.a + obj.b for obj in objs]
        """,
    )

    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "unresolved-attribute" in output
    assert "Furu[int]" in output
