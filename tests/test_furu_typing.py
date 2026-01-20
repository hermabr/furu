"""Tests that verify the type checker catches incorrect Furu subclass definitions.

These tests run `ty` on code snippets and assert that the expected type errors are produced.
This ensures our type annotations correctly enforce the Furu contract.
"""

import subprocess
import tempfile
from pathlib import Path

import pytest

# All test cases: (name, code, should_pass, expected_errors)
# expected_errors is a list of strings that should appear in output if should_pass=False
TYPING_TEST_CASES: list[tuple[str, str, bool, list[str]]] = [
    (
        "correct_furu_subclass",
        """
from pathlib import Path
import furu

class CorrectData(furu.Furu[Path]):
    name: str = furu.chz.field(default="correct")

    def _create(self) -> Path:
        path = self.furu_dir / "data.txt"
        path.write_text(self.name)
        return path

    def _load(self) -> Path:
        return self.furu_dir / "data.txt"
""",
        True,
        [],
    ),
    (
        "subclass_changing_return_type",
        """
from pathlib import Path
import furu

class BaseData(furu.Furu[Path]):
    def _create(self) -> Path:
        return self.furu_dir / "data.txt"

    def _load(self) -> Path:
        return self.furu_dir / "data.txt"

class WrongSubclass(BaseData):
    def _create(self) -> str:  # Wrong: should be Path
        return "wrong"

    def _load(self) -> str:  # Wrong: should be Path
        return "wrong"
""",
        False,
        ["invalid-method-override", "_create", "_load"],
    ),
    (
        "mismatched_create_type",
        """
from pathlib import Path
import furu

class MismatchedCreate(furu.Furu[Path]):
    def _create(self) -> str:  # Wrong: declared Furu[Path]
        return "should be Path"

    def _load(self) -> Path:
        return self.furu_dir / "data.txt"
""",
        False,
        ["invalid-method-override", "_create"],
    ),
    (
        "mismatched_load_type",
        """
from pathlib import Path
import furu

class MismatchedLoad(furu.Furu[Path]):
    def _create(self) -> Path:
        return self.furu_dir / "data.txt"

    def _load(self) -> str:  # Wrong: declared Furu[Path]
        return "should be Path"
""",
        False,
        ["invalid-method-override", "_load"],
    ),
    (
        "correct_inheritance_chain",
        """
from pathlib import Path
import furu

class Data(furu.Furu[Path]):
    name: str = furu.chz.field(default="base")

    def _create(self) -> Path:
        path = self.furu_dir / "data.txt"
        path.write_text(self.name)
        return path

    def _load(self) -> Path:
        return self.furu_dir / "data.txt"

class DataA(Data):
    extra: str = furu.chz.field(default="a")

    def _create(self) -> Path:  # Correct: same return type as parent
        path = self.furu_dir / "data.txt"
        path.write_text(f"{self.name} {self.extra}")
        return path
    # _load inherited - correct
""",
        True,
        [],
    ),
    (
        "polymorphic_dependency",
        """
from pathlib import Path
import json
import furu

class Data(furu.Furu[Path]):
    name: str = furu.chz.field(default="base")

    def _create(self) -> Path:
        path = self.furu_dir / "data.json"
        path.write_text(json.dumps({"name": self.name}))
        return path

    def _load(self) -> Path:
        return self.furu_dir / "data.json"

class DataA(Data):
    url: str = furu.chz.field(default="http://example.com")

    def _create(self) -> Path:
        path = self.furu_dir / "data.json"
        path.write_text(json.dumps({"name": self.name, "url": self.url}))
        return path

class Train(furu.Furu[Path]):
    data: Data  # Accepts any Data subclass

    def _create(self) -> Path:
        data_path = self.data.load_or_create()  # Works with any Data
        return self.furu_dir / "model.bin"

    def _load(self) -> Path:
        return self.furu_dir / "model.bin"

# Usage should type check correctly
data_a = DataA(name="test")
train = Train(data=data_a)  # DataA is a valid Data
""",
        True,
        [],
    ),
]


def run_ty_check(code: str) -> tuple[int, str]:
    """Run ty check on a code snippet and return (returncode, output)."""
    project_root = Path(__file__).parent.parent

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        f.flush()
        file_path = Path(f.name)

    try:
        result = subprocess.run(
            ["uv", "run", "ty", "check", str(file_path)],
            capture_output=True,
            text=True,
            cwd=project_root,
        )
        return result.returncode, result.stdout + result.stderr
    finally:
        file_path.unlink()


@pytest.mark.parametrize(
    "name,code,should_pass,expected_errors",
    TYPING_TEST_CASES,
    ids=[case[0] for case in TYPING_TEST_CASES],
)
def test_furu_typing(
    name: str,
    code: str,
    should_pass: bool,
    expected_errors: list[str],
) -> None:
    """Test that ty correctly validates Furu subclass typing."""
    returncode, output = run_ty_check(code)

    if should_pass:
        assert returncode == 0, f"Expected no errors for {name}, got:\n{output}"
    else:
        assert returncode != 0, f"Expected type errors for {name}, got no errors"
        for error in expected_errors:
            assert error in output, (
                f"Expected '{error}' in output for {name}:\n{output}"
            )
