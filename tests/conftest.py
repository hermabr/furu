import sys
from pathlib import Path


# Make `import furu` work in a src-layout checkout without requiring an install.
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


pytest_plugins = ["furu.testing"]
