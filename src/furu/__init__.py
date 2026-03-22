from importlib.metadata import version

from furu.core import Furu
from furu.validate import validate

__version__ = version("furu")

__all__ = ["__version__", "Furu", "validate"]
