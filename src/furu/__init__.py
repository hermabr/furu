from importlib.metadata import version

from furu.core import Furu
from furu.logging import get_logger
from furu.validate import validate

__version__ = version("furu")

__all__ = ["__version__", "Furu", "get_logger", "validate"]
