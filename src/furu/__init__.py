from importlib.metadata import version

from furu.core import Furu
from furu.logging import get_logger

__version__ = version("furu")

__all__ = ["__version__", "Furu", "get_logger"]
