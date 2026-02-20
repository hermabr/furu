from importlib.metadata import version

from furu.core import Furu

__version__ = version("furu")

__all__ = ["__version__", "Furu"]
