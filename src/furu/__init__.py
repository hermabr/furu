from importlib.metadata import version

from furu.config import FuruSettings, configure, load_settings, settings
from furu.core import Furu

__version__ = version("furu")

__all__ = [
    "__version__",
    "Furu",
    "FuruSettings",
    "configure",
    "load_settings",
    "settings",
]
