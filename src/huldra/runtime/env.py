"""
Optional `.env` loading.

If `python-dotenv` is installed, load a `.env` file from the current directory or
parents (as implemented by `dotenv.load_dotenv()`).
"""

import contextlib


def load_env() -> None:
    with contextlib.suppress(ImportError):
        from dotenv import load_dotenv

        load_dotenv()


# Preserve previous behavior: attempt to load `.env` at import-time.
load_env()

