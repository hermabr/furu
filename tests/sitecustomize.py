# Worker children are fresh interpreters that conftest.py cannot patch; its
# PYTHONPATH loads this file there instead.
import sys

if sys.orig_argv[1:] == ["-m", "furu.worker._child"]:
    from canned_probes import CANNED_PROBES

    for (owner, name), fake in CANNED_PROBES.items():
        setattr(owner, name, fake)
