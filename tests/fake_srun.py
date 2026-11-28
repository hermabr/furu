"""A stand-in for srun, for tests of the steps a Slurm job is cut into."""

from __future__ import annotations

import stat
import sys
from pathlib import Path

# Notes its options, one JSON line a call, in $FURU_FAKE_SLURM_RECORD_FILE and
# runs the program that follows them. Told to stop, it kills the program as
# Slurm kills a step, and says so as srun does: 128 + the signal.
_SOURCE = """\
import json
import os
import signal
import subprocess
import sys

options = [arg for arg in sys.argv[1:] if arg.startswith("--")]
with open(os.environ["FURU_FAKE_SLURM_RECORD_FILE"], "a", encoding="utf-8") as file:
    file.write(json.dumps({"executable": "srun", "argv": options}) + "\\n")
# A step is no child of srun: a signal to srun does not reach it.
program = subprocess.Popen(sys.argv[1 + len(options) :], start_new_session=True)
signal.signal(signal.SIGTERM, lambda *_: program.kill())
code = program.wait()
sys.exit(128 - code if code < 0 else code)
"""


def install(bin_dir: Path) -> None:
    srun = bin_dir / "srun"
    srun.write_text(f"#!{sys.executable}\n{_SOURCE}")
    srun.chmod(srun.stat().st_mode | stat.S_IXUSR)
