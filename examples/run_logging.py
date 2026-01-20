from __future__ import annotations

import logging
from pathlib import Path

import furu

from my_project.pipelines import TrainTextModel


def main() -> None:
    examples_root = Path(__file__).resolve().parent
    furu.set_furu_root(examples_root / ".furu")
    furu.FURU_CONFIG.ignore_git_diff = True

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    obj = TrainTextModel()
    log.info("about to run: %s", obj.to_python(multiline=False))
    obj.load_or_create()
    log.info("wrote logs to: %s", obj.furu_dir / ".furu" / "furu.log")


if __name__ == "__main__":
    main()
