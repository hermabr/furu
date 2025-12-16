from __future__ import annotations

import logging
from pathlib import Path

import huldra

from my_project.pipelines import TrainTextModel


def main() -> None:
    examples_root = Path(__file__).resolve().parent
    huldra.set_huldra_root(examples_root / ".huldra")
    huldra.HULDRA_CONFIG.ignore_git_diff = True

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    obj = TrainTextModel()
    log.info("about to run: %s", obj.to_python(multiline=False))
    obj.load_or_create()
    log.info("wrote logs to: %s", obj.huldra_dir / ".huldra" / "huldra.log")


if __name__ == "__main__":
    main()
