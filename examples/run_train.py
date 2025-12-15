from __future__ import annotations

from pathlib import Path

import huldra

from my_project.pipelines import TrainModel


def main() -> None:
    examples_root = Path(__file__).resolve().parent
    huldra.set_huldra_root(examples_root / ".huldra")
    huldra.HULDRA_CONFIG.ignore_git_diff = True

    obj = TrainModel(lr=3e-4, steps=2_000)
    artifact = obj.load_or_create()
    print("artifact:", artifact)
    print("artifact dir:", obj.huldra_dir)
    print("log:", obj.huldra_dir / "huldra.log")


if __name__ == "__main__":
    main()
