from __future__ import annotations

from pathlib import Path

import huldra

from my_project.pipelines import PrepareDataset, TrainTextModel


def main() -> None:
    examples_root = Path(__file__).resolve().parent
    huldra.set_huldra_root(examples_root / ".huldra")
    huldra.HULDRA_CONFIG.ignore_git_diff = True

    model = TrainTextModel(dataset=PrepareDataset(name="toy"))
    out = model.load_or_create()

    print("model output:", out)
    print("model dir:", model.huldra_dir)
    print("model log:", model.huldra_dir / ".huldra" / "huldra.log")
    print("dataset dir:", model.dataset.huldra_dir)
    print("dataset log:", model.dataset.huldra_dir / ".huldra" / "huldra.log")


if __name__ == "__main__":
    main()
