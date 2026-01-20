from __future__ import annotations

from pathlib import Path

import furu

from my_project.pipelines import PrepareDataset, TrainTextModel


def main() -> None:
    examples_root = Path(__file__).resolve().parent
    furu.set_furu_root(examples_root / ".furu")
    furu.FURU_CONFIG.ignore_git_diff = True

    model = TrainTextModel(dataset=PrepareDataset(name="toy"))
    out = model.load_or_create()

    print("model output:", out)
    print("model dir:", model.furu_dir)
    print("model log:", model.furu_dir / ".furu" / "furu.log")
    print("dataset dir:", model.dataset.furu_dir)
    print("dataset log:", model.dataset.furu_dir / ".furu" / "furu.log")


if __name__ == "__main__":
    main()
