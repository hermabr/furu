from __future__ import annotations

from pathlib import Path

from my_project.pipelines import PrepareDataset, TrainModel

import furu


def main() -> None:
    try:
        examples_root = Path(__file__).resolve().parent
    except Exception:
        examples_root = Path(".").resolve().parent
    furu.set_furu_root(examples_root / ".furu")
    # furu.FURU_CONFIG.ignore_git_diff = True

    obj = TrainModel(lr=3e-4, steps=2_000, dataset=PrepareDataset(name="mydata"))
    artifact = obj.load_or_create()
    print("artifact:", artifact)
    print("artifact dir:", obj.furu_dir)
    print("log:", obj.furu_dir / ".furu" / "furu.log")


if __name__ == "__main__":
    main()
