from __future__ import annotations

import json
import logging
from pathlib import Path

import huldra

log = logging.getLogger(__name__)


class TrainModel(huldra.Huldra[Path]):
    lr: float = huldra.chz.field(default=3e-4)
    steps: int = huldra.chz.field(default=2_000)

    def _create(self) -> Path:
        log.info("training model lr=%s steps=%s", self.lr, self.steps)
        (self.huldra_dir / "metrics.json").write_text(
            json.dumps({"lr": self.lr, "steps": self.steps}, indent=2)
        )
        assert False, "Darn!"
        ckpt = self.huldra_dir / "checkpoint.bin"
        ckpt.write_bytes(b"fake-checkpoint-bytes")
        return ckpt

    def _load(self) -> Path:
        return self.huldra_dir / "checkpoint.bin"


class PrepareDataset(huldra.Huldra[Path]):
    name: str = huldra.chz.field(default="toy")

    def _create(self) -> Path:
        log.info("preparing dataset name=%s", self.name)
        path = self.huldra_dir / "data.txt"
        path.write_text("hello\nworld\n")
        return path

    def _load(self) -> Path:
        return self.huldra_dir / "data.txt"


class TrainTextModel(huldra.Huldra[Path]):
    dataset: PrepareDataset = huldra.chz.field(default_factory=PrepareDataset)

    def _create(self) -> Path:
        log.info("training text model")
        dataset_path = self.dataset.load_or_create()
        model_path = self.huldra_dir / "model.txt"
        model_path.write_text(f"trained on:\n{dataset_path.read_text()}")
        return model_path

    def _load(self) -> Path:
        return self.huldra_dir / "model.txt"
