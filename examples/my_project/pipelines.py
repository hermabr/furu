from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import furu

log = logging.getLogger(__name__)


class TrainModel(furu.Furu[Path]):
    lr: float = furu.chz.field(default=3e-4)
    steps: int = furu.chz.field(default=2_000)
    dataset: PrepareDataset
    sleep_sec: float = furu.chz.field(default=0.0)

    def _create(self) -> Path:
        log.info("training model lr=%s steps=%s", self.lr, self.steps)
        (self.furu_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "lr": self.lr,
                    "steps": self.steps,
                    "dataset": str(self.dataset.load_or_create()),
                },
                indent=2,
            )
        )
        ckpt = self.furu_dir / "checkpoint.bin"
        if self.sleep_sec > 0:
            time.sleep(self.sleep_sec)
        ckpt.write_bytes(b"fake-checkpoint-bytes")
        return ckpt

    def _load(self) -> Path:
        return self.furu_dir / "checkpoint.bin"


class PrepareDataset(furu.Furu[Path]):
    name: str = furu.chz.field(default="toy")

    def _create(self) -> Path:
        log.info("preparing dataset name=%s", self.name)
        path = self.furu_dir / "data.txt"
        path.write_text("hello\nworld\n")
        return path

    def _load(self) -> Path:
        return self.furu_dir / "data.txt"


class TrainTextModel(furu.Furu[Path]):
    dataset: PrepareDataset = furu.chz.field(default_factory=PrepareDataset)

    def _create(self) -> Path:
        log.info("training text model")
        dataset_path = self.dataset.load_or_create()
        model_path = self.furu_dir / "model.txt"
        model_path.write_text(f"trained on:\n{dataset_path.read_text()}")
        return model_path

    def _load(self) -> Path:
        return self.furu_dir / "model.txt"
