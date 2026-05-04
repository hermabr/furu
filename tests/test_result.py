from dataclasses import dataclass

import numpy as np
import pytest
from pydantic import BaseModel

from furu.result import load_result, save_result


@dataclass(frozen=True)
class DataclassResult:
    coords: tuple[int, int]
    weights: np.ndarray


class PydanticResult(BaseModel):
    metrics: dict[str, float]


def test_result_manifest_saves_numpy_arrays_externally(tmp_path):
    result_path = tmp_path / "result"
    weights = np.arange(6, dtype=np.float32).reshape(2, 3)

    save_result(
        {"metrics": {"loss": 0.1}, "coords": (1, 2), "weights": weights},
        result_path,
        has_lock=lambda: True,
        lock_path=tmp_path / "compute.lock",
    )

    assert (result_path / "manifest.json").exists()
    assert (result_path / "artifacts" / "weights" / "data.npy").exists()
    loaded = load_result(result_path)
    assert loaded["metrics"] == {"loss": 0.1}
    assert loaded["coords"] == (1, 2)
    np.testing.assert_array_equal(loaded["weights"], weights)


def test_result_manifest_uses_padded_list_paths_for_external_arrays(tmp_path):
    result_path = tmp_path / "result"
    arrays = [np.array([i]) for i in range(12)]

    save_result(
        {"arrays": arrays},
        result_path,
        has_lock=lambda: True,
        lock_path=tmp_path / "compute.lock",
    )

    assert (result_path / "artifacts" / "arrays" / "arr_idx_00" / "data.npy").exists()
    assert (result_path / "artifacts" / "arrays" / "arr_idx_11" / "data.npy").exists()


def test_result_manifest_round_trips_pydantic_and_dataclasses(tmp_path):
    pydantic_path = tmp_path / "pydantic"
    save_result(
        PydanticResult(metrics={"loss": 0.12}),
        pydantic_path,
        has_lock=lambda: True,
        lock_path=tmp_path / "compute.lock",
    )
    assert load_result(pydantic_path) == PydanticResult(metrics={"loss": 0.12})

    dataclass_path = tmp_path / "dataclass"
    dataclass_result = DataclassResult(
        coords=(1, 2), weights=np.arange(4, dtype=np.int64)
    )
    save_result(
        dataclass_result,
        dataclass_path,
        has_lock=lambda: True,
        lock_path=tmp_path / "compute.lock",
    )
    loaded = load_result(dataclass_path)
    assert loaded.coords == (1, 2)
    np.testing.assert_array_equal(loaded.weights, dataclass_result.weights)


@pytest.mark.parametrize("key", ["", "a/b", "$furu"])
def test_result_manifest_rejects_invalid_keys(tmp_path, key):
    with pytest.raises(ValueError, match="invalid result key"):
        save_result(
            {key: 1},
            tmp_path / "result",
            has_lock=lambda: True,
            lock_path=tmp_path / "compute.lock",
        )
