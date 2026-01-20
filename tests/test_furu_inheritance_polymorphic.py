"""Tests for Furu inheritance with polymorphic dependencies.

Tests the pattern where:
- Data is an abstract base class (subclass of Furu)
- DataA and DataB are concrete implementations of Data
- Train accepts any Data subclass as a dependency
"""

import json

from furu.serialization import FuruSerializer
from dashboard.pipelines import Data, DataA, DataB, Train


def test_train_with_data_a(furu_tmp_root) -> None:
    """Train should work with DataA as the data dependency."""
    data = DataA(name="dataset_a", source_url="http://example.com/dataset_a")
    train = Train(data=data, epochs=5)

    result = train.load_or_create()

    # Verify training output
    content = json.loads(result.read_text())
    assert content["epochs"] == 5
    assert content["data_type"] == "A"
    assert content["data_name"] == "dataset_a"

    # Verify data was created
    assert data.exists()
    data_content = json.loads(data.load_or_create().read_text())
    assert data_content["type"] == "A"
    assert data_content["url"] == "http://example.com/dataset_a"


def test_train_with_data_b(furu_tmp_root) -> None:
    """Train should work with DataB as the data dependency."""
    data = DataB(name="dataset_b", local_path="/data/local/b")
    train = Train(data=data, epochs=20)

    result = train.load_or_create()

    # Verify training output
    content = json.loads(result.read_text())
    assert content["epochs"] == 20
    assert content["data_type"] == "B"
    assert content["data_name"] == "dataset_b"

    # Verify data was created
    assert data.exists()
    data_content = json.loads(data.load_or_create().read_text())
    assert data_content["type"] == "B"
    assert data_content["path"] == "/data/local/b"


def test_same_train_different_data_produces_different_hashes(furu_tmp_root) -> None:
    """Train with DataA vs DataB should have different furu_hash."""
    data_a = DataA(name="shared_name")
    data_b = DataB(name="shared_name")

    train_a = Train(data=data_a, epochs=10)
    train_b = Train(data=data_b, epochs=10)

    # Different data types should produce different hashes
    hash_a = FuruSerializer.compute_hash(train_a)
    hash_b = FuruSerializer.compute_hash(train_b)
    assert hash_a != hash_b

    # Both should work
    result_a = train_a.load_or_create()
    result_b = train_b.load_or_create()

    content_a = json.loads(result_a.read_text())
    content_b = json.loads(result_b.read_text())

    assert content_a["data_type"] == "A"
    assert content_b["data_type"] == "B"


def test_data_subclasses_have_different_hashes(furu_tmp_root) -> None:
    """DataA and DataB with same name should have different hashes."""
    data_a = DataA(name="same_name")
    data_b = DataB(name="same_name")

    hash_a = FuruSerializer.compute_hash(data_a)
    hash_b = FuruSerializer.compute_hash(data_b)
    assert hash_a != hash_b


def test_train_caches_correctly_with_polymorphic_data(furu_tmp_root) -> None:
    """Second load_or_create should use _load, not _create."""
    data = DataA(name="cached_test")
    train = Train(data=data, epochs=3)

    # First call creates
    result1 = train.load_or_create()
    assert train.exists()

    # Modify the file to verify we're loading, not recreating
    content = json.loads(result1.read_text())
    content["marker"] = "modified"
    result1.write_text(json.dumps(content))

    # Second call should load (returning modified content)
    result2 = train.load_or_create()
    content2 = json.loads(result2.read_text())
    assert content2.get("marker") == "modified"


def test_abstract_data_base_class_raises_not_implemented(furu_tmp_root) -> None:
    """Calling load_or_create on abstract Data base should raise."""
    # Data is abstract - _create raises NotImplementedError
    data = Data(name="abstract")

    try:
        data.load_or_create()
        raise AssertionError("Expected NotImplementedError")
    except NotImplementedError:
        pass
