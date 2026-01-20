import furu


class Dummy(furu.Furu[int]):
    def _create(self) -> int:
        return 1

    def _load(self) -> int:
        return 1


def test_raw_dir_is_scoped_to_object(furu_tmp_root) -> None:
    obj = Dummy()
    assert obj.raw_dir == furu.FURU_CONFIG.raw_dir
    assert obj.raw_dir == furu_tmp_root / "raw"
