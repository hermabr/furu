import furu


def test_get_and_set_furu_root(furu_tmp_root, tmp_path) -> None:
    assert furu.get_furu_root(version_controlled=False) == furu_tmp_root / "data"
    assert furu.get_furu_root(version_controlled=True) == furu_tmp_root / "git"
    assert furu.FURU_CONFIG.raw_dir == furu_tmp_root / "raw"

    furu.set_furu_root(tmp_path)
    assert furu.FURU_CONFIG.base_root == tmp_path.resolve()
