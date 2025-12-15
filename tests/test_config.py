import huldra


def test_get_and_set_huldra_root(huldra_tmp_root, tmp_path) -> None:
    assert huldra.get_huldra_root(version_controlled=False) == huldra_tmp_root / "data"
    assert huldra.get_huldra_root(version_controlled=True) == huldra_tmp_root / "git"
    assert huldra.HULDRA_CONFIG.raw_dir == huldra_tmp_root / "raw"

    huldra.set_huldra_root(tmp_path)
    assert huldra.HULDRA_CONFIG.base_root == tmp_path.resolve()
