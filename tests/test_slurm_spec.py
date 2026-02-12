from types import MappingProxyType

from furu.execution.slurm_spec import SlurmSpec, slurm_spec_key


def test_slurm_spec_normalizes_none_and_empty_extra_for_equality() -> None:
    without_extra = SlurmSpec(partition="cpu")
    with_empty_extra = SlurmSpec(partition="cpu", extra={})

    assert without_extra == with_empty_extra
    assert hash(without_extra) == hash(with_empty_extra)
    assert slurm_spec_key(without_extra) == slurm_spec_key(with_empty_extra)


def test_slurm_spec_normalizes_mapping_types_for_equality() -> None:
    proxy_extra = MappingProxyType(
        {
            "slurm_additional_parameters": MappingProxyType(
                {
                    "qos": "high",
                }
            ),
        }
    )
    spec_with_proxy = SlurmSpec(partition="cpu", extra=proxy_extra)
    spec_with_dict = SlurmSpec(
        partition="cpu",
        extra={"slurm_additional_parameters": {"qos": "high"}},
    )

    assert spec_with_proxy == spec_with_dict
    assert hash(spec_with_proxy) == hash(spec_with_dict)
    assert slurm_spec_key(spec_with_proxy) == slurm_spec_key(spec_with_dict)
