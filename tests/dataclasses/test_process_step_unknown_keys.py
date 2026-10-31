import pytest

from modacor.io.io_sources import IoSources
from modacor.modules.technique_modules.scattering.polarization_correction import PolarizationCorrection

POLARISATION_HINT = (
    ". Did you mean 'polarization_factor' or 'minimum_polarization_factor' or 'polarization_angular_offset'?"
)


@pytest.mark.parametrize("method", ["normalize_config_value", "validate_config_value"])
@pytest.mark.parametrize("key", ["polarisation_factor", "zzzzzzzzzzzz"])
def test_unknown_config_key_preserves_prefix_and_suggests_only_close_matches(method, key):
    known_keys = ", ".join(sorted(PolarizationCorrection.effective_config_schema()))
    prefix = f"Key {key} not found in configuration. Known keys: {known_keys}"

    with pytest.raises(KeyError) as exc_info:
        getattr(PolarizationCorrection, method)(key, 0.9)

    message = exc_info.value.args[0]
    assert message.startswith(prefix)
    if key == "polarisation_factor":
        assert message == prefix + POLARISATION_HINT
    else:
        assert message == prefix
        assert "Did you mean" not in message


def test_unknown_config_key_suggestion_at_construction():
    with pytest.raises(KeyError) as exc_info:
        PolarizationCorrection(io_sources=IoSources(), configuration={"polarisation_factor": 0.9})

    assert exc_info.value.args[0].endswith(POLARISATION_HINT)
