import numpy as np
import pytest

from spherexstack import mosaic, units


@pytest.fixture
def wcs_6_15():
    """A WCS with the native ~6.15"/px pixel scale."""
    ref_wcs, _ref_shape = mosaic.build_reference_wcs(10.0, 0.0, 12 * 6.15 / 3600.0, 6 * 6.15 / 3600.0, 6.15)
    return ref_wcs


def test_mjy_sr_is_a_no_op(wcs_6_15):
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    converted, bunit = units.convert_units(data, "mjy_sr", wcs_6_15)
    assert bunit == "MJy/sr"
    np.testing.assert_array_equal(converted, data)


def test_mjy_matches_the_irsa_tutorial_formula(wcs_6_15):
    """mJy = (MJy/sr) * 23.5045 * (pixel scale in arcsec)^2 / 1000, per IRSA's own
    SPHEREx mosaic-cube tutorial."""
    data = np.array([[1.0, 2.0]])
    converted, bunit = units.convert_units(data, "mjy", wcs_6_15)
    assert bunit == "mJy"
    expected = data * 23.5045 * (6.15**2) / 1000.0
    np.testing.assert_allclose(converted, expected, rtol=1e-6)


def test_unknown_units_raises(wcs_6_15):
    with pytest.raises(ValueError):
        units.convert_units(np.zeros((2, 2)), "jy", wcs_6_15)


def test_pixscale_arcsec_from_wcs_round_trips(wcs_6_15):
    assert units.pixscale_arcsec_from_wcs(wcs_6_15) == pytest.approx(6.15, rel=1e-6)
