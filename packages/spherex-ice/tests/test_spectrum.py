import numpy as np
import pytest
from astropy.wcs import WCS

from spherex_ice import spectrum


def _make_wcs(nx=41, ny=41, pixscale_arcsec=6.15):
    w = WCS(naxis=2)
    w.wcs.crpix = [(nx + 1) / 2.0, (ny + 1) / 2.0]
    w.wcs.cdelt = [-pixscale_arcsec / 3600.0, pixscale_arcsec / 3600.0]
    w.wcs.crval = [10.0, 20.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w


def test_aperture_recovers_background_subtracted_flat_field(level2_fits_factory):
    """A perfectly flat field, after background subtraction, gives ~zero flux."""
    nx, ny = 41, 41
    w = _make_wcs(nx, ny)
    bg_value = 5.0  # MJy/sr
    image = np.full((ny, nx), bg_value)
    fp = level2_fits_factory("flat.fits", w, (ny, nx), image=image, lam0=3.0, dlam_per_px=0.0)

    spec = spectrum.extract_aperture_spectrum_at_position(
        10.0, 20.0, [fp], bad_bits=[], aperture_px=4.0, annulus_in_px=8.0, annulus_out_px=12.0,
    )
    assert spec["lam"].size == 1
    assert spec["flux"][0] == pytest.approx(0.0, abs=1e-6)


def test_aperture_recovers_a_known_point_excess(level2_fits_factory):
    """A compact excess well inside the aperture (away from its edge) should be recovered,
    converted to mJy via the same MJy/sr -> mJy formula as spherexstack.units."""
    nx, ny = 41, 41
    w = _make_wcs(nx, ny)
    bg_value = 5.0  # MJy/sr
    excess_value = 2.0  # MJy/sr, added on top of bg_value
    image = np.full((ny, nx), bg_value)
    cx, cy = nx // 2, ny // 2
    image[cy - 1 : cy + 2, cx - 1 : cx + 2] += excess_value  # 3x3 patch = 9 px
    n_excess_px = 9

    fp = level2_fits_factory("excess.fits", w, (ny, nx), image=image, lam0=3.0, dlam_per_px=0.0)

    spec = spectrum.extract_aperture_spectrum_at_position(
        10.0, 20.0, [fp], bad_bits=[], aperture_px=4.0, annulus_in_px=8.0, annulus_out_px=12.0,
    )
    assert spec["lam"].size == 1

    excess_mjy = excess_value * 23.5045 * (6.15**2) / 1000.0
    expected_flux = n_excess_px * excess_mjy
    assert spec["flux"][0] == pytest.approx(expected_flux, rel=0.02)


def test_aperture_skips_exposure_with_no_overlap(level2_fits_factory):
    """A position far outside the cutout's footprint yields no spectrum point."""
    nx, ny = 41, 41
    w = _make_wcs(nx, ny)
    image = np.full((ny, nx), 5.0)
    fp = level2_fits_factory("flat.fits", w, (ny, nx), image=image, lam0=3.0, dlam_per_px=0.0)

    spec = spectrum.extract_aperture_spectrum_at_position(
        100.0, 80.0, [fp], bad_bits=[], aperture_px=4.0,
    )
    assert spec["lam"].size == 0
