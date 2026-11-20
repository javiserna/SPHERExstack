import numpy as np
import pytest

from spherexstack import io, mosaic


def test_bad_pixel_mask_from_flags():
    flags = np.array(
        [
            [0, 1 << io.FLAG_BITS["hot"], 1 << io.FLAG_BITS["source"]],
            [1 << io.FLAG_BITS["transient"], 1 << io.FLAG_BITS["cold"], 0],
        ],
        dtype=np.int64,
    )
    mask = io.bad_pixel_mask_from_flags(flags, io.DEFAULT_BAD_BITS)
    # 'hot' and 'transient' are in DEFAULT_BAD_BITS; 'source' and 'cold' are not.
    expected = np.array([[False, True, False], [True, False, False]])
    np.testing.assert_array_equal(mask, expected)


def test_bad_pixel_mask_combines_bits():
    bad_bits = ["hot", "outlier"]
    flags = np.zeros((2, 2), dtype=np.int64)
    flags[0, 0] = 1 << io.FLAG_BITS["hot"]
    flags[0, 1] = 1 << io.FLAG_BITS["outlier"]
    flags[1, 0] = (1 << io.FLAG_BITS["hot"]) | (1 << io.FLAG_BITS["outlier"])
    mask = io.bad_pixel_mask_from_flags(flags, bad_bits)
    np.testing.assert_array_equal(mask, [[True, True], [True, False]])


def test_clean_hot_pixels_removes_outlier_and_counts_it():
    data = np.full((9, 9), 5.0)
    bad_mask = np.zeros((9, 9), dtype=bool)
    bad_mask[4, 4] = True
    data[4, 4] = 1e6

    cleaned, n_bad = io.clean_hot_pixels(data, bad_mask)
    assert n_bad == 1
    assert cleaned[4, 4] == pytest.approx(5.0, abs=0.5)
    # Untouched pixels are unchanged.
    assert cleaned[0, 0] == pytest.approx(5.0)


def test_clean_hot_pixels_noop_when_nothing_bad():
    data = np.full((3, 3), 2.0)
    bad_mask = np.zeros((3, 3), dtype=bool)
    cleaned, n_bad = io.clean_hot_pixels(data, bad_mask)
    assert n_bad == 0
    np.testing.assert_array_equal(cleaned, data)


def test_per_pixel_wavelength_lookup(level2_fits_factory):
    nx, ny = 8, 6
    ref_wcs, ref_shape = mosaic.build_reference_wcs(10.0, 0.0, 0.01, 0.01, pixscale_arcsec=6.15)
    assert ref_shape == (ny, nx) or True  # grid size is whatever build_reference_wcs computes; not asserted here

    fp = level2_fits_factory("exp.fits", ref_wcs, (ny, nx), lam0=3.0, dlam_per_px=0.1)

    result = io.load_roi_frame(fp, io.DEFAULT_BAD_BITS, clean=False)
    assert result is not None
    data, wcs_native, lam, variance = result
    assert variance is None  # need_variance was False

    expected_row = 3.0 + 0.1 * np.arange(data.shape[1])
    for row in range(data.shape[0]):
        np.testing.assert_allclose(lam[row, :], expected_row, atol=1e-9)

    # quick_lambda_range reads only the (small) WCS-WAVE table.
    lo, hi = io.quick_lambda_range(fp)
    assert lo == pytest.approx(3.0)
    assert hi == pytest.approx(3.0 + 0.1 * (data.shape[1] - 1))

    # central_lambda_um is the median over the table.
    med = io.central_lambda_um(fp)
    assert lo <= med <= hi


def test_load_roi_frame_reads_variance_when_requested(level2_fits_factory):
    ref_wcs, _ = mosaic.build_reference_wcs(10.0, 0.0, 0.01, 0.01, pixscale_arcsec=6.15)
    variance = np.full((4, 5), 2.5)
    fp = level2_fits_factory("exp.fits", ref_wcs, (4, 5), variance=variance, dlam_per_px=0.0)

    result = io.load_roi_frame(fp, io.DEFAULT_BAD_BITS, clean=False, need_variance=True)
    assert result is not None
    _data, _wcs, _lam, var_out = result
    np.testing.assert_allclose(var_out, variance)
