import numpy as np
import pytest

from spherexstack import mosaic


@pytest.fixture
def ref_grid():
    """A small reference grid: 6 rows x 12 columns."""
    ra_center, dec_center = 10.0, 0.0
    pixscale = 6.15
    nx, ny = 12, 6
    width_deg = nx * pixscale / 3600.0
    height_deg = ny * pixscale / 3600.0
    ref_wcs, ref_shape = mosaic.build_reference_wcs(ra_center, dec_center, width_deg, height_deg, pixscale)
    assert ref_shape == (ny, nx)
    return ref_wcs, ref_shape


def test_flat_field_recovered_after_reprojection_and_coaddition(level2_fits_factory, ref_grid):
    """Several flat-field exposures at the same wavelength should coadd to that flat value."""
    ref_wcs, ref_shape = ref_grid
    value = 42.0

    files = []
    for i in range(3):
        wcs_i = mosaic._shifted_wcs(ref_wcs, 0, 0)  # identical footprint, exact pixel alignment
        fp = level2_fits_factory(
            f"exp_{i}.fits",
            wcs_i,
            ref_shape,
            image=np.full(ref_shape, value),
            dlam_per_px=0.0,  # flat wavelength: every pixel is in the same bin
            lam0=3.0,
        )
        files.append(fp)

    mosaics, coverages, _weights = mosaic.build_lambda_mosaics_streaming(
        files,
        bad_bits=[],
        clean=False,
        ref_wcs=ref_wcs,
        ref_shape=ref_shape,
        bin_centers=np.array([3.0]),
        dlambda=0.5,
        match_background=False,
        min_overlap_px=1,
    )
    assert coverages[0] == pytest.approx(1.0)
    np.testing.assert_allclose(mosaics[0], value, atol=1e-4)


def test_background_offset_matching_removes_injected_offset(level2_fits_factory, ref_grid):
    ref_wcs, ref_shape = ref_grid
    ny, nx = ref_shape
    value = 10.0
    offset = 4.0

    # Exposure A covers columns [0, 8); exposure B covers columns [4, 12) -- overlap in [4, 8).
    wcs_a = mosaic._shifted_wcs(ref_wcs, 0, 0)
    shape_a = (ny, 8)
    wcs_b = mosaic._shifted_wcs(ref_wcs, 4, 0)
    shape_b = (ny, 8)

    fp_a = level2_fits_factory(
        "exp_a.fits", wcs_a, shape_a, image=np.full(shape_a, value), dlam_per_px=0.0, lam0=3.0
    )
    fp_b = level2_fits_factory(
        "exp_b.fits", wcs_b, shape_b, image=np.full(shape_b, value + offset), dlam_per_px=0.0, lam0=3.0
    )

    common_kwargs = dict(
        resolved=[fp_a, fp_b],
        bad_bits=[],
        clean=False,
        ref_wcs=ref_wcs,
        ref_shape=ref_shape,
        bin_centers=np.array([3.0]),
        dlambda=0.5,
        min_overlap_px=1,
    )

    mosaics_matched, _cov, _w = mosaic.build_lambda_mosaics_streaming(match_background=True, **common_kwargs)
    mosaics_unmatched, _cov2, _w2 = mosaic.build_lambda_mosaics_streaming(match_background=False, **common_kwargs)

    # With matching: the whole field should recover the common value, offset removed.
    np.testing.assert_allclose(mosaics_matched[0], value, atol=1e-3)

    # Without matching: the overlap region (columns 4..7) averages A and B unweighted,
    # landing at value + offset/2, not at the true value.
    overlap_mean_unmatched = np.nanmean(mosaics_unmatched[0][:, 4:8])
    assert overlap_mean_unmatched == pytest.approx(value + offset / 2.0, abs=1e-3)
    assert abs(overlap_mean_unmatched - value) > offset / 4.0


def test_ivar_equals_mean_when_variances_are_equal(level2_fits_factory, ref_grid):
    ref_wcs, ref_shape = ref_grid
    value_a, value_b = 10.0, 20.0
    same_variance = 4.0

    wcs_a = mosaic._shifted_wcs(ref_wcs, 0, 0)
    wcs_b = mosaic._shifted_wcs(ref_wcs, 0, 0)  # fully overlapping, same footprint

    fp_a = level2_fits_factory(
        "exp_a.fits",
        wcs_a,
        ref_shape,
        image=np.full(ref_shape, value_a),
        variance=np.full(ref_shape, same_variance),
        dlam_per_px=0.0,
    )
    fp_b = level2_fits_factory(
        "exp_b.fits",
        wcs_b,
        ref_shape,
        image=np.full(ref_shape, value_b),
        variance=np.full(ref_shape, same_variance),
        dlam_per_px=0.0,
    )

    common_kwargs = dict(
        resolved=[fp_a, fp_b],
        bad_bits=[],
        clean=False,
        ref_wcs=ref_wcs,
        ref_shape=ref_shape,
        bin_centers=np.array([3.0]),
        dlambda=0.5,
        match_background=False,
        min_overlap_px=1,
    )

    mosaics_mean, _c1, _w1 = mosaic.build_lambda_mosaics_streaming(weighting="mean", **common_kwargs)
    mosaics_ivar, _c2, _w2 = mosaic.build_lambda_mosaics_streaming(weighting="ivar", **common_kwargs)

    expected = (value_a + value_b) / 2.0
    np.testing.assert_allclose(mosaics_mean[0], expected, atol=1e-4)
    np.testing.assert_allclose(mosaics_ivar[0], expected, atol=1e-4)
    np.testing.assert_allclose(mosaics_ivar[0], mosaics_mean[0], atol=1e-6)


def test_ivar_differs_from_mean_when_variances_differ(level2_fits_factory, ref_grid):
    """Sanity check for the test above: unequal variances SHOULD change the weighted result."""
    ref_wcs, ref_shape = ref_grid
    value_a, value_b = 10.0, 20.0

    wcs_a = mosaic._shifted_wcs(ref_wcs, 0, 0)
    wcs_b = mosaic._shifted_wcs(ref_wcs, 0, 0)

    fp_a = level2_fits_factory(
        "exp_a.fits", wcs_a, ref_shape, image=np.full(ref_shape, value_a), variance=np.full(ref_shape, 1.0),
        dlam_per_px=0.0,
    )
    fp_b = level2_fits_factory(
        "exp_b.fits", wcs_b, ref_shape, image=np.full(ref_shape, value_b), variance=np.full(ref_shape, 4.0),
        dlam_per_px=0.0,
    )

    mosaics_ivar, _c, _w = mosaic.build_lambda_mosaics_streaming(
        resolved=[fp_a, fp_b],
        bad_bits=[],
        clean=False,
        ref_wcs=ref_wcs,
        ref_shape=ref_shape,
        bin_centers=np.array([3.0]),
        dlambda=0.5,
        match_background=False,
        min_overlap_px=1,
        weighting="ivar",
    )
    unweighted_mean = (value_a + value_b) / 2.0
    weighted_expected = (value_a / 1.0 + value_b / 4.0) / (1.0 / 1.0 + 1.0 / 4.0)
    assert weighted_expected != pytest.approx(unweighted_mean)
    np.testing.assert_allclose(mosaics_ivar[0], weighted_expected, atol=1e-3)


def test_merge_of_two_tiles_equals_single_mosaic_over_union(ref_grid, tmp_path):
    ref_wcs, ref_shape = ref_grid
    ny, nx = ref_shape
    value = 7.0
    weight_each = 2.0

    # Tile 1 covers columns [0, 8), tile 2 covers columns [4, 12) -- same overlap setup as the
    # background test, but now at the *mosaic* level (MOSAIC+WEIGHT FITS), exercising
    # build_merged_mosaic instead of the per-exposure streaming accumulator.
    wcs_1 = mosaic._shifted_wcs(ref_wcs, 0, 0)
    shape_1 = (ny, 8)
    wcs_2 = mosaic._shifted_wcs(ref_wcs, 4, 0)
    shape_2 = (ny, 8)

    tile1_path = tmp_path / "tile1.fits"
    tile2_path = tmp_path / "tile2.fits"
    mosaic.save_mosaic_fits(
        tile1_path, np.full(shape_1, value), np.full(shape_1, weight_each), wcs_1, lambda_um=3.0, dlambda=0.1
    )
    mosaic.save_mosaic_fits(
        tile2_path, np.full(shape_2, value), np.full(shape_2, weight_each), wcs_2, lambda_um=3.0, dlambda=0.1
    )

    final_mosaic, weight_acc, lambda_um, dlambda_um, coverage = mosaic.build_merged_mosaic(
        [tile1_path, tile2_path], ref_wcs, ref_shape, match_background=False, min_overlap_px=1
    )

    assert coverage == pytest.approx(1.0)
    assert lambda_um == pytest.approx(3.0)
    assert dlambda_um == pytest.approx(0.1)
    # Same value everywhere (including the overlap, where weight doubles but the mean stays `value`).
    np.testing.assert_allclose(final_mosaic, value, atol=1e-4)
    # Overlap columns [4, 8) accumulated weight from both tiles; the rest only from one.
    np.testing.assert_allclose(weight_acc[:, 4:8], 2 * weight_each, atol=1e-4)
    np.testing.assert_allclose(weight_acc[:, :4], weight_each, atol=1e-4)
    np.testing.assert_allclose(weight_acc[:, 8:], weight_each, atol=1e-4)
