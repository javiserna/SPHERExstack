import numpy as np

from spherex_ice import tau as tau_mod


def test_build_epoch_tau_map_matches_hand_computed_value(level2_fits_factory, small_wcs):
    shape = (6, 6)
    lam_blue, lam_line, lam_red = 2.625, 3.05, 3.875
    f_blue_val, f_line_val, f_red_val = 2.0, 1.0, 4.0

    f_blue = level2_fits_factory(
        "blue.fits", small_wcs, shape, image=np.full(shape, f_blue_val), lam0=lam_blue
    )
    f_line = level2_fits_factory(
        "line.fits", small_wcs, shape, image=np.full(shape, f_line_val), lam0=lam_line
    )
    f_red = level2_fits_factory(
        "red.fits", small_wcs, shape, image=np.full(shape, f_red_val), lam0=lam_red
    )

    result = tau_mod.build_epoch_tau_map(
        [f_blue, f_line, f_red],
        bad_bits=[],
        clean=False,
        ref_wcs=small_wcs,
        ref_shape=shape,
        lam_line=lam_line,
        dlambda_line=0.05,
        cont_blue_lo=2.50,
        cont_blue_hi=2.75,
        cont_red_lo=3.75,
        cont_red_hi=4.00,
        match_background=True,
        min_overlap_px=5,
        weighting="mean",
    )

    frac = (lam_line - lam_blue) / (lam_red - lam_blue)
    f_cont_expected = f_blue_val + (f_red_val - f_blue_val) * frac
    tau_expected = np.log(f_cont_expected / f_line_val)

    assert result["coverage"] == 1.0
    assert np.allclose(result["tau"], tau_expected, atol=1e-3)

    # With only one exposure per band, every pixel has weight 1: requiring
    # min_weight=2 should mask the whole map out rather than trust it.
    masked = tau_mod.build_epoch_tau_map(
        [f_blue, f_line, f_red],
        bad_bits=[], clean=False, ref_wcs=small_wcs, ref_shape=shape,
        lam_line=lam_line, dlambda_line=0.05, cont_blue_lo=2.50, cont_blue_hi=2.75,
        cont_red_lo=3.75, cont_red_hi=4.00, match_background=True, min_overlap_px=5,
        weighting="mean", min_weight=2.0,
    )
    assert masked["coverage"] == 0.0
    assert np.all(np.isnan(masked["tau"]))


def test_emission_kind_reports_line_minus_continuum(level2_fits_factory, small_wcs):
    shape = (6, 6)
    lam_blue, lam_line, lam_red = 2.77, 3.29, 3.85
    f_blue_val, f_line_val, f_red_val = 1.0, 5.0, 2.0  # a PAH-like bump above the continuum

    f_blue = level2_fits_factory(
        "blue.fits", small_wcs, shape, image=np.full(shape, f_blue_val), lam0=lam_blue
    )
    f_line = level2_fits_factory(
        "line.fits", small_wcs, shape, image=np.full(shape, f_line_val), lam0=lam_line
    )
    f_red = level2_fits_factory(
        "red.fits", small_wcs, shape, image=np.full(shape, f_red_val), lam0=lam_red
    )

    preset = tau_mod.FEATURE_PRESETS["pah"]
    result = tau_mod.build_epoch_tau_map(
        [f_blue, f_line, f_red],
        bad_bits=[],
        clean=False,
        ref_wcs=small_wcs,
        ref_shape=shape,
        lam_line=preset["lam_line"],
        dlambda_line=preset["dlambda_line"],
        cont_blue_lo=preset["cont_blue_lo"],
        cont_blue_hi=preset["cont_blue_hi"],
        cont_red_lo=preset["cont_red_lo"],
        cont_red_hi=preset["cont_red_hi"],
        match_background=True,
        min_overlap_px=5,
        weighting="mean",
        kind=preset["kind"],
    )

    lam_blue_c = 0.5 * (preset["cont_blue_lo"] + preset["cont_blue_hi"])
    lam_red_c = 0.5 * (preset["cont_red_lo"] + preset["cont_red_hi"])
    frac = (lam_line - lam_blue_c) / (lam_red_c - lam_blue_c)
    f_cont_expected = f_blue_val + (f_red_val - f_blue_val) * frac
    expected = f_line_val - f_cont_expected

    assert preset["kind"] == "emission"
    assert result["kind"] == "emission"
    assert result["coverage"] == 1.0
    assert np.allclose(result["tau"], expected, atol=1e-3)


def test_outlier_exposure_is_rejected_from_its_band(level2_fits_factory, small_wcs):
    shape = (6, 6)
    lam_blue, lam_line, lam_red = 2.625, 3.05, 3.875

    f_blue = level2_fits_factory("blue.fits", small_wcs, shape, image=np.full(shape, 2.0), lam0=lam_blue)
    f_red = level2_fits_factory("red.fits", small_wcs, shape, image=np.full(shape, 4.0), lam0=lam_red)
    # Three exposures all sampling the same H2O line band: two agree at 1.0,
    # one is a 20x outlier (a calibration glitch, say) -- it should be
    # dropped, not blended into the average.
    line_good_1 = level2_fits_factory(
        "line_good_1.fits", small_wcs, shape, image=np.full(shape, 1.0), lam0=lam_line
    )
    line_good_2 = level2_fits_factory(
        "line_good_2.fits", small_wcs, shape, image=np.full(shape, 1.0), lam0=lam_line
    )
    line_outlier = level2_fits_factory(
        "line_outlier.fits", small_wcs, shape, image=np.full(shape, 20.0), lam0=lam_line
    )
    files = [f_blue, f_red, line_good_1, line_good_2, line_outlier]

    kwargs = dict(
        bad_bits=[],
        clean=False,
        ref_wcs=small_wcs,
        ref_shape=shape,
        lam_line=lam_line,
        dlambda_line=0.05,
        cont_blue_lo=2.50,
        cont_blue_hi=2.75,
        cont_red_lo=3.75,
        cont_red_hi=4.00,
        match_background=False,
        min_overlap_px=5,
        weighting="mean",
    )

    rejected = tau_mod.build_epoch_tau_map(files, reject_outliers=True, **kwargs)
    assert np.allclose(rejected["f_line"], 1.0, atol=1e-3)

    kept = tau_mod.build_epoch_tau_map(files, reject_outliers=False, **kwargs)
    assert np.allclose(kept["f_line"], (1.0 + 1.0 + 20.0) / 3.0, atol=1e-3)
