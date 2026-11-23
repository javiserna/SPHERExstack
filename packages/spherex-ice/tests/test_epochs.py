from spherex_ice import epochs as epochs_mod


def test_group_into_epochs_splits_on_gap(level2_fits_factory, small_wcs):
    shape = (8, 8)
    f1 = level2_fits_factory("a.fits", small_wcs, shape, mjd_obs=60000.0)
    f2 = level2_fits_factory("b.fits", small_wcs, shape, mjd_obs=60001.0)
    f3 = level2_fits_factory("c.fits", small_wcs, shape, mjd_obs=60030.0)  # big gap -> new epoch
    f4 = level2_fits_factory("d.fits", small_wcs, shape, mjd_obs=60031.0)

    eps = epochs_mod.group_into_epochs([f1, f2, f3, f4], max_gap_days=5.0, max_span_days=20.0)

    assert len(eps) == 2
    assert len(eps[0].files) == 2
    assert len(eps[1].files) == 2
    assert eps[0].mjd_min == 60000.0
    assert eps[0].mjd_max == 60001.0
    assert eps[1].mjd_min == 60030.0


def test_group_into_epochs_enforces_max_span(level2_fits_factory, small_wcs):
    shape = (8, 8)
    # Gaps are all small (1 day), but the total span exceeds max_span_days.
    files = [
        level2_fits_factory(f"f{i}.fits", small_wcs, shape, mjd_obs=60000.0 + i)
        for i in range(0, 25)
    ]
    eps = epochs_mod.group_into_epochs(files, max_gap_days=5.0, max_span_days=20.0)

    assert len(eps) >= 2
    assert all(e.span_days <= 20.0 for e in eps)


def test_group_into_epochs_sorts_unordered_input(level2_fits_factory, small_wcs):
    shape = (8, 8)
    f_late = level2_fits_factory("late.fits", small_wcs, shape, mjd_obs=60005.0)
    f_early = level2_fits_factory("early.fits", small_wcs, shape, mjd_obs=60000.0)

    eps = epochs_mod.group_into_epochs([f_late, f_early], max_gap_days=5.0, max_span_days=20.0)

    assert len(eps) == 1
    assert eps[0].files[0].name == "early.fits"
    assert eps[0].files[1].name == "late.fits"
