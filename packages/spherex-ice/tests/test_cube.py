import numpy as np

from spherex_ice import cube as cube_mod
from spherex_ice import tau as tau_mod


def _fake_result(f_blue, f_line, f_red, lam_blue, lam_line, lam_red, shape=(4, 4)):
    ones = np.ones(shape)
    return {
        "f_blue": np.full(shape, f_blue), "f_line": np.full(shape, f_line), "f_red": np.full(shape, f_red),
        "w_blue": ones.copy(), "w_line": ones.copy(), "w_red": ones.copy(),
        "cov_blue": 1.0, "cov_line": 1.0, "cov_red": 1.0,
        "lam_blue": lam_blue, "lam_line": lam_line, "lam_red": lam_red,
    }


def test_bands_from_result_matches_feature_bands():
    feature = tau_mod.FEATURE_PRESETS["h2o"]
    result = _fake_result(2.0, 1.0, 4.0, 2.625, 3.05, 3.875)

    bands = cube_mod.bands_from_result("h2o", feature, result)

    assert [b["name"] for b in bands] == ["h2o_blue", "h2o_line", "h2o_red"]
    assert [b["role"] for b in bands] == ["blue", "line", "red"]
    assert np.allclose(bands[0]["flux"], 2.0)
    assert np.allclose(bands[1]["flux"], 1.0)
    assert np.allclose(bands[2]["flux"], 4.0)
    assert bands[1]["dlambda"] == feature["dlambda_line"]


def test_save_and_load_cube_roundtrip(tmp_path, small_wcs):
    feature_h2o = tau_mod.FEATURE_PRESETS["h2o"]
    feature_co2 = tau_mod.FEATURE_PRESETS["co2"]
    result_h2o = _fake_result(2.0, 1.0, 4.0, 2.625, 3.05, 3.875)
    result_co2 = _fake_result(5.0, 3.0, 6.0, 4.0, 4.27, 4.525)

    bands = (
        cube_mod.bands_from_result("h2o", feature_h2o, result_h2o)
        + cube_mod.bands_from_result("co2", feature_co2, result_co2)
    )
    path = tmp_path / "epoch00_cube.fits"
    cube_mod.save_epoch_cube_fits(path, bands, small_wcs, epoch_index=0, mjd_min=60000.0, mjd_max=60010.0)

    cube = cube_mod.load_cube(path)
    assert cube["epoch_index"] == 0
    assert cube["mjd_min"] == 60000.0
    assert cube["mjd_max"] == 60010.0
    assert len(cube["bands"]) == 6
    names = [b["name"] for b in cube["bands"]]
    assert names == ["h2o_blue", "h2o_line", "h2o_red", "co2_blue", "co2_line", "co2_red"]
    assert cube["flux_cube"].shape == (6, 4, 4)
    assert np.allclose(cube["flux_cube"][1], 1.0)  # h2o_line
    assert np.allclose(cube["flux_cube"][4], 3.0)  # co2_line


def test_tau_from_cube_matches_hand_computed_value(tmp_path, small_wcs):
    feature = tau_mod.FEATURE_PRESETS["h2o"]
    lam_blue, lam_line, lam_red = 2.625, 3.05, 3.875
    f_blue_val, f_line_val, f_red_val = 2.0, 1.0, 4.0
    result = _fake_result(f_blue_val, f_line_val, f_red_val, lam_blue, lam_line, lam_red)
    bands = cube_mod.bands_from_result("h2o", feature, result)

    path = tmp_path / "epoch00_cube.fits"
    cube_mod.save_epoch_cube_fits(path, bands, small_wcs, epoch_index=0, mjd_min=60000.0, mjd_max=60001.0)
    cube = cube_mod.load_cube(path)

    derived = cube_mod.tau_from_cube(cube, "h2o", kind="absorption")

    frac = (lam_line - lam_blue) / (lam_red - lam_blue)
    f_cont_expected = f_blue_val + (f_red_val - f_blue_val) * frac
    tau_expected = np.log(f_cont_expected / f_line_val)

    assert derived["coverage"] == 1.0
    assert np.allclose(derived["tau"], tau_expected, atol=1e-5)


def test_tau_from_cube_rejects_unknown_feature(tmp_path, small_wcs):
    feature = tau_mod.FEATURE_PRESETS["h2o"]
    result = _fake_result(2.0, 1.0, 4.0, 2.625, 3.05, 3.875)
    bands = cube_mod.bands_from_result("h2o", feature, result)
    path = tmp_path / "epoch00_cube.fits"
    cube_mod.save_epoch_cube_fits(path, bands, small_wcs, epoch_index=0, mjd_min=60000.0, mjd_max=60001.0)
    cube = cube_mod.load_cube(path)

    try:
        cube_mod.tau_from_cube(cube, "co2", kind="absorption")
    except ValueError as exc:
        assert "h2o" in str(exc)
    else:
        raise AssertionError("expected a ValueError for a feature not present in the cube")
