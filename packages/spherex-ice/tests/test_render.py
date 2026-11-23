import numpy as np
from astropy.io import fits

from spherex_ice import render as render_mod


def test_save_tau_fits_stores_feature_label(tmp_path, small_wcs):
    shape = (4, 4)
    result = {
        "tau": np.zeros(shape), "coverage": 1.0, "kind": "absorption",
        "f_blue": np.zeros(shape), "f_line": np.zeros(shape),
        "f_red": np.zeros(shape), "f_cont": np.zeros(shape),
        "w_blue": np.ones(shape), "w_line": np.ones(shape), "w_red": np.ones(shape),
        "lam_blue": 2.625, "lam_line": 3.05, "lam_red": 3.875,
    }
    path = tmp_path / "tau.fits"

    render_mod.save_tau_fits(path, result, small_wcs, 0, 60000.0, 60001.0, feature_label="3.05 um H2O ice")

    with fits.open(path) as hdul:
        assert hdul["TAU"].header["FEATURE"] == "3.05 um H2O ice"
