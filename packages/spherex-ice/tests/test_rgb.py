from pathlib import Path

import numpy as np
from astropy.io import fits

from spherex_ice import rgb as rgb_mod


def _write_tau_fits(path: Path, tau: np.ndarray, wcs) -> None:
    hdr = wcs.to_header()
    hdr["FEATURE"] = "test feature"
    hdr["KIND"] = "absorption"
    tau_hdu = fits.ImageHDU(data=tau.astype(np.float32), header=hdr, name="TAU")
    fits.HDUList([fits.PrimaryHDU(), tau_hdu]).writeto(path, overwrite=True)


def test_compose_rgb_scales_each_channel_independently(tmp_path, small_wcs):
    shape = (6, 6)
    red = np.full(shape, 2.0)
    green = np.full(shape, 4.0)

    red_path = tmp_path / "red.fits"
    green_path = tmp_path / "green.fits"
    _write_tau_fits(red_path, red, small_wcs)
    _write_tau_fits(green_path, green, small_wcs)

    result = rgb_mod.compose_rgb(red_path, green_path, None, red_vmax=2.0, green_vmax=8.0)

    assert result["shape"] == shape
    assert np.allclose(result["rgb"][..., 0], 1.0)  # red: 2.0 clipped/scaled by vmax=2.0 -> 1.0
    assert np.allclose(result["rgb"][..., 1], 0.5)  # green: 4.0 / vmax=8.0 -> 0.5
    assert np.allclose(result["rgb"][..., 2], 0.0)  # blue: no file given -> all zero


def test_compose_rgb_rejects_mismatched_shapes(tmp_path, small_wcs):
    _write_tau_fits(tmp_path / "red.fits", np.zeros((6, 6)), small_wcs)
    _write_tau_fits(tmp_path / "green.fits", np.zeros((8, 8)), small_wcs)

    try:
        rgb_mod.compose_rgb(tmp_path / "red.fits", tmp_path / "green.fits", None)
    except ValueError as exc:
        assert "don't match" in str(exc)
    else:
        raise AssertionError("expected a ValueError for mismatched channel shapes")
