"""Synthetic SPHEREx-Level-2-like FITS file builder, adapted from spherexstack's test suite
(same WCS-WAVE "-TAB" lookup table mechanism), with an added MJD-OBS header for epoch tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS


def make_level2_fits(
    path: Path,
    wcs: WCS,
    shape: tuple[int, int],
    image: np.ndarray | None = None,
    flags: np.ndarray | None = None,
    lam0: float = 3.0,
    dlam_per_px: float = 0.0,
    bw0: float = 0.01,
    mjd_obs: float = 60000.0,
) -> Path:
    ny, nx = shape
    if image is None:
        image = np.ones(shape, dtype=float)
    if flags is None:
        flags = np.zeros(shape, dtype=np.int32)

    nxg, nyg = nx, ny
    x_grid = np.arange(1, nxg + 1, dtype=float)
    y_grid = np.arange(1, nyg + 1, dtype=float)
    values = np.zeros((nyg, nxg, 2), dtype=float)
    for ix in range(nxg):
        values[:, ix, 0] = lam0 + dlam_per_px * ix
        values[:, ix, 1] = bw0
    lambda_1d = values[0, :, 0].copy()

    col_x = fits.Column(name="X", format=f"{nxg}D", array=x_grid[None, :])
    col_y = fits.Column(name="Y", format=f"{nyg}D", array=y_grid[None, :])
    col_v = fits.Column(
        name="VALUES", format=f"{nyg * nxg * 2}D", dim=f"(2,{nxg},{nyg})", array=values[None, ...]
    )
    col_l1d = fits.Column(name="LAMBDA_1D", format=f"{nxg}D", array=lambda_1d[None, :])
    wave_hdu = fits.BinTableHDU.from_columns([col_x, col_y, col_v, col_l1d], name="WCS-WAVE")
    wave_hdu.header["EXTVER"] = 1

    hdr = wcs.to_header()
    hdr["WCSNAMEW"] = "WAVE-TAB-TEST"
    hdr["CTYPE1W"] = "WAVE-TAB"
    hdr["CUNIT1W"] = "um"
    hdr["PS1_0W"] = "WCS-WAVE"
    hdr["PS1_1W"] = "LAMBDA_1D"
    hdr["PS1_2W"] = "X"
    hdr["PV1_1W"] = 1
    hdr["CRPIX1W"] = 1.0
    hdr["CRVAL1W"] = 1.0
    hdr["CDELT1W"] = 1.0
    hdr["CTYPE2W"] = "LINEAR"
    hdr["CRPIX2W"] = 1.0
    hdr["CRVAL2W"] = 0.0
    hdr["CDELT2W"] = 1.0
    hdr["MJD-OBS"] = mjd_obs

    primary = fits.PrimaryHDU()
    image_hdu = fits.ImageHDU(data=image.astype(np.float64), header=hdr, name="IMAGE")
    flags_hdu = fits.ImageHDU(data=flags.astype(np.int32), name="FLAGS")
    fits.HDUList([primary, image_hdu, flags_hdu, wave_hdu]).writeto(path, overwrite=True)
    return path


@pytest.fixture
def level2_fits_factory(tmp_path):
    def factory(name: str, wcs: WCS, shape: tuple[int, int], **kwargs) -> Path:
        return make_level2_fits(tmp_path / name, wcs, shape, **kwargs)

    return factory


@pytest.fixture
def small_wcs():
    w = WCS(naxis=2)
    w.wcs.crpix = [5.0, 5.0]
    w.wcs.cdelt = [-6.15 / 3600.0, 6.15 / 3600.0]
    w.wcs.crval = [10.0, 20.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w
