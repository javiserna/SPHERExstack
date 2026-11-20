"""Shared pytest fixtures: a tiny synthetic SPHEREx-Level-2-like FITS file builder.

The synthetic file reproduces the HDU structure the pipeline actually reads
(``IMAGE``, ``FLAGS``, ``VARIANCE``, ``WCS-WAVE``) including a real FITS Paper
III "-TAB" lookup table wired up as the alternate WCS key ``'W'``, the same
mechanism :func:`spherexstack.io.load_roi_frame` uses via
``WCS(header, hdul, key="W")``.

Simplification versus the real instrument data: here wavelength is modeled as
a function of pixel axis 1 (column/X) only, via a 1-D lookup table, instead of
SPHEREx's real (weaker) 2-D row-and-column dependence -- this exercises the
same code path (the per-pixel wavelength lookup through the 'W' alternate
WCS) without needing to reverse-engineer the exact multi-axis table layout of
the real instrument files, which is not documented in the source scripts
(see INVENTORY.md item 1).
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
    variance: np.ndarray | None = None,
    lam0: float = 3.0,
    dlam_per_px: float = 0.02,
    bw0: float = 0.01,
) -> Path:
    """Write a synthetic Level-2-like FITS file at ``path``.

    Parameters
    ----------
    wcs : astropy.wcs.WCS
        The exposure's own sky WCS (becomes the unkeyed/primary WCS of the
        ``IMAGE`` HDU, as read back by ``WCS(header)`` with no key).
    shape : (ny, nx)
    image, flags, variance : ndarray, optional
        Defaults to zeros, zeros (int), and ones respectively.
    lam0, dlam_per_px : float
        Per-pixel wavelength (microns) is ``lam0 + dlam_per_px * x`` (x = 0-indexed
        column). Use ``dlam_per_px=0`` for a spatially flat wavelength.
    """
    ny, nx = shape
    if image is None:
        image = np.zeros(shape, dtype=float)
    if flags is None:
        flags = np.zeros(shape, dtype=np.int32)
    if variance is None:
        variance = np.ones(shape, dtype=float)

    nxg, nyg = nx, ny
    x_grid = np.arange(1, nxg + 1, dtype=float)
    y_grid = np.arange(1, nyg + 1, dtype=float)
    values = np.zeros((nyg, nxg, 2), dtype=float)
    for ix in range(nxg):
        values[:, ix, 0] = lam0 + dlam_per_px * ix
        values[:, ix, 1] = bw0
    lambda_1d = values[0, :, 0].copy()  # (nxg,)

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

    primary = fits.PrimaryHDU()
    image_hdu = fits.ImageHDU(data=image.astype(np.float64), header=hdr, name="IMAGE")
    flags_hdu = fits.ImageHDU(data=flags.astype(np.int32), name="FLAGS")
    variance_hdu = fits.ImageHDU(data=variance.astype(np.float64), name="VARIANCE")
    fits.HDUList([primary, image_hdu, flags_hdu, variance_hdu, wave_hdu]).writeto(path, overwrite=True)
    return path


@pytest.fixture
def level2_fits_factory(tmp_path):
    """Returns a function ``factory(name, wcs, shape, **kwargs) -> Path`` writing a fixture FITS."""

    def factory(name: str, wcs: WCS, shape: tuple[int, int], **kwargs) -> Path:
        return make_level2_fits(tmp_path / name, wcs, shape, **kwargs)

    return factory
