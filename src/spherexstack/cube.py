"""Build and read (RA, Dec, lambda) spectral cubes.

This is new code (the original scripts never wrote a wavelength-axis FITS
cube; only single-wavelength ``MOSAIC``+``WEIGHT`` FITS and GIF/PNG/MP4
outputs existed). It stacks the per-bin mosaics that
:func:`spherexstack.mosaic.build_lambda_mosaics_streaming` (or
:func:`spherexstack.mosaic.build_lambda_mosaics_legacy`) already computes --
without changing how those arrays are computed -- into a 3D array with a
proper spectral WCS axis, plus a per-slice coverage table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from spherexstack import mosaic


def build_cube_wcs(ref_wcs: WCS, bin_centers: np.ndarray) -> WCS:
    """3D WCS: the ROI's RA/Dec axes plus a linear wavelength axis.

    ``bin_centers`` (in microns) is expected to be evenly spaced (as produced
    by ``numpy.linspace`` upstream); the third axis is built as a linear WCS
    with ``CTYPE3='WAVE'``. Per the FITS WCS standard, astropy normalizes the
    stored ``CRVAL3``/``CDELT3``/``CUNIT3`` to meters (the algorithm's required
    SI unit) when the header is written, even though the values are supplied
    here in microns.
    """
    n_bins = len(bin_centers)
    w = WCS(naxis=3)
    w.wcs.ctype = [ref_wcs.wcs.ctype[0], ref_wcs.wcs.ctype[1], "WAVE"]
    w.wcs.crpix = [ref_wcs.wcs.crpix[0], ref_wcs.wcs.crpix[1], 1.0]
    w.wcs.crval = [ref_wcs.wcs.crval[0], ref_wcs.wcs.crval[1], float(bin_centers[0])]
    cdelt3 = float(bin_centers[1] - bin_centers[0]) if n_bins > 1 else 1.0
    w.wcs.cdelt = [ref_wcs.wcs.cdelt[0], ref_wcs.wcs.cdelt[1], cdelt3]
    w.wcs.cunit = ["deg", "deg", "um"]
    return w


def assemble_cube(
    mosaics: list[np.ndarray], coverages: list[float], weights: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack per-bin mosaic/weight arrays into 3D cubes, validating coverage.

    Returns
    -------
    cube : ndarray, shape (n_bins, ny, nx)
    weight_cube : ndarray, shape (n_bins, ny, nx)
    coverage : ndarray, shape (n_bins,)

    Raises
    ------
    ValueError
        If no bin has any coverage at all (the cube would be entirely NaN).
    """
    coverage = np.array(coverages, dtype=float)
    if not np.any(coverage > 0):
        raise ValueError(
            "Zero coverage in every wavelength bin -- the cube would be all-NaN. "
            "Check that the FITS list, ROI and wavelength range actually overlap."
        )
    cube = np.stack(mosaics, axis=0)
    weight_cube = np.stack(weights, axis=0)
    return cube, weight_cube, coverage


def build_cube(
    resolved: list[Path],
    bad_bits: list[str],
    clean: bool,
    ref_wcs: WCS,
    ref_shape: tuple[int, int],
    bin_centers: np.ndarray,
    dlambda: float,
    match_background: bool,
    min_overlap_px: int,
    weighting: str = "mean",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a (n_bins, ny, nx) cube by stacking the streaming mosaic builder's bins.

    Thin wrapper around :func:`spherexstack.mosaic.build_lambda_mosaics_streaming`:
    the per-bin arrays it already computes are stacked into 3D arrays, nothing
    about the accumulation itself changes.

    Returns
    -------
    cube : ndarray, shape (n_bins, ny, nx)
    weight_cube : ndarray, shape (n_bins, ny, nx)
    coverage : ndarray, shape (n_bins,)

    Raises
    ------
    ValueError
        If no bin has any coverage at all (the cube would be entirely NaN).
    """
    mosaics, coverages, weights = mosaic.build_lambda_mosaics_streaming(
        resolved,
        bad_bits,
        clean,
        ref_wcs,
        ref_shape,
        bin_centers,
        dlambda,
        match_background,
        min_overlap_px,
        weighting=weighting,
    )
    return assemble_cube(mosaics, coverages, weights)


def save_cube_fits(
    path: Path,
    cube: np.ndarray,
    weight_cube: np.ndarray,
    ref_wcs: WCS,
    bin_centers: np.ndarray,
    dlambda: float,
    coverage: np.ndarray,
    extra_header: dict[str, Any] | None = None,
    bunit: str = "MJy/sr",
) -> None:
    """Write a spectral cube to FITS: ``CUBE`` + ``WEIGHT`` image HDUs and a ``COVERAGE`` table.

    HDUs
    ----
    PRIMARY : empty
    CUBE : float32, shape (n_bins, ny, nx), 3D WCS (RA, Dec, WAVE[um])
    WEIGHT : float32, same shape, total weight per voxel (exposure count under ``weighting="mean"``)
    COVERAGE : binary table, one row per wavelength bin, columns
        ``LAMBDA_UM`` and ``COVERAGE_FRAC``
    """
    cube_wcs = build_cube_wcs(ref_wcs, bin_centers)
    hdr = cube_wcs.to_header()
    hdr["DLAMBDA_UM"] = (float(dlambda), "Half-width of the synthetic band per bin (microns)")
    if extra_header:
        for k, v in extra_header.items():
            hdr[k] = v

    cube_hdr = hdr.copy()
    cube_hdr["BUNIT"] = (bunit, "Surface brightness (native) or flux density (--units mjy)")
    weight_hdr = hdr.copy()
    weight_hdr["BUNIT"] = ("", "Total weight per voxel (exposure count under weighting=mean)")

    cube_hdu = fits.ImageHDU(data=cube.astype(np.float32), header=cube_hdr, name="CUBE")
    weight_hdu = fits.ImageHDU(data=weight_cube.astype(np.float32), header=weight_hdr, name="WEIGHT")
    coverage_hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="LAMBDA_UM", format="D", array=np.asarray(bin_centers, dtype=float)),
            fits.Column(name="COVERAGE_FRAC", format="D", array=np.asarray(coverage, dtype=float)),
        ],
        name="COVERAGE",
    )
    fits.HDUList([fits.PrimaryHDU(), cube_hdu, weight_hdu, coverage_hdu]).writeto(path, overwrite=True)


def load_cube_fits(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, WCS, np.ndarray, np.ndarray, float | None]:
    """Inverse of :func:`save_cube_fits`.

    Returns
    -------
    cube : ndarray, shape (n_bins, ny, nx)
    weight_cube : ndarray, shape (n_bins, ny, nx)
    wcs : astropy.wcs.WCS
        3D WCS of the cube.
    bin_centers : ndarray, shape (n_bins,)
    coverage : ndarray, shape (n_bins,)
    dlambda_um : float or None
    """
    with fits.open(path, memmap=False) as hdul:
        cube = np.asarray(hdul["CUBE"].data, dtype=float)
        weight_cube = np.asarray(hdul["WEIGHT"].data, dtype=float)
        wcs = WCS(hdul["CUBE"].header)
        dlambda_um = hdul["CUBE"].header.get("DLAMBDA_UM")
        table = hdul["COVERAGE"].data
        bin_centers = np.asarray(table["LAMBDA_UM"], dtype=float)
        coverage = np.asarray(table["COVERAGE_FRAC"], dtype=float)
    cube = np.where(weight_cube > 0, cube, np.nan)
    return cube, weight_cube, wcs, bin_centers, coverage, dlambda_um
