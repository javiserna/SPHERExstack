"""FITS loading, FLAGS-based bad-pixel masking, and per-pixel wavelength lookup.

This module only touches science data (no plotting). It reads the HDUs written by
SPHEREx Level 2 cutouts (``IMAGE``, ``FLAGS``, ``WCS-WAVE``, optionally ``VARIANCE``)
and turns them into plain NumPy arrays plus an ``astropy.wcs.WCS``.

Wavelength lookup uses the ``WCS-WAVE`` binary table through the alternate WCS key
``'W'`` (FITS Paper III "WAVE-TAB" standard), which gives each pixel its own
wavelength -- more accurate than assuming a constant wavelength per detector row,
since SPHEREx's linear variable filter also shifts slightly with column.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from astropy import log as _astropy_log
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.utils.exceptions import AstropyWarning
from astropy.wcs import WCS

# Real SPHEREx Level 2 headers carry SIP coefficients without a "-SIP" CTYPE suffix;
# astropy.wcs applies them anyway (correctly) but logs an INFO-level notice about it
# on every single WCS construction -- not a `warnings.warn`, so the usual
# `warnings.simplefilter("ignore", ...)` around WCS(...) calls does not catch it. With
# thousands of exposures this buries every other message, so drop astropy's own
# below-WARNING log output globally rather than per call site.
_astropy_log.setLevel("WARNING")

# ---------------------------------------------------------------------------
# FLAGS quality mask (HDU 'FLAGS'). Bit numbers and their meaning come from the
# HIERARCH MP_* header keywords of the SPHEREx Level 2 pipeline.
# ---------------------------------------------------------------------------
FLAG_BITS = {
    "transient": 0,  # transient detected during the SUR (e.g. cosmic ray)
    "overflow": 1,  # saturation
    "sur_error": 2,  # onboard processing error
    "phantom": 4,  # ghost pixel (not real sky)
    "reference": 5,  # reference pixel (not real sky)
    "nonfunc": 6,  # permanently dead
    "dichroic": 7,  # low efficiency from the dichroic
    "missing": 9,  # data lost onboard
    "hot": 10,  # known hot pixel
    "cold": 11,  # anomalously low signal
    "fullsample": 12,  # has full-sample history (not itself a defect)
    "phanmiss": 14,  # ghost correction not applied
    "nonlinear": 15,  # nonlinearity correction not reliable
    "persist": 17,  # persistence (afterglow of a bright source)
    "outlier": 19,  # flagged by the "Detect Outlier Pixels" step
    "source": 21,  # mapped to a known source (NOT a defect)
}

# Bits that correspond to defects/artifacts (excludes 'source', 'reference',
# 'phantom', 'fullsample', 'dichroic', 'cold', which are not "hot pixel" type defects).
DEFAULT_BAD_BITS = [
    "transient",
    "overflow",
    "sur_error",
    "nonfunc",
    "missing",
    "hot",
    "nonlinear",
    "persist",
    "outlier",
]


def bad_pixel_mask_from_flags(flags: np.ndarray, bad_bits: list[str]) -> np.ndarray:
    """Boolean mask, True where a pixel is flagged by any of ``bad_bits``."""
    mask_val = 0
    for name in bad_bits:
        mask_val |= 1 << FLAG_BITS[name]
    return (flags.astype(np.int64) & mask_val) != 0


def clean_hot_pixels(
    data: np.ndarray, bad_mask: np.ndarray, max_fill_passes: int = 4
) -> tuple[np.ndarray, int]:
    """In-paint pixels marked bad by interpolating from their neighbors.

    Uses growing Gaussian kernels so that small connected clusters of bad
    pixels (e.g. persistence trails) can also be filled, not just isolated
    hot pixels.

    Parameters
    ----------
    data : ndarray
        2D image.
    bad_mask : ndarray
        Boolean mask, same shape as ``data``, True where a pixel is bad.
    max_fill_passes : int, default 4
        Maximum number of growing-kernel interpolation passes.

    Returns
    -------
    cleaned : ndarray
        Copy of ``data`` with bad pixels replaced.
    n_bad : int
        Number of pixels that were replaced.
    """
    cleaned = np.asarray(data, dtype=float).copy()
    n_bad = int(bad_mask.sum())
    if n_bad == 0:
        return cleaned, 0

    cleaned[bad_mask] = np.nan
    stddev = 1.0
    for _ in range(max_fill_passes):
        if not np.isnan(cleaned).any():
            break
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=AstropyWarning)
            cleaned = interpolate_replace_nans(cleaned, Gaussian2DKernel(x_stddev=stddev))
        stddev *= 2.0

    remaining = np.isnan(cleaned)
    if remaining.any():
        cleaned[remaining] = np.nanmedian(cleaned)

    return cleaned, n_bad


def read_list(list_path: Path) -> list[Path]:
    """Read a newline-separated list of FITS paths (``#`` comments and blank lines skipped)."""
    lines = []
    with open(list_path, encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            lines.append(Path(s))
    return lines


def resolve_list_paths(list_path: Path) -> list[Path]:
    """Read ``list_path`` and resolve each entry relative to the list file's directory."""
    files = read_list(list_path)
    if not files:
        raise ValueError(f"{list_path} is empty (or only has comments).")
    base_dir = list_path.parent
    resolved = []
    for p in files:
        rp = (base_dir / p).resolve() if not p.is_absolute() else p
        if not rp.exists():
            raise FileNotFoundError(f"Does not exist: {rp}")
        resolved.append(rp)
    return resolved


def get_image_and_wcs(fits_path: Path, image_ext: str = "IMAGE") -> tuple[np.ndarray, WCS]:
    """Read the 2D image and its WCS from HDU ``image_ext`` (falls back to the first 2D HDU)."""
    with fits.open(fits_path, memmap=False) as hdul:
        if image_ext not in hdul:
            for hdu in hdul:
                if hdu.data is None:
                    continue
                data = np.asarray(hdu.data)
                if data.ndim == 2:
                    return data, WCS(hdu.header)
                if data.ndim == 3 and data.shape[0] == 1:
                    return data[0], WCS(hdu.header)
            raise ValueError(f"No 2D image found in {fits_path}")

        hdu = hdul[image_ext]
        data = np.asarray(hdu.data)
        if data.ndim == 3 and data.shape[0] == 1:
            data = data[0]
        if data.ndim != 2:
            raise ValueError(f"HDU {image_ext} is not 2D in {fits_path} (shape={data.shape})")
        return data, WCS(hdu.header)


def get_flags(fits_path: Path, flags_ext: str = "FLAGS") -> np.ndarray | None:
    """Read the ``FLAGS`` quality-mask HDU (same shape as ``IMAGE``), or None if absent."""
    with fits.open(fits_path, memmap=False) as hdul:
        if flags_ext not in hdul:
            return None
        data = np.asarray(hdul[flags_ext].data)
        if data.ndim == 3 and data.shape[0] == 1:
            data = data[0]
        return data


def get_variance(fits_path: Path, variance_ext: str = "VARIANCE") -> np.ndarray | None:
    """Read the ``VARIANCE`` HDU (same shape as ``IMAGE``), or None if absent.

    Used for ``--weighting ivar``. Requires the FITS file to have been
    downloaded with ``--keep-hdus`` including ``VARIANCE``.
    """
    with fits.open(fits_path, memmap=False) as hdul:
        if variance_ext not in hdul:
            return None
        data = np.asarray(hdul[variance_ext].data)
        if data.ndim == 3 and data.shape[0] == 1:
            data = data[0]
        return np.asarray(data, dtype=float)


def central_lambda_um(fits_path: Path) -> float:
    """Representative wavelength (microns) of a FITS file, from ``WCS-WAVE``/``VALUES[...,0]``."""
    with fits.open(fits_path, memmap=False) as hdul:
        if "WCS-WAVE" not in hdul:
            return np.nan
        tab = hdul["WCS-WAVE"].data
        if tab is None or len(tab) == 0:
            return np.nan
        values = np.array(tab["VALUES"][0], dtype=float)  # (nyg, nxg, 2)
        lam = values[..., 0]
        return float(np.nanmedian(lam))


def lambda_axis_from_wcs_wave(
    fits_path: Path, ny_out: int, y_pix: np.ndarray | None = None
) -> np.ndarray:
    """Wavelength (microns) as a function of detector row Y, from ``WCS-WAVE``.

    Uses ``VALUES[..., 0]`` (component 0) collapsed over X, interpolated onto
    ``y_pix`` (absolute detector Y). If ``y_pix`` is None, assumes
    ``0..ny_out-1``.
    """
    with fits.open(fits_path, memmap=False) as hdul:
        if "WCS-WAVE" not in hdul:
            return np.full(ny_out, np.nan, dtype=float)

        tab = hdul["WCS-WAVE"].data
        if tab is None or len(tab) == 0:
            return np.full(ny_out, np.nan, dtype=float)

        y_grid = np.array(tab["Y"][0], dtype=float)  # (nyg,)
        values = np.array(tab["VALUES"][0], dtype=float)  # (nyg, nxg, 2)
        lam_grid = values[..., 0]  # (nyg, nxg) microns
        lam_y_grid = np.nanmedian(lam_grid, axis=1)  # (nyg,) collapsed over X

        if y_pix is None:
            y_pix = np.arange(ny_out, dtype=float)
        else:
            y_pix = np.asarray(y_pix, dtype=float)

        order = np.argsort(y_grid)
        y_grid_sorted = y_grid[order]
        lam_y_grid_sorted = lam_y_grid[order]

        return np.interp(y_pix, y_grid_sorted, lam_y_grid_sorted, left=np.nan, right=np.nan)


def quick_lambda_range(fp: Path) -> tuple[float, float] | None:
    """Wavelength range (microns) from the (small) ``WCS-WAVE`` table only, without loading IMAGE.

    Used to set the bin centers of a wavelength sweep before the accumulation pass.
    """
    with fits.open(fp, memmap=True) as hdul:
        if "WCS-WAVE" not in hdul:
            return None
        tab = hdul["WCS-WAVE"].data
        if tab is None or len(tab) == 0:
            return None
        values = np.array(tab["VALUES"][0], dtype=float)
        lam = values[..., 0]
        lam = lam[np.isfinite(lam)]
        if lam.size == 0:
            return None
        return float(lam.min()), float(lam.max())


def load_roi_frame(
    fp: Path, bad_bits: list[str], clean: bool, need_variance: bool = False
) -> tuple[np.ndarray, WCS, np.ndarray, np.ndarray | None] | None:
    """Load one ROI-mode cutout (already server-side cropped) plus its per-pixel wavelength map.

    Unlike :func:`load_target_cutout_and_lambda`, the file is already a small
    cutout, so no further ``Cutout2D`` is needed: the full image is read as-is
    and its per-pixel wavelength map is computed.

    Parameters
    ----------
    fp : Path
        FITS file path.
    bad_bits : list of str
        FLAGS bit names to treat as defects.
    clean : bool
        If True, in-paint bad pixels found via FLAGS.
    need_variance : bool, default False
        If True, also read and return the ``VARIANCE`` HDU (for ``ivar`` weighting).

    Returns
    -------
    tuple or None
        ``(data, wcs, lam, variance)`` with ``variance`` set to None when
        ``need_variance`` is False or the HDU is absent, or None if
        ``IMAGE``/``WCS-WAVE`` is missing or the image has no finite data.
    """
    with fits.open(fp, memmap=False) as hdul:
        if "IMAGE" not in hdul or "WCS-WAVE" not in hdul:
            return None
        img = np.asarray(hdul["IMAGE"].data, dtype=float)
        header = hdul["IMAGE"].header

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=AstropyWarning)
            w = WCS(header)
            w_wave = WCS(header, hdul, key="W")

        variance = None
        if need_variance:
            if "VARIANCE" in hdul:
                variance = np.asarray(hdul["VARIANCE"].data, dtype=float)
            else:
                variance = None

        if clean and "FLAGS" in hdul:
            flags = np.asarray(hdul["FLAGS"].data)
            if flags.shape == img.shape:
                bad_mask = bad_pixel_mask_from_flags(flags, bad_bits)
                img, _n_bad = clean_hot_pixels(img, bad_mask)

    if img.size == 0 or not np.any(np.isfinite(img)):
        return None

    ny, nx = img.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    lam, _bandwidth = w_wave.all_pix2world(xx.ravel(), yy.ravel(), 0)
    lam = lam.reshape(ny, nx)

    return img, w, lam, variance


def load_target_cutout_and_lambda(
    fp: Path, target: SkyCoord, cutout_size: int, bad_bits: list[str], clean: bool
) -> tuple[np.ndarray, WCS, np.ndarray] | None:
    """Load a pixel-sized cutout around a point target, plus its per-pixel wavelength map.

    Reads ``IMAGE`` + ``FLAGS`` + ``WCS-WAVE``, optionally cleans hot pixels,
    crops a ``cutout_size`` x ``cutout_size`` window around ``target``, and
    computes the wavelength map of that window via the alternate WCS ``'W'``.

    Returns
    -------
    tuple or None
        ``(data, wcs, lam)`` of the cutout, or None if there is no overlap or no data.
    """
    with fits.open(fp, memmap=False) as hdul:
        if "IMAGE" not in hdul or "WCS-WAVE" not in hdul:
            return None
        img = np.asarray(hdul["IMAGE"].data, dtype=float)
        header = hdul["IMAGE"].header

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=AstropyWarning)
            w = WCS(header)
            w_wave = WCS(header, hdul, key="W")

        if clean and "FLAGS" in hdul:
            flags = np.asarray(hdul["FLAGS"].data)
            if flags.shape == img.shape:
                bad_mask = bad_pixel_mask_from_flags(flags, bad_bits)
                img, _n_bad = clean_hot_pixels(img, bad_mask)

    try:
        co = Cutout2D(
            img, position=target, size=(cutout_size, cutout_size), wcs=w, mode="partial",
            fill_value=np.nan,
        )
    except Exception:
        return None
    if co.data.size == 0 or not np.any(np.isfinite(co.data)):
        return None

    ysl = co.slices_original[0]
    xsl = co.slices_original[1]
    yy, xx = np.mgrid[ysl.start : ysl.stop, xsl.start : xsl.stop]
    lam, _bandwidth = w_wave.all_pix2world(xx.ravel(), yy.ravel(), 0)
    lam = lam.reshape(yy.shape)

    return co.data, co.wcs, lam
