"""Per-pixel SPHEREx spectrum extraction at one sky position, from raw exposure cutouts.

Every tau/flux map this package builds comes from reprojecting just the handful of
narrow bands one ``--feature`` needs (see :mod:`spherex_ice.tau`). The *full* spectrum
at a given sky position -- every detector/epoch's one pixel nearest that position,
each with its own flux and its own per-pixel wavelength (SPHEREx's linear variable
filter shifts the wavelength solution with detector position, hence ``WCS-WAVE``,
see :mod:`spherexstack.io`) -- isn't something any map already holds, but every raw
cutout that overlaps the position has what's needed for its own one point of it.
Used by :mod:`spherex_ice.viewer` for the interactive click-to-see-spectrum tool
(``spherex-ice-inspect``).

Two extraction modes are offered: a single-pixel spectrum (:func:`extract_spectrum_at_position`,
the default) and a background-subtracted circular-aperture spectrum
(:func:`extract_aperture_spectrum_at_position`), for when a single pixel's noise is too high to
judge a feature by (an extended, diffuse source, say). The aperture mode follows the same
aperture-plus-annulus recipe as IRSA's own SPHEREx mosaic-cube tutorial
(https://caltech-ipac.github.io/irsa-tutorials/spherex-mosaic/), including its MJy/sr -> mJy
conversion (see :mod:`spherexstack.units`) so a sum over pixels is a meaningful flux.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import SigmaClip
from astropy.utils.exceptions import AstropyWarning
from astropy.wcs import WCS
from photutils.aperture import ApertureStats, CircularAnnulus, CircularAperture, aperture_photometry
from spherexstack import io as sx_io
from spherexstack import units as sx_units


def list_cutout_files(cutouts_dir: Path) -> list[Path]:
    """Every ``.fits`` exposure cutout under ``cutouts_dir`` (as left by ``spherex-ice-get``
    or ``spherex-ice-epochs``), sorted for reproducible output ordering."""
    return sorted(Path(cutouts_dir).glob("*.fits"))


def extract_spectrum_at_position(
    ra_deg: float, dec_deg: float, cutout_paths: list[Path], bad_bits: list[str]
) -> dict[str, np.ndarray | list[str]]:
    """The SPHEREx spectrum at one sky position: one (wavelength, flux) point from each
    cutout whose footprint covers it, sorted by wavelength.

    For each file, only the ``IMAGE`` header (cheap) is read to test whether the
    position's pixel falls inside that exposure; the per-pixel flux, FLAGS value, and
    ``WCS-WAVE`` wavelength are only computed for files that actually overlap, since
    that's normally a small fraction of the files in ``cutout_paths``. A pixel flagged
    bad (any bit in ``bad_bits``, same set used for the tau maps -- see
    :data:`spherexstack.io.DEFAULT_BAD_BITS`) is skipped rather than returned.

    Returns
    -------
    dict with keys:
        ``lam`` : wavelength (microns), ascending.
        ``flux`` : surface brightness (MJy/sr) at that pixel, same order as ``lam``.
        ``files`` : source cutout filename for each point, same order.
    """
    target = SkyCoord(ra_deg, dec_deg, unit="deg")
    lam_list: list[float] = []
    flux_list: list[float] = []
    src_list: list[str] = []

    for fp in cutout_paths:
        try:
            with fits.open(fp, memmap=True) as hdul:
                if "IMAGE" not in hdul or "WCS-WAVE" not in hdul:
                    continue
                header = hdul["IMAGE"].header
                ny, nx = int(header["NAXIS2"]), int(header["NAXIS1"])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=AstropyWarning)
                    w = WCS(header)
                x, y = w.world_to_pixel(target)
                xi, yi = int(round(float(x))), int(round(float(y)))
                if not (0 <= xi < nx and 0 <= yi < ny):
                    continue

                flux = float(hdul["IMAGE"].data[yi, xi])
                if not np.isfinite(flux):
                    continue
                if "FLAGS" in hdul:
                    flagval = int(hdul["FLAGS"].data[yi, xi])
                    if sx_io.bad_pixel_mask_from_flags(np.array([[flagval]]), bad_bits)[0, 0]:
                        continue

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=AstropyWarning)
                    w_wave = WCS(header, hdul, key="W")
                lam, _bandwidth = w_wave.all_pix2world([xi], [yi], 0)
                lam_val = float(lam[0])
                if not np.isfinite(lam_val):
                    continue

                lam_list.append(lam_val)
                flux_list.append(flux)
                src_list.append(fp.name)
        except Exception:  # noqa: BLE001 - one bad/unreadable cutout shouldn't kill the click
            continue

    lam_arr = np.asarray(lam_list, dtype=float)
    order = np.argsort(lam_arr)
    return {
        "lam": lam_arr[order],
        "flux": np.asarray(flux_list, dtype=float)[order],
        "files": [src_list[i] for i in order],
    }


def extract_aperture_spectrum_at_position(
    ra_deg: float,
    dec_deg: float,
    cutout_paths: list[Path],
    bad_bits: list[str],
    aperture_px: float,
    annulus_in_px: float | None = None,
    annulus_out_px: float | None = None,
    min_aperture_coverage: float = 0.5,
) -> dict[str, np.ndarray | list[str]]:
    """Background-subtracted circular-aperture spectrum at one sky position.

    One point per exposure, like :func:`extract_spectrum_at_position`, but each flux value
    is a sum over a circular aperture of radius ``aperture_px`` (SPHEREx native pixels),
    with the local background estimated from the mean of a ``CircularAnnulus``
    (``annulus_in_px``-``annulus_out_px``, default 2x/3x the aperture radius) and
    subtracted off -- the same aperture-plus-annulus recipe IRSA's own SPHEREx
    mosaic-cube tutorial uses
    (https://caltech-ipac.github.io/irsa-tutorials/spherex-mosaic/). Flagged/non-finite
    pixels are excluded from both the aperture sum and the background estimate rather
    than zeroed. A sum of MJy/sr values has no direct physical meaning, so each exposure's
    image is first converted to mJy/pixel (see :mod:`spherexstack.units`) before summing,
    making the returned flux a genuine aperture flux density, in mJy.

    An exposure is skipped if less than ``min_aperture_coverage`` (default 50%) of the
    aperture's own area is covered by good, in-bounds pixels (near the cutout's edge, say),
    or if the annulus has no usable background pixels at all.

    Returns
    -------
    dict with keys:
        ``lam`` : wavelength (microns), ascending.
        ``flux`` : background-subtracted aperture flux (mJy), same order as ``lam``.
        ``files`` : source cutout filename for each point, same order.
    """
    if annulus_in_px is None:
        annulus_in_px = 2.0 * aperture_px
    if annulus_out_px is None:
        annulus_out_px = 3.0 * aperture_px

    target = SkyCoord(ra_deg, dec_deg, unit="deg")
    lam_list: list[float] = []
    flux_list: list[float] = []
    src_list: list[str] = []
    aperture_area_full = np.pi * aperture_px**2

    for fp in cutout_paths:
        try:
            with fits.open(fp, memmap=False) as hdul:
                if "IMAGE" not in hdul or "WCS-WAVE" not in hdul:
                    continue
                header = hdul["IMAGE"].header
                ny, nx = int(header["NAXIS2"]), int(header["NAXIS1"])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=AstropyWarning)
                    w = WCS(header)
                xf, yf = (float(v) for v in w.world_to_pixel(target))
                xi, yi = int(round(xf)), int(round(yf))
                if not (0 <= xi < nx and 0 <= yi < ny):
                    continue

                data = np.asarray(hdul["IMAGE"].data, dtype=float)
                bad_mask = ~np.isfinite(data)
                if "FLAGS" in hdul:
                    flags = np.asarray(hdul["FLAGS"].data)
                    if flags.shape == data.shape:
                        bad_mask |= sx_io.bad_pixel_mask_from_flags(flags, bad_bits)

                data_mjy, _bunit = sx_units.convert_units(data, "mjy", w)

                aperture = CircularAperture((xf, yf), r=aperture_px)
                annulus = CircularAnnulus((xf, yf), r_in=annulus_in_px, r_out=annulus_out_px)

                covered_area = aperture.area_overlap(data_mjy, mask=bad_mask)
                if covered_area < min_aperture_coverage * aperture_area_full:
                    continue

                bkg_stats = ApertureStats(
                    data_mjy, annulus, mask=bad_mask, sigma_clip=SigmaClip(sigma=3.0, maxiters=10)
                )
                if bkg_stats.mean is None or not np.isfinite(bkg_stats.mean):
                    continue

                phot = aperture_photometry(data_mjy, aperture, mask=bad_mask)
                flux = float(phot["aperture_sum"][0]) - float(bkg_stats.mean) * covered_area
                if not np.isfinite(flux):
                    continue

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=AstropyWarning)
                    w_wave = WCS(header, hdul, key="W")
                lam, _bandwidth = w_wave.all_pix2world([xi], [yi], 0)
                lam_val = float(lam[0])
                if not np.isfinite(lam_val):
                    continue

                lam_list.append(lam_val)
                flux_list.append(flux)
                src_list.append(fp.name)
        except Exception:  # noqa: BLE001 - one bad/unreadable cutout shouldn't kill the click
            continue

    lam_arr = np.asarray(lam_list, dtype=float)
    order = np.argsort(lam_arr)
    return {
        "lam": lam_arr[order],
        "flux": np.asarray(flux_list, dtype=float)[order],
        "files": [src_list[i] for i in order],
    }
