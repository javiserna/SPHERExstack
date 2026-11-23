"""Per-pixel 3.05 um H2O ice peak optical depth, following Hora et al. (2026, ApJ 1001:165).

That paper's ice absorption maps (Section 3) are built per *sky pixel*, not
per point source: the continuum at the line center ``F_lambda_c`` is
estimated by linearly interpolating between wavelengths adjacent to the
line, and the peak optical depth is

    tau = ln(F_lambda_c / F_lambda)

with ``F_lambda`` the flux at the ice absorption line center. This module
reuses :mod:`spherexstack.mosaic`'s reprojection/background-matching
machinery to build three monochromatic mosaics (a blue continuum band, the
3.05 um H2O line itself, and a red continuum band) on one common pixel grid,
then combines them pixel-by-pixel into a peak-tau map.

Default continuum bands (2.50-2.75 and 3.75-4.00 um) match the ones already
validated for single-source SPHEREx H2O spectroscopy in this project's
``ice_mantle_fitter_multiepoch.py`` (``--cont_blue_lo/hi``, ``--cont_red_lo/hi``).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
from astropy.wcs import WCS
from matplotlib.colors import LinearSegmentedColormap
from spherexstack import io as sio
from spherexstack import mosaic as mosaic_mod

H2O_LAMBDA_UM = 3.05


def default_vmax_percentile(kind: str) -> float:
    """Percentile of a map's own finite values used as its default display ``vmax`` (see
    ``spherex_ice.cli``'s ``--vmax`` help and :mod:`spherex_ice.viewer`): 99.9 for an
    absorption (tau) feature, where a handful of very high pixels next to bright/compact
    sources are expected and shouldn't by themselves wash out the extended structure;
    95 for an emission feature (flux excess, no fixed physical scale to begin with).
    """
    return 99.9 if kind == "absorption" else 95.0

KIND_CHOICES = ("absorption", "emission")

# A black -> saturated blue -> white scale for H2O ice (picked over matplotlib's
# built-in "bone"/"Blues_r", both too washed-out, to match the colored maps in
# Hora et al. 2026's own figures).
H2O_CMAP = LinearSegmentedColormap.from_list(
    "spherex_ice_h2o", ["black", "#08306b", "#2171b5", "#6baed6", "#c6dbef", "white"]
)

# Named presets for SPHEREx spectral features covered in Hora et al. (2026),
# so a line other than H2O is just --feature co2 away rather than four
# separate band flags. Continuum bands for co2/co reuse the ranges already
# validated for single-source SPHEREx spectroscopy in this project's
# ice_mantle_fitter_multiepoch.py (--co2_cont_*/--co_cont_*); the PAH ranges
# match spherexstack's own single-wavelength PAH mosaic (3.24-3.34 um line,
# continuum at 2.77/3.85 um).
#
# H2O and CO2 and CO are ABSORPTION features (ice): the reported metric is
# the peak optical depth, tau = ln(F_continuum / F_line). PAH at 3.3 um is
# an EMISSION feature (aromatic hydrocarbon dust), so the natural metric
# there is continuum-subtracted flux excess, F_line - F_continuum, not a
# log-ratio -- see ``kind`` below and :func:`build_epoch_tau_map`.
FEATURE_PRESETS: dict[str, dict[str, Any]] = {
    "h2o": {
        "label": "3.05 um H2O ice",
        "symbol": r"$\tau_{peak}^{\rm{H_2O}}$",
        "kind": "absorption",
        "lam_line": 3.05, "dlambda_line": 0.05,
        "cont_blue_lo": 2.50, "cont_blue_hi": 2.75,
        "cont_red_lo": 3.75, "cont_red_hi": 4.00,
        "contour_level": 0.5,
        "cmap": H2O_CMAP,
    },
    "co2": {
        "label": "4.27 um CO2 ice",
        "symbol": r"$\tau_{peak}^{\rm{CO_2}}$",
        "kind": "absorption",
        "lam_line": 4.27, "dlambda_line": 0.05,
        "cont_blue_lo": 3.90, "cont_blue_hi": 4.10,
        "cont_red_lo": 4.45, "cont_red_hi": 4.60,
        "contour_level": 0.2,
    },
    "co": {
        "label": "4.67 um CO ice",
        "symbol": r"$\tau_{peak}^{\rm{CO}}$",
        "kind": "absorption",
        "lam_line": 4.67, "dlambda_line": 0.05,
        "cont_blue_lo": 4.40, "cont_blue_hi": 4.55,
        "cont_red_lo": 4.85, "cont_red_hi": 5.00,
        "contour_level": 0.2,
    },
    "pah": {
        "label": "3.3 um PAH",
        "symbol": r"$\Delta F_{\rm{PAH}}$",
        "kind": "emission",
        "lam_line": 3.29, "dlambda_line": 0.05,
        "cont_blue_lo": 2.67, "cont_blue_hi": 2.87,
        "cont_red_lo": 3.75, "cont_red_hi": 3.95,
        "cmap": "hot",
        "reticle_color": "cyan",
    },
    # Hydrogen recombination lines (Paschen/Brackett series) -- also EMISSION
    # features, same F_line - F_continuum metric as PAH. Line centers are
    # standard vacuum wavelengths (accurate to far better than SPHEREx's own
    # resolution element here); unlike h2o/co2/co/pah, these continuum bands
    # are NOT independently validated against real SPHEREx data in this
    # project -- they're a reasonable, clear-of-other-known-lines starting
    # point (narrower +/-0.025 um line window, since an unresolved
    # recombination line has no broad wings to worry about diluting the
    # way the ice bands do) -- check the result for your field and adjust
    # --cont_*/--dlambda_line if needed. Note Br beta's line sits inside
    # h2o's blue continuum band (2.50-2.75 um): a strong Br beta source
    # will put a little excess emission into an H2O map's continuum there.
    "paalpha": {
        "label": "1.8751 um Pa-alpha",
        "symbol": r"$\Delta F_{\rm{Pa\alpha}}$",
        "kind": "emission",
        "lam_line": 1.8751, "dlambda_line": 0.025,
        "cont_blue_lo": 1.78, "cont_blue_hi": 1.82,
        "cont_red_lo": 1.93, "cont_red_hi": 1.97,
    },
    "pabeta": {
        "label": "1.2818 um Pa-beta",
        "symbol": r"$\Delta F_{\rm{Pa\beta}}$",
        "kind": "emission",
        "lam_line": 1.2818, "dlambda_line": 0.025,
        "cont_blue_lo": 1.19, "cont_blue_hi": 1.23,
        "cont_red_lo": 1.33, "cont_red_hi": 1.37,
    },
    "padelta": {
        "label": "1.0052 um Pa-delta",
        "symbol": r"$\Delta F_{\rm{Pa\delta}}$",
        "kind": "emission",
        "lam_line": 1.0052, "dlambda_line": 0.025,
        "cont_blue_lo": 0.94, "cont_blue_hi": 0.98,
        "cont_red_lo": 1.04, "cont_red_hi": 1.08,
    },
    "bralpha": {
        "label": "4.0512 um Br-alpha",
        "symbol": r"$\Delta F_{\rm{Br\alpha}}$",
        "kind": "emission",
        "lam_line": 4.0512, "dlambda_line": 0.025,
        "cont_blue_lo": 3.95, "cont_blue_hi": 3.99,
        "cont_red_lo": 4.10, "cont_red_hi": 4.14,
    },
    "brbeta": {
        "label": "2.6259 um Br-beta",
        "symbol": r"$\Delta F_{\rm{Br\beta}}$",
        "kind": "emission",
        "lam_line": 2.6259, "dlambda_line": 0.025,
        "cont_blue_lo": 2.50, "cont_blue_hi": 2.56,
        "cont_red_lo": 2.69, "cont_red_hi": 2.75,
    },
}

# How far (as a multiplicative factor) a single exposure's own median in-band
# flux may sit from the group's median-of-medians before it's treated as an
# outlier and dropped. Found necessary in practice: with only a handful of
# exposures contributing to a band (common for a single epoch), one exposure
# with an anomalous flux level (a calibration glitch, stray light, ...) has
# no redundancy to be down-weighted by background-matching -- instead its
# offset propagates through the whole mosaic via the shared correction solve,
# turning tau negative over the *entire* field, not just that exposure's own
# footprint. Caught on a real field: one such exposure out of only 3
# contributing to the H2O line band shifted the field's median tau from
# +0.34 to -0.11 (62% of pixels negative) until it was excluded.
OUTLIER_FACTOR = 4.0


def _exposure_band_median(fp: Path, lam_lo: float, lam_hi: float, bad_bits: list[str]) -> float | None:
    """Median flux of the pixels of ``fp`` whose own wavelength falls in [lam_lo, lam_hi],
    excluding FLAGS-bad pixels. None if nothing in that file falls in the band.
    """
    with fits.open(fp, memmap=False) as hdul:
        if "IMAGE" not in hdul or "WCS-WAVE" not in hdul:
            return None
        img = np.asarray(hdul["IMAGE"].data, dtype=float)
        header = hdul["IMAGE"].header
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=AstropyWarning)
            w_wave = WCS(header, hdul, key="W")
        ny, nx = img.shape
        yy, xx = np.mgrid[0:ny, 0:nx]
        lam, _bw = w_wave.all_pix2world(xx.ravel(), yy.ravel(), 0)
        lam = lam.reshape(ny, nx)
        mask = (lam >= lam_lo) & (lam <= lam_hi)
        if "FLAGS" in hdul:
            flags = np.asarray(hdul["FLAGS"].data)
            if flags.shape == img.shape:
                mask &= ~sio.bad_pixel_mask_from_flags(flags, bad_bits)
        vals = img[mask]
        vals = vals[np.isfinite(vals)]
        return float(np.median(vals)) if vals.size else None


def _filter_outlier_exposures(
    resolved: list[Path], lam_lo: float, lam_hi: float, bad_bits: list[str]
) -> list[Path]:
    """Drop any exposure whose own in-band median flux is off by more than
    :data:`OUTLIER_FACTOR` from the group's median-of-medians (either way).
    Needs at least 3 contributing exposures to have a meaningful "group" to
    compare against; with fewer, there is no way to tell an outlier from a
    real source, so nothing is dropped.
    """
    medians = {}
    for fp in resolved:
        m = _exposure_band_median(fp, lam_lo, lam_hi, bad_bits)
        if m is not None:
            medians[fp] = m
    if len(medians) < 3:
        return resolved

    ref = float(np.median(list(medians.values())))
    if ref <= 0:
        return resolved

    kept = []
    for fp in resolved:
        m = medians.get(fp)
        if m is not None and (m > OUTLIER_FACTOR * ref or m < ref / OUTLIER_FACTOR):
            print(
                f"[WARN] Excluding {fp.name}: its median flux in {lam_lo:.3f}-{lam_hi:.3f} um "
                f"({m:.4g}) is more than {OUTLIER_FACTOR:.0f}x off the other "
                f"{len(medians) - 1} contributing exposures' median ({ref:.4g}) -- likely a "
                "calibration glitch or stray light, not real sky signal."
            )
            continue
        kept.append(fp)
    return kept


def build_epoch_tau_map(
    resolved: list[Path],
    bad_bits: list[str],
    clean: bool,
    ref_wcs: WCS,
    ref_shape: tuple[int, int],
    lam_line: float = H2O_LAMBDA_UM,
    dlambda_line: float = 0.05,
    cont_blue_lo: float = 2.50,
    cont_blue_hi: float = 2.75,
    cont_red_lo: float = 3.75,
    cont_red_hi: float = 4.00,
    match_background: bool = True,
    min_overlap_px: int = 20,
    weighting: str = "mean",
    reject_outliers: bool = True,
    min_weight: float = 1.0,
    kind: str = "absorption",
) -> dict[str, Any]:
    """Build a per-pixel spectral feature map for one epoch's exposures, by linearly
    interpolating a continuum between a blue and a red band and comparing it to the
    line band -- the technique Hora et al. (2026, ApJ 1001:165, Section 3) use for
    their widefield H2O/CO2 ice maps. Despite the name, this works for any line with
    the same three-band shape, not just H2O: see :data:`FEATURE_PRESETS`.

    ``resolved`` should be just the exposures belonging to a single epoch
    (see :mod:`spherex_ice.epochs`), already reprojected onto ``ref_wcs``.

    ``kind``: ``"absorption"`` (default) reports the peak optical depth,
    ``tau = ln(F_continuum / F_line)``, appropriate for an ice absorption
    line (H2O, CO2, CO) where the line sits below the continuum. ``"emission"``
    reports the continuum-subtracted flux excess, ``F_line - F_continuum``,
    appropriate for an emission feature (e.g. the 3.3 um PAH band) where the
    line sits above it -- a log-ratio isn't meaningful there since F_line
    being brighter than F_continuum is the expected, physical case, not an
    error. Either way the result is stored under the ``"tau"`` key, so the
    two most-used quantities in this module share one calculation path; the
    FITS/PNG output label what it actually is.

    ``reject_outliers`` (default True): before building each band's mosaic,
    drop any exposure whose own median in-band flux is off by more than
    :data:`OUTLIER_FACTOR` from its peers (see that constant's docstring for
    why this matters with few exposures per band). Each band can end up
    dropping a different subset of ``resolved``; the three bands are still
    built in one combined pass over their union (see ``bin_sources`` in
    :func:`spherexstack.mosaic.build_lambda_mosaics_streaming`) rather than
    each band separately re-reading/re-cleaning every exposure from scratch.

    ``min_weight`` (default 1.0, i.e. no extra masking): a pixel is only kept
    if all three bands (blue/line/red) have at least this much per-pixel
    weight there -- under ``weighting="mean"`` that weight is simply the
    number of exposures that landed on that pixel. A pixel backed by only
    one exposure in a band is exactly where leftover band-to-band
    registration mismatches (a star's PSF landing on a slightly different
    pixel in each band) show up as an isolated bright/dark pixel pair,
    since there is no second sample to average it down -- raising this to
    2 trades some coverage for pixels you can trust more.

    Returns a dict with ``tau`` (2D ndarray, NaN where there is no valid or
    sufficiently-supported measurement), ``coverage`` (fraction of finite
    tau pixels), the intermediate flux mosaics/coverages, and the per-pixel
    weight map for each band, for diagnostics.
    """
    if kind not in KIND_CHOICES:
        raise ValueError(f"Unknown kind '{kind}', expected one of {KIND_CHOICES}.")
    lam_blue = 0.5 * (cont_blue_lo + cont_blue_hi)
    dlambda_blue = 0.5 * (cont_blue_hi - cont_blue_lo)
    lam_red = 0.5 * (cont_red_lo + cont_red_hi)
    dlambda_red = 0.5 * (cont_red_hi - cont_red_lo)

    bin_centers = np.array([lam_blue, lam_line, lam_red])
    dlambdas = np.array([dlambda_blue, dlambda_line, dlambda_red])
    bin_sources = [
        set(
            _filter_outlier_exposures(resolved, lam0 - db, lam0 + db, bad_bits)
            if reject_outliers
            else resolved
        )
        for lam0, db in zip(bin_centers, dlambdas, strict=True)
    ]
    # The union of what each band actually uses: a file only in one band's
    # set (e.g. one outlier-rejected from "line" but fine for "blue"/"red")
    # is still read/cleaned/wavelength-mapped once here and simply skipped
    # for the bin(s) it was rejected from, rather than every bin re-reading
    # every file from scratch in its own separate pass.
    union = [fp for fp in resolved if any(fp in s for s in bin_sources)]

    print(
        f"[INFO] Blue {cont_blue_lo:.3f}-{cont_blue_hi:.3f} um, "
        f"line {lam_line - dlambda_line:.3f}-{lam_line + dlambda_line:.3f} um, "
        f"red {cont_red_lo:.3f}-{cont_red_hi:.3f} um (one combined pass over {len(union)} exposures)..."
    )
    mosaics, coverages, weights = mosaic_mod.build_lambda_mosaics_streaming(
        union,
        bad_bits,
        clean=clean,
        ref_wcs=ref_wcs,
        ref_shape=ref_shape,
        bin_centers=bin_centers,
        dlambda=dlambdas,
        match_background=match_background,
        min_overlap_px=min_overlap_px,
        weighting=weighting,
        bin_sources=bin_sources,
    )
    (f_blue, f_line, f_red) = mosaics
    (cov_blue, cov_line, cov_red) = coverages
    (w_blue, w_line, w_red) = weights

    # Linear interpolation of the continuum flux at the line center, between
    # the blue and red continuum bands -- Hora et al. 2026, Section 3.
    frac = (lam_line - lam_blue) / (lam_red - lam_blue)
    f_cont = f_blue + (f_red - f_blue) * frac

    enough_support = (w_blue >= min_weight) & (w_line >= min_weight) & (w_red >= min_weight)
    if kind == "absorption":
        valid = np.isfinite(f_cont) & np.isfinite(f_line) & (f_cont > 0) & (f_line > 0) & enough_support
        with np.errstate(invalid="ignore", divide="ignore"):
            tau = np.log(f_cont / f_line)
    else:
        valid = np.isfinite(f_cont) & np.isfinite(f_line) & enough_support
        tau = f_line - f_cont
    tau = np.where(valid, tau, np.nan)
    coverage = float(np.isfinite(tau).sum()) / tau.size

    return {
        "tau": tau,
        "kind": kind,
        "coverage": coverage,
        "f_blue": f_blue,
        "f_line": f_line,
        "f_red": f_red,
        "f_cont": f_cont,
        "w_blue": w_blue,
        "w_line": w_line,
        "w_red": w_red,
        "cov_blue": cov_blue,
        "cov_line": cov_line,
        "cov_red": cov_red,
        "lam_blue": lam_blue,
        "lam_line": lam_line,
        "lam_red": lam_red,
    }
