"""Combine several epochs' feature maps into one, by taking the per-pixel median
across whichever epochs have a valid value there, rather than keeping each
epoch's map separate.

This is a genuinely different product from a single epoch's map, not just a
cleaner-looking average: it's more robust to the kind of single-epoch artifact
this project has already run into in practice (one outlier exposure skewing a
whole epoch, a band-to-band registration mismatch near a star) -- as long as
fewer than half the combined epochs are affected at a given pixel, the median
isn't. The tradeoff is that it can no longer show real time variability, which
is exactly what keeping epochs separate is for -- use both: epochs to check
whether a feature is stable or varying, the median to see the field's overall,
most-trustworthy structure.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


def combine_epochs_median(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-pixel median of ``tau`` across epochs, ignoring NaNs.

    Parameters
    ----------
    results : list of dict
        Each a :func:`spherex_ice.tau.build_epoch_tau_map` result (at least
        two needed for the median to differ from a single epoch's own map).

    Returns
    -------
    dict with ``tau`` (the per-pixel median), ``n_epochs`` (per-pixel count of
    how many of the given epochs had a valid value there -- a pixel covered by
    only one epoch simply gets that epoch's own value), and ``coverage``.
    """
    if not results:
        raise ValueError("Need at least one epoch's result to combine.")
    stack = np.stack([r["tau"] for r in results], axis=0)
    n_epochs = np.sum(np.isfinite(stack), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # "All-NaN slice encountered"
        tau = np.nanmedian(stack, axis=0)
    tau = np.where(n_epochs > 0, tau, np.nan)
    coverage = float(np.isfinite(tau).sum()) / tau.size
    return {"tau": tau, "n_epochs": n_epochs.astype(np.int16), "coverage": coverage}


def save_median_fits(
    path: Path,
    result: dict[str, Any],
    ref_wcs: WCS,
    feature_label: str,
    kind: str,
    units: str,
    epoch_indices: list[int],
    mjd_min: float,
    mjd_max: float,
) -> None:
    """Write the median-combined map (+ its per-pixel epoch-count map) to FITS."""
    hdr = ref_wcs.to_header()
    hdr["FEATURE"] = (feature_label, "Spectral feature this map is of")
    hdr["KIND"] = (kind, "absorption or emission")
    hdr["UNITS"] = (units, "Units of the TAU HDU")
    hdr["N_EPOCH"] = (len(epoch_indices), "Number of epochs combined")
    # Not "EPOCHS": astropy.wcs reads a trailing capital letter on a recognized WCS
    # keyword (here "EPOCH", a deprecated alias for EQUINOX) as an alternate-WCS
    # version code, and warns when it can't make sense of it as one.
    hdr["SXEPOCHS"] = (",".join(str(i) for i in epoch_indices), "Epoch indices combined")
    hdr["MJD_MIN"] = (float(mjd_min), "First exposure MJD-OBS across all combined epochs")
    hdr["MJD_MAX"] = (float(mjd_max), "Last exposure MJD-OBS across all combined epochs")
    hdr["COVERAG"] = (float(result["coverage"]), "Fraction of pixels with a valid TAU-HDU value")

    hdul = fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(data=result["tau"].astype(np.float32), header=hdr.copy(), name="TAU"),
            fits.ImageHDU(data=result["n_epochs"], header=hdr.copy(), name="N_EPOCHS"),
        ]
    )
    hdul.writeto(path, overwrite=True)
