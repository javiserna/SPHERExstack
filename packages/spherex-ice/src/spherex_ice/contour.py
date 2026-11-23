"""Contour products: a reusable sky mask at a chosen signal threshold.

For any feature map this package builds (H2O/CO2/CO optical depth, or a PAH/
recombination-line emission excess), it's often more useful to have "where
is this above some meaningful level" as its own product than to eyeball the
grayscale map: a FITS mask (with the same WCS as the map it came from, so it
can be reprojected, regridded, or overlaid onto anything else) for use in
other analyses, and a PNG showing exactly where that boundary falls on the
map itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from spherex_ice.render import _draw_target_reticle, _format_fov, _make_image_axes


def compute_contour_mask(tau: np.ndarray, level: float) -> np.ndarray:
    """Boolean mask, True where ``tau`` is finite and at or above ``level``."""
    return np.isfinite(tau) & (tau >= level)


def default_contour_level(tau: np.ndarray, kind: str) -> float:
    """A sensible default threshold when ``--contour_level`` isn't given.

    For an absorption (optical depth) feature: 0.2, a commonly-used "clearly
    present" ice threshold. For an emission feature (flux units, no single
    conventional cutoff): 30% of this epoch's own robust (95th-percentile)
    signal level, so it adapts to how bright that particular line/field is.
    """
    if kind == "absorption":
        return 0.2
    finite = tau[np.isfinite(tau)]
    if finite.size == 0:
        return 0.0
    return 0.3 * float(np.percentile(finite, 95))


def save_contour_fits(
    path: Path,
    mask: np.ndarray,
    ref_wcs: WCS,
    level: float,
    feature_label: str,
    kind: str,
    units: str,
    epoch_index: int | str,
    mjd_min: float,
    mjd_max: float,
) -> None:
    """Write the contour mask (1 = at/above ``level``, 0 = not) to FITS, with the same
    WCS as the map it was computed from, so it can be used directly in other analyses.

    ``epoch_index`` is normally an int (one epoch); pass ``"median"`` for a
    contour built from :func:`spherex_ice.combine.combine_epochs_median`'s output.
    """
    hdr = ref_wcs.to_header()
    hdr["FEATURE"] = (feature_label, "Spectral feature this contour is drawn from")
    hdr["KIND"] = (kind, "absorption or emission")
    hdr["LEVEL"] = (float(level), "Contour threshold, in UNITS")
    hdr["UNITS"] = (units, "Units of LEVEL (and of the source tau/flux map)")
    hdr["BUNIT"] = "1 = at/above LEVEL, 0 = below or no data"
    # Not "EPOCH": that's a reserved WCS keyword (a deprecated alias for EQUINOX),
    # and astropy.wcs warns about it when parsing this header back.
    hdr["SXEPOCH"] = (str(epoch_index), "Observation epoch index, or 'median' across several")
    hdr["MJD_MIN"] = (float(mjd_min), "First exposure MJD-OBS in this epoch")
    hdr["MJD_MAX"] = (float(mjd_max), "Last exposure MJD-OBS in this epoch")
    fits.PrimaryHDU(data=mask.astype(np.uint8), header=hdr).writeto(path, overwrite=True)


def render_contour_map(
    tau: np.ndarray,
    level: float,
    out_png: Path,
    epoch_index: int,
    mjd_min: float,
    mjd_max: float,
    coverage: float,
    units: str,
    vmax: float,
    fov_deg: tuple[float, float] | None = None,
    invert: bool = True,
    mark_target: bool = True,
    title: str = "",
    wcs: WCS | None = None,
    cmap: str | Any | None = None,
    reticle_color: str = "red",
) -> None:
    """Render the same feature map as :func:`spherex_ice.render.render_tau_map`
    (grayscale, or ``cmap`` when given -- see that function), with a contour line
    drawn at ``level`` on top, so the boundary is visible in context.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if cmap is not None:
        bg_color, fg_color, nodata_color = "black", "white", "#262626"
        contour_color = "#39ff14"
        cmap_obj = (plt.get_cmap(cmap) if isinstance(cmap, str) else cmap).copy()
    else:
        bg_color = "white" if invert else "black"
        nodata_color = "#dcdcdc" if invert else "#262626"
        fg_color = "black" if invert else "white"
        contour_color = "#1fb8ff" if invert else "#ffd21f"
        cmap_obj = plt.get_cmap("gray_r" if invert else "gray").copy()
    cmap_obj.set_bad(color=nodata_color)

    fig = plt.figure(figsize=(5.2, 5.6), dpi=160)
    fig.patch.set_facecolor(bg_color)
    ax = _make_image_axes(fig, wcs, bg_color, fg_color)

    vmin = min(0.0, level)
    tau_clipped = np.clip(tau, vmin, vmax)
    tau_display = np.ma.masked_invalid(tau_clipped)
    ax.imshow(tau_display, origin="lower", interpolation="nearest", cmap=cmap_obj, vmin=vmin, vmax=vmax)

    # NaN -> far below any real level, so data gaps never produce a spurious contour.
    tau_for_contour = np.where(np.isfinite(tau), tau, vmin - 1e6)
    ax.contour(tau_for_contour, levels=[level], colors=[contour_color], linewidths=1.1)
    if mark_target:
        _draw_target_reticle(ax, tau.shape, color=reticle_color)

    fig.text(0.5, 0.975, title, color=fg_color, fontsize=9, ha="center", va="top")
    fig.text(
        0.5, 0.925, f"contour at {level:.3g} {units}",
        color=contour_color, fontsize=8.5, ha="center", va="top",
    )

    date_label = f"MJD {mjd_min:.2f}-{mjd_max:.2f} (span {mjd_max - mjd_min:.1f} d)"
    ax.text(
        0.02, 0.02, f"{date_label}   coverage {100 * coverage:.0f}%",
        transform=ax.transAxes, color=fg_color, fontsize=9, ha="left", va="bottom",
        bbox=dict(facecolor=bg_color, alpha=0.4, edgecolor="none", pad=2),
    )
    if fov_deg is not None:
        ax.text(
            0.98, 0.98, f"FOV {_format_fov(*fov_deg)}",
            transform=ax.transAxes, color=fg_color, fontsize=9, ha="right", va="top",
            bbox=dict(facecolor=bg_color, alpha=0.4, edgecolor="none", pad=2),
        )
    fig.savefig(out_png, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
