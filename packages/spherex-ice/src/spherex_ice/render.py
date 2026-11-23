"""Saving and rendering 3.05 um H2O ice peak optical depth maps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


def save_tau_fits(
    path: Path,
    result: dict[str, Any],
    ref_wcs: WCS,
    epoch_index: int,
    mjd_min: float,
    mjd_max: float,
    feature_label: str | None = None,
) -> None:
    """Write an epoch's tau map (+ the three flux mosaics it was built from) to FITS."""
    hdr = ref_wcs.to_header()
    if feature_label is not None:
        hdr["FEATURE"] = (feature_label, "Spectral feature this map is of")
    hdr["LAMBDA_C"] = (float(result["lam_line"]), "Feature line center (microns)")
    hdr["LAM_BLUE"] = (float(result["lam_blue"]), "Blue continuum band center (microns)")
    hdr["LAM_RED"] = (float(result["lam_red"]), "Red continuum band center (microns)")
    hdr["KIND"] = (result.get("kind", "absorption"), "absorption or emission (see package docs)")
    # Not "EPOCH": that's a reserved WCS keyword (a deprecated alias for EQUINOX),
    # and astropy.wcs warns about it when parsing this header back.
    hdr["SXEPOCH"] = (int(epoch_index), "Observation epoch index (spherex_ice)")
    hdr["MJD_MIN"] = (float(mjd_min), "First exposure MJD-OBS in this epoch")
    hdr["MJD_MAX"] = (float(mjd_max), "Last exposure MJD-OBS in this epoch")
    hdr["COVERAG"] = (float(result["coverage"]), "Fraction of pixels with a valid TAU-HDU value")

    hdul = fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(data=result["tau"].astype(np.float32), header=hdr.copy(), name="TAU"),
            fits.ImageHDU(data=result["f_blue"].astype(np.float32), header=hdr.copy(), name="F_BLUE"),
            fits.ImageHDU(data=result["f_line"].astype(np.float32), header=hdr.copy(), name="F_LINE"),
            fits.ImageHDU(data=result["f_red"].astype(np.float32), header=hdr.copy(), name="F_RED"),
            fits.ImageHDU(data=result["f_cont"].astype(np.float32), header=hdr.copy(), name="F_CONT"),
            fits.ImageHDU(data=result["w_blue"].astype(np.float32), header=hdr.copy(), name="W_BLUE"),
            fits.ImageHDU(data=result["w_line"].astype(np.float32), header=hdr.copy(), name="W_LINE"),
            fits.ImageHDU(data=result["w_red"].astype(np.float32), header=hdr.copy(), name="W_RED"),
        ]
    )
    hdul.writeto(path, overwrite=True)


def _format_fov(width_deg: float, height_deg: float) -> str:
    if width_deg >= 1.0 / 60.0:
        return f"{width_deg:.2f}° x {height_deg:.2f}°"
    return f"{width_deg * 3600:.1f}\" x {height_deg * 3600:.1f}\""


def fov_deg_from_wcs(wcs: WCS, shape: tuple[int, int]) -> tuple[float, float]:
    ny, nx = shape
    scale = np.sqrt(np.sum(wcs.pixel_scale_matrix**2, axis=0))
    return float(abs(scale[0]) * nx), float(abs(scale[1]) * ny)


def _make_image_axes(fig: Any, wcs: WCS | None, bg_color: str, fg_color: str, top: float = 0.85) -> Any:
    """The main image axes: real RA/Dec tick labels, in decimal degrees, when ``wcs``
    is given (via astropy's WCSAxes, so ``ax.imshow``/``ax.plot`` etc. all keep working
    exactly as on a plain Axes, in the same pixel coordinates) -- no grid lines drawn
    over the image itself, just the axis ticks/labels. Falls back to a bare, tickless
    Axes when ``wcs`` is None (e.g. a caller with no WCS on hand).

    ``top`` is where the axes region ends (figure fraction) -- whatever's above it is
    left for a title/colorbar/legend, same as before this had real tick labels.
    """
    if wcs is not None:
        ax = fig.add_axes([0.14, 0.09, 0.84, top - 0.09], projection=wcs)
        lon, lat = ax.coords[0], ax.coords[1]
        lon.set_axislabel("RA (deg)", color=fg_color, fontsize=8)
        lat.set_axislabel("Dec (deg)", color=fg_color, fontsize=8)
        lon.set_major_formatter("d.dd")
        lat.set_major_formatter("d.dd")
        lon.set_ticklabel(color=fg_color, size=7)
        lat.set_ticklabel(color=fg_color, size=7)
        lon.set_ticks(color=fg_color)
        lat.set_ticks(color=fg_color)
        for sp in ax.spines.values():
            sp.set_edgecolor(fg_color)
            sp.set_linewidth(0.6)
    else:
        ax = fig.add_axes([0.0, 0.0, 1.0, top])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_facecolor(bg_color)
    return ax


def _draw_target_reticle(
    ax: Any, shape: tuple[int, int], color: str = "red", linewidth: float = 1.3
) -> None:
    """A discreet sniper-style reticle pointing at the target: four short ticks with a gap
    at the center, so nothing is drawn over the target pixel itself and it never encloses it.

    The target sits at the exact geometric center of the grid, since every reference WCS in
    this package is built centered on the requested RA/Dec (:func:`spherexstack.mosaic.build_reference_wcs`).
    """
    ny, nx = shape
    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
    size = min(nx, ny)
    gap, length = size * 0.035, size * 0.07
    ax.plot([cx, cx], [cy + gap, cy + gap + length], color=color, linewidth=linewidth, solid_capstyle="butt")
    ax.plot([cx, cx], [cy - gap - length, cy - gap], color=color, linewidth=linewidth, solid_capstyle="butt")
    ax.plot([cx + gap, cx + gap + length], [cy, cy], color=color, linewidth=linewidth, solid_capstyle="butt")
    ax.plot([cx - gap - length, cx - gap], [cy, cy], color=color, linewidth=linewidth, solid_capstyle="butt")


def render_tau_map(
    tau: np.ndarray,
    out_png: Path,
    epoch_index: int,
    mjd_min: float,
    mjd_max: float,
    coverage: float,
    vmax: float = 1.1,
    fov_deg: tuple[float, float] | None = None,
    invert: bool = True,
    mark_target: bool = True,
    title: str = "3.05 um H2O ice peak optical depth (tau)",
    wcs: WCS | None = None,
    cmap: str | Any | None = None,
    reticle_color: str = "red",
) -> None:
    """Render a feature map: brighter = more signal (higher optical depth, or more line
    excess for an emission feature), consistent with Figure 3 of Hora et al. (2026) (a
    dark background with bright, high-signal filaments).

    ``vmax`` is the value that saturates to the brightest color. The default
    of 1.1 here is only a fallback for a caller that doesn't pass one
    (e.g. an all-NaN map); ``spherex_ice.cli`` always computes and passes
    its own, data-driven ``vmax`` (a percentile of that map's own finite
    values -- see its ``--vmax`` help) rather than relying on this default,
    since no fixed scale applies across fields of very different optical
    depth or, for an emission feature, flux. A pixel can come out
    negative (e.g. for an absorption line, F_line measured brighter than the
    interpolated continuum) where the blue/line/red mosaics aren't perfectly
    registered with each other -- usually right next to a compact (stellar)
    source, whose PSF lands on a slightly different pixel in each band since
    each band is built from a different, small subset of exposures/dithers.
    Those pixels are clipped to the bottom of the color scale here rather
    than hidden, so they stay visible as a quality flag; pixels with no
    valid measurement at all use a slightly different shade of the
    background color instead, so the two don't read as the same thing.

    ``cmap``, when given (e.g. a feature preset's own ``"cmap"``, see
    :data:`spherex_ice.tau.FEATURE_PRESETS`) -- a named matplotlib colormap,
    or an actual ``Colormap`` object (for a custom one, e.g. ``tau.H2O_CMAP``)
    -- replaces the default grayscale, and is always shown against
    a black background (these false-color maps already run black-to-bright
    on their own, so ``invert`` -- which only flips the grayscale case
    white-on-black for print legibility -- doesn't apply to them).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if cmap is not None:
        bg_color, fg_color, nodata_color = "black", "white", "#262626"
        cmap_obj = (plt.get_cmap(cmap) if isinstance(cmap, str) else cmap).copy()
    else:
        bg_color = "white" if invert else "black"
        nodata_color = "#dcdcdc" if invert else "#262626"
        fg_color = "black" if invert else "white"
        cmap_obj = plt.get_cmap("gray_r" if invert else "gray").copy()
    cmap_obj.set_bad(color=nodata_color)

    fig = plt.figure(figsize=(5.2, 5.6), dpi=160)
    fig.patch.set_facecolor(bg_color)
    # Top-to-bottom: a title above the colorbar, the colorbar itself (with its
    # own tick labels just below it), then the image filling the rest -- kept
    # as separate bands so none of them overlap regardless of font metrics.
    ax = _make_image_axes(fig, wcs, bg_color, fg_color)
    cax = fig.add_axes([0.12, 0.90, 0.76, 0.035])

    tau_clipped = np.clip(tau, 0.0, vmax)
    tau_display = np.ma.masked_invalid(tau_clipped)
    ax.imshow(tau_display, origin="lower", interpolation="nearest", cmap=cmap_obj, vmin=0.0, vmax=vmax)
    if mark_target:
        _draw_target_reticle(ax, tau.shape, color=reticle_color)

    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=vmax)
    cbar = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_obj), cax=cax, orientation="horizontal"
    )
    cbar.ax.tick_params(axis="x", colors=fg_color, labelsize=8, pad=1, length=3, width=0.8)
    cbar.outline.set_edgecolor(fg_color)
    cbar.outline.set_linewidth(0.6)
    fig.text(0.5, 0.975, title, color=fg_color, fontsize=9, ha="center", va="top")

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
