"""GIF/MP4/PNG rendering.

This is the only module that needs the plotting dependencies (``matplotlib``,
``imageio``, ``pillow``) -- installable via the ``spherexstack[animate]``
extra. Building a mosaic or a cube (:mod:`spherexstack.mosaic`,
:mod:`spherexstack.cube`) never requires them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import imageio.v2 as imageio
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from PIL import Image
except ImportError as exc:  # pragma: no cover - exercised only without the extra installed
    raise ImportError(
        "spherexstack.animate needs the plotting extras. Install with "
        "'pip install spherexstack[animate]'."
    ) from exc


def robust_limits(images: list[np.ndarray], p_lo: float = 1.0, p_hi: float = 99.0) -> tuple[float, float]:
    """Global ``(vmin, vmax)`` from percentiles over all finite pixels of all images.

    Each image is subsampled to at most 200,000 finite pixels before the
    percentile is taken, for speed. Falls back to min/max if the percentiles
    coincide or are not finite.
    """
    vals = []
    for im in images:
        x = im[np.isfinite(im)]
        if x.size:
            if x.size > 200_000:
                idx = np.random.choice(x.size, size=200_000, replace=False)
                x = x[idx]
            vals.append(x)
    if not vals:
        return (0.0, 1.0)
    allv = np.concatenate(vals)
    vmin = np.percentile(allv, p_lo)
    vmax = np.percentile(allv, p_hi)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.nanmin(allv))
        vmax = float(np.nanmax(allv))
        if vmin == vmax:
            vmax = vmin + 1.0
    return float(vmin), float(vmax)


def render_rgb_composite(
    mosaic_r: np.ndarray,
    mosaic_g: np.ndarray,
    mosaic_b: np.ndarray,
    out_png: Path,
    wavelengths_um: tuple[float, float, float],
    p_lo: float = 1.0,
    p_hi: float = 99.0,
    fov_deg: tuple[float, float] | None = None,
) -> None:
    """Compose three single-wavelength mosaics into one RGB image.

    Each channel is stretched independently with its own robust ``(p_lo,
    p_hi)`` percentile limits, since different wavelengths can have very
    different flux scales -- there is no shared colorbar to keep consistent
    here, unlike :func:`render_lambda_slice`. A pixel with no coverage in a
    given channel (NaN) renders as 0 in that channel, not as a hole in the
    image; a pixel with no coverage in all three channels is black.
    """
    channels = []
    for m in (mosaic_r, mosaic_g, mosaic_b):
        vmin, vmax = robust_limits([m], p_lo=p_lo, p_hi=p_hi)
        x = np.asarray(m, dtype=float)
        x_clip = np.clip(x, vmin, vmax)
        denom = (vmax - vmin) if (vmax > vmin) else 1.0
        chan = (x_clip - vmin) / denom
        channels.append(np.where(np.isfinite(x), chan, 0.0))

    rgb_img = np.stack(channels, axis=-1)

    fig = plt.figure(figsize=(5.2, 5.2), dpi=160)
    fig.patch.set_facecolor("black")
    ax = fig.add_axes([0.00, 0.00, 1.00, 1.00])
    ax.set_facecolor("black")
    for sp in ax.spines.values():
        sp.set_visible(False)

    ax.imshow(rgb_img, origin="lower", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])

    lam_r, lam_g, lam_b = wavelengths_um
    ax.text(
        0.02,
        0.02,
        rf"R {lam_r:.2f}  G {lam_g:.2f}  B {lam_b:.2f} $\mu$m",
        transform=ax.transAxes,
        color="white",
        fontsize=8,
        ha="left",
        va="bottom",
        bbox=dict(facecolor="black", alpha=0.4, edgecolor="none", pad=2),
    )
    if fov_deg is not None:
        ax.text(
            0.98,
            0.02,
            _format_fov(*fov_deg),
            transform=ax.transAxes,
            color="white",
            fontsize=8,
            ha="right",
            va="bottom",
            bbox=dict(facecolor="black", alpha=0.4, edgecolor="none", pad=2),
        )

    fig.savefig(out_png, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)


def _format_fov(width_deg: float, height_deg: float) -> str:
    """Human-readable field-of-view size, in whichever unit reads best at that scale."""
    if max(width_deg, height_deg) >= 1.0:
        return f"FOV {width_deg:.2f}° x {height_deg:.2f}°"
    width_arcmin, height_arcmin = width_deg * 60.0, height_deg * 60.0
    if max(width_arcmin, height_arcmin) >= 1.0:
        return f"FOV {width_arcmin:.1f}' x {height_arcmin:.1f}'"
    width_arcsec, height_arcsec = width_deg * 3600.0, height_deg * 3600.0
    return f'FOV {width_arcsec:.0f}" x {height_arcsec:.0f}"'


def fov_deg_from_wcs(wcs, shape: tuple[int, int]) -> tuple[float, float]:
    """Field of view ``(width_deg, height_deg)`` of a 2D image grid from its WCS.

    Uses the celestial sub-WCS (so this also works on a cube's 3D WCS, which
    carries a spectral axis alongside RA/Dec) and the pixel scale at the
    reference point; accurate for the simple, unrotated TAN grids this
    package builds.
    """
    from astropy.wcs.utils import proj_plane_pixel_scales

    ny, nx = shape
    scales = proj_plane_pixel_scales(wcs.celestial)
    return float(abs(scales[0]) * nx), float(abs(scales[1]) * ny)


def render_lambda_slice(
    mosaic: np.ndarray,
    out_png: Path,
    vmin: float,
    vmax: float,
    lam_center: float,
    dlambda: float,
    coverage_frac: float,
    lam_bar_min: float = 0.7,
    lam_bar_max: float = 5.0,
    cmap_name: str = "rainbow",
    fov_deg: tuple[float, float] | None = None,
) -> None:
    """Render one monochromatic mosaic slice, tinted by its wavelength, with a lambda colorbar.

    ``fov_deg``, if given as ``(width_deg, height_deg)``, is printed in the
    bottom-right corner in whichever unit (degrees/arcmin/arcsec) best suits
    that size.
    """
    x = np.asarray(mosaic, dtype=float)
    x_clip = np.clip(x, vmin, vmax)
    denom = (vmax - vmin) if (vmax > vmin) else 1.0
    inten = (x_clip - vmin) / denom
    mask = np.isfinite(x)
    inten = np.where(mask, inten, 0.0)

    cmap = cm.get_cmap(cmap_name)
    t = np.clip((lam_center - lam_bar_min) / (lam_bar_max - lam_bar_min), 0.0, 1.0)
    color = np.array(cmap(t)[:3])

    ny, nx = inten.shape
    rgb_img = np.zeros((ny, nx, 3), dtype=float)
    rgb_img[mask] = inten[mask, None] * color[None, :]

    fig = plt.figure(figsize=(5.2, 5.2), dpi=160)
    fig.patch.set_facecolor("black")
    ax = fig.add_axes([0.00, 0.00, 1.00, 0.955])
    cax = fig.add_axes([0.12, 0.962, 0.76, 0.016])
    ax.set_facecolor("black")
    for sp in ax.spines.values():
        sp.set_visible(False)
    cax.set_facecolor("black")
    for sp in cax.spines.values():
        sp.set_visible(False)

    ax.imshow(rgb_img, origin="lower", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])

    ax.text(
        0.02,
        0.02,
        f"{lam_center:.3f} um   coverage {100 * coverage_frac:.0f}%",
        transform=ax.transAxes,
        color="white",
        fontsize=8,
        ha="left",
        va="bottom",
        bbox=dict(facecolor="black", alpha=0.4, edgecolor="none", pad=2),
    )

    if fov_deg is not None:
        ax.text(
            0.98,
            0.02,
            _format_fov(*fov_deg),
            transform=ax.transAxes,
            color="white",
            fontsize=8,
            ha="right",
            va="bottom",
            bbox=dict(facecolor="black", alpha=0.4, edgecolor="none", pad=2),
        )

    norm = mcolors.Normalize(vmin=lam_bar_min, vmax=lam_bar_max)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    tick_vals = [0.7, 1.0, 2.0, 3.0, 4.0, 5.0]
    cbar.set_ticks(tick_vals)
    # Only the last tick spells out the unit (microns); the others keep their plain
    # one-decimal labels to avoid cluttering the bar.
    tick_labels = [f"{v:.1f}" for v in tick_vals[:-1]] + [rf"{tick_vals[-1]:.0f} $\mu$m"]
    cbar.set_ticklabels(tick_labels)
    cbar.ax.set_xlim(lam_bar_min, lam_bar_max)

    # Marker on the colorbar: thin lines at [lam_center-dlambda, lam_center+dlambda] used
    # in this bin, plus a solid line at lam_center, to show at a glance where this frame
    # falls within the full wavelength range. No shaded fill: a translucent overlay here
    # would wash out the gradient's true color under the marker, and since the marker
    # moves every frame, that made the fixed colorbar look like it was "shifting" between
    # frames even though the gradient itself never changes.
    lo = max(lam_center - dlambda, lam_bar_min)
    hi = min(lam_center + dlambda, lam_bar_max)
    cbar.ax.axvline(lo, color="white", linewidth=0.9, linestyle="--", zorder=5)
    cbar.ax.axvline(hi, color="white", linewidth=0.9, linestyle="--", zorder=5)
    cbar.ax.axvline(lam_center, color="white", linewidth=1.6, zorder=6)

    cbar.ax.tick_params(axis="x", colors="white", labelsize=8, pad=1, length=3, width=0.8)
    cbar.ax.xaxis.set_ticks_position("bottom")
    cbar.set_label("")
    cax.set_yticks([])
    cbar.outline.set_edgecolor("white")
    cbar.outline.set_linewidth(0.6)

    fig.savefig(out_png, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)


def render_frame_lambda_by_row(
    img2d: np.ndarray,
    out_png: Path,
    vmin: float,
    vmax: float,
    lam_y: np.ndarray,
    lam_min: float,
    lam_max: float,
    cover_mask: np.ndarray | None = None,
    cmap_name: str = "rainbow",
    context_gray: bool = True,
    show_lambda_colorbar: bool = True,
    lambda_colorbar_style: str = "top",
) -> None:
    """Render one raw-exposure frame, colored row-by-row by that row's wavelength.

    Used by the point-target "raw" animation mode (one frame per exposure,
    no wavelength binning/coaddition across exposures -- see
    ``spherex-animate --mode raw``), where each detector row has its own
    wavelength because of SPHEREx's linear variable filter.
    """
    x = np.asarray(img2d, dtype=float)

    x_clip = np.clip(x, vmin, vmax)
    denom = (vmax - vmin) if (vmax > vmin) else 1.0
    inten = (x_clip - vmin) / denom
    inten[~np.isfinite(inten)] = 0.0

    ny, nx = inten.shape

    if cover_mask is None:
        mask = np.isfinite(x)
    else:
        mask = (cover_mask > 0) & np.isfinite(x)

    lam = np.asarray(lam_y, dtype=float)
    if lam.size != ny:
        lam = np.interp(np.arange(ny), np.linspace(0, ny - 1, lam.size), lam)

    if not np.isfinite(lam_min) or not np.isfinite(lam_max) or lam_min == lam_max:
        lam_min, lam_max = np.nanmin(lam), np.nanmax(lam)
        if not np.isfinite(lam_min) or not np.isfinite(lam_max) or lam_min == lam_max:
            lam_min, lam_max = 0.0, 1.0

    # Wavelength range covered by THIS frame (from lam_y).
    lam_finite = lam[np.isfinite(lam)]
    if lam_finite.size:
        frame_lam_min = float(np.nanmin(lam_finite))
        frame_lam_max = float(np.nanmax(lam_finite))
    else:
        frame_lam_min = np.nan
        frame_lam_max = np.nan

    # Target wavelength (central row of the cutout).
    y0 = ny // 2
    target_lam = float(lam[y0]) if (0 <= y0 < lam.size and np.isfinite(lam[y0])) else np.nan

    t = (lam - lam_min) / (lam_max - lam_min)
    t = np.clip(t, 0.0, 1.0)
    t[~np.isfinite(t)] = 0.5

    cmap = cm.get_cmap(cmap_name)
    row_rgb = cmap(t)[:, :3]  # (ny, 3)

    rgb_img = np.zeros((ny, nx, 3), dtype=float)

    if context_gray:
        rgb_img[..., 0] = inten
        rgb_img[..., 1] = inten
        rgb_img[..., 2] = inten
        rgb_img *= 0.20

    colored = inten[:, :, None] * row_rgb[:, None, :]  # (ny, nx, 3)
    rgb_img[mask] = colored[mask]

    style = (lambda_colorbar_style or "top").lower()

    fig = plt.figure(figsize=(5.2, 5.2), dpi=160)
    fig.patch.set_facecolor("black")

    if show_lambda_colorbar and style != "right":
        # Image first (bottom), then the colorbar (top), so tick labels are not hidden.
        ax = fig.add_axes([0.00, 0.00, 1.00, 0.955])
        cax = fig.add_axes([0.12, 0.962, 0.76, 0.016])  # [left, bottom, width, height]
        ax.set_zorder(1)
        cax.set_zorder(10)
    else:
        cax = None
        ax = fig.add_axes([0.00, 0.00, 1.00, 1.00])

    ax.set_facecolor("black")
    for sp in ax.spines.values():
        sp.set_visible(False)

    if cax is not None:
        cax.set_facecolor("black")
        for sp in cax.spines.values():
            sp.set_visible(False)
        cax.patch.set_alpha(0.0)

    ax.imshow(rgb_img, origin="lower", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])

    if show_lambda_colorbar:
        if style != "right":
            lam_bar_min, lam_bar_max = 0.7, 5.0
        else:
            lam_bar_min, lam_bar_max = lam_min, lam_max

        norm = mcolors.Normalize(vmin=lam_bar_min, vmax=lam_bar_max)
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        if style == "right":
            cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
            cbar.set_label("Wavelength (um)", fontsize=8)
            cbar.ax.tick_params(labelsize=7)
        else:
            if cax is None:
                cax = ax
            cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")

            desired_ticks = np.array([0.7, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
            cbar.set_ticks(desired_ticks)
            cbar.ax.set_xlim(0.7, 5.0)
            if np.isfinite(frame_lam_min) and np.isfinite(frame_lam_max):
                x0 = float(np.clip(frame_lam_min, lam_bar_min, lam_bar_max))
                x1 = float(np.clip(frame_lam_max, lam_bar_min, lam_bar_max))
                if x1 < x0:
                    x0, x1 = x1, x0

                dlam = max(0.0, x1 - x0)
                full = max(1e-6, lam_bar_max - lam_bar_min)
                frac = np.clip(dlam / full, 0.0, 1.0)
                shade_alpha = float(np.clip(0.03 + 0.18 * frac, 0.03, 0.20))
                cbar.ax.axvspan(x0, x1, color="white", alpha=shade_alpha, zorder=2)

                cbar.ax.axvline(x0, color="white", linewidth=1.2, alpha=0.95, zorder=3)
                cbar.ax.axvline(x1, color="white", linewidth=1.2, alpha=0.95, zorder=3)

            if np.isfinite(target_lam):
                xt = float(np.clip(target_lam, lam_bar_min, lam_bar_max))
                cbar.ax.axvline(xt, color="white", linewidth=2.0, alpha=0.95, zorder=4)
                cbar.ax.annotate(
                    "",
                    xy=(xt, 0.02),
                    xycoords=("data", "axes fraction"),
                    xytext=(xt, 0.95),
                    textcoords=("data", "axes fraction"),
                    arrowprops=dict(arrowstyle="-|>", color="white", lw=1.2),
                    zorder=5,
                )

            cbar.ax.tick_params(axis="x", colors="white", labelsize=8, pad=1, length=3, width=0.8)
            for tick in cbar.ax.get_xticklabels():
                tick.set_clip_on(False)
            cbar.ax.xaxis.set_ticks_position("bottom")
            cbar.ax.xaxis.set_label_position("top")
            cbar.set_label("")

            cax.tick_params(left=False, labelleft=False)
            cax.set_frame_on(False)
            cbar.outline.set_edgecolor("white")
            try:
                cbar.outline.set_linewidth(0.6)
            except Exception:
                pass

            cax.set_facecolor((0, 0, 0, 0.10))
            cax.set_yticks([])

    fig.savefig(out_png, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)


def write_animation(
    frame_paths: list[Path],
    out_path: Path,
    fps: float,
    sec_per_frame: float | None = None,
    mp4: bool = False,
) -> None:
    """Write a sequence of PNG frames to a GIF or MP4.

    GIFs are written directly with Pillow (``duration`` in milliseconds),
    since ``imageio``'s ``mimsave(duration=...)`` does not reliably write a
    nonzero per-frame duration. MP4 is written via ``imageio`` + ``ffmpeg``
    (``libx264``).
    """
    if mp4 or out_path.suffix.lower() == ".mp4":
        writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8)
        try:
            for p in frame_paths:
                writer.append_data(imageio.imread(p))
        finally:
            writer.close()
        print(f"[DONE] MP4: {out_path}")
        return

    dur_sec = (1.0 / float(fps)) if sec_per_frame is None else float(sec_per_frame)
    dur_ms = int(round(dur_sec * 1000))
    pil_frames = [Image.open(p).convert("RGB") for p in frame_paths]
    quantized_frames = _quantize_frames_to_shared_palette(pil_frames)
    # No `disposal=` override here: Pillow's multi-frame GIF writer only encodes the
    # bounding box that changed from the previous frame and leaves the rest alone by
    # default, which is what we want. Passing disposal=2 ("restore to background")
    # instead repaints every pixel OUTSIDE that changed bbox with the GIF's
    # background palette index (0) on each frame -- for frames that mostly agree on
    # which pixels are plain black, index 0 is not necessarily black (it's just
    # whatever color ended up first in the shared palette), so that setting was
    # quietly recoloring most of the frame.
    quantized_frames[0].save(
        out_path, save_all=True, append_images=quantized_frames[1:], duration=dur_ms, loop=0
    )
    print(f"[DONE] GIF: {out_path} ({dur_ms} ms/frame)")


def _render_lambda_slice_chrome_bbox(width_px: int, height_px: int) -> tuple[int, int, int, int] | None:
    """Pixel bbox (top, bottom, left, right) of render_lambda_slice's colorbar strip.

    Computed from its own known figure/axes layout (the same literal
    ``figsize``/``dpi``/``add_axes`` values), not guessed or hardcoded as raw
    pixel numbers, so it stays correct if that layout ever changes. The bbox
    is worked out at that layout's native render resolution and then scaled
    to whatever ``(width_px, height_px)`` is actually asked for, so this still
    applies to frames that were uniformly resized after rendering (e.g. for a
    smaller GIF) -- not just the exact native size. Returns None if the
    layout can't be measured at all (unexpected matplotlib behavior), so
    callers can fall back to treating the animation generically.
    """
    figsize, dpi = 5.2, 160
    ref_px = round(figsize * dpi)
    fig = plt.figure(figsize=(figsize, figsize), dpi=dpi)
    try:
        # Force the Agg canvas explicitly: whatever the process's default interactive
        # backend is (if any) can apply its own device-pixel-ratio scaling to
        # get_renderer()/get_tightbbox(), inflating pixel coordinates relative to what
        # fig.savefig(...) actually writes to the PNG (832x832). Agg is what savefig
        # uses for raster output, so measuring against it keeps the two consistent.
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        FigureCanvasAgg(fig)

        # Mirror render_lambda_slice's own colorbar setup exactly (ticks, labels, tick
        # params and all): get_tightbbox on a bare, otherwise-untouched axes measures
        # nothing like the real, tick-labeled colorbar and gives a meaningless bbox.
        cax = fig.add_axes([0.12, 0.962, 0.76, 0.016])
        norm = mcolors.Normalize(vmin=0.7, vmax=5.0)
        sm = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap("rainbow"))
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
        tick_vals = [0.7, 1.0, 2.0, 3.0, 4.0, 5.0]
        cbar.set_ticks(tick_vals)
        cbar.set_ticklabels([f"{v:.1f}" for v in tick_vals[:-1]] + [rf"{tick_vals[-1]:.0f} $\mu$m"])
        cbar.ax.set_xlim(0.7, 5.0)
        cbar.ax.tick_params(axis="x", colors="white", labelsize=8, pad=1, length=3, width=0.8)
        cbar.ax.xaxis.set_ticks_position("bottom")
        cax.set_yticks([])

        fig.canvas.draw()
        tight = cax.get_tightbbox(fig.canvas.get_renderer())
        # Some backends report get_renderer()/get_width_height() in a device-pixel
        # space scaled from the figure's own dpi (e.g. a HiDPI default), which
        # fig.savefig(...) does not apply to the actual written PNG -- rescale
        # get_tightbbox's coordinates back to the reference (ref_px, ref_px) space.
        canvas_w, canvas_h = fig.canvas.get_width_height()
        scale_x = ref_px / canvas_w
        scale_y = ref_px / canvas_h
        margin = 3
        top_ref = max(0, int(round(ref_px - tight.y1 * scale_y)) - margin)
        bottom_ref = min(ref_px, int(round(ref_px - tight.y0 * scale_y)) + margin)
        left_ref = max(0, int(round(tight.x0 * scale_x)) - margin)
        right_ref = min(ref_px, int(round(tight.x1 * scale_x)) + margin)
        if bottom_ref <= top_ref or right_ref <= left_ref:
            return None

        # Scale the reference-resolution bbox to whatever size was actually asked for.
        sx, sy = width_px / ref_px, height_px / ref_px
        top = int(round(top_ref * sy))
        bottom = int(round(bottom_ref * sy))
        left = int(round(left_ref * sx))
        right = int(round(right_ref * sx))
        if bottom <= top or right <= left:
            return None
        return top, bottom, left, right
    finally:
        plt.close(fig)


def _quantize_frames_to_shared_palette(frames: list, n_static_colors: int = 96) -> list:
    """Quantize an animation's frames for a GIF save, keeping the colorbar stable
    without sacrificing each frame's own data quality.

    Earlier approaches all failed on real data:

    - Letting Pillow quantize each RGB frame's 256-color palette independently (its
      default) makes every frame's *own* data look as good as a standalone image, but
      anything meant to look identical across frames (the fixed wavelength colorbar)
      visibly shifts color frame to frame, since each frame's quantizer rounds it
      differently.
    - Reserving a fixed set of colors sampled from the colorbar's own colormap at full
      saturation, so a population-based quantizer wouldn't starve it, backfired
      instead: a frame's own data pixels are that same colormap *darkened* by
      intensity, so dim/mid-tone data pixels had no close match among the reserved
      full-brightness entries and got snapped up to a brighter one, producing visible
      posterized rings and oversaturated colors on real science data.
    - Reserving colors sampled from the *actual* colorbar pixels (instead of a
      synthetic full-saturation sample), but still quantizing the whole frame against
      ``reserved + this frame's own data colors`` in one nearest-color pass, still let
      the colorbar drift: a frame's own data colors come from that same colormap
      family, so for some frames a data color ends up nearer to a given colorbar
      pixel than any reserved entry, and nearest-color matching picks it instead --
      different winners in different frames.
    - Deciding which pixels are "shared chrome" by checking where every frame agrees
      (within a tolerance) falls apart on a cube where most wavelength bins have no
      coverage: a sky pixel that is genuinely black in most (empty) bins but bright in
      a few covered ones still has most frames "agreeing" on black, so real data gets
      mistaken for chrome.

    So: locate the colorbar by its actual on-figure position (render_lambda_slice's
    own known axes layout), not by guessing from pixel statistics -- that is immune to
    how much of the data is empty. Within that strip, take the per-pixel *median*
    across frames rather than any single frame's own pixels: the strip is identical
    in every frame except for the thin per-frame wavelength marker drawn on top of it,
    so the median reconstructs the marker-free gradient (the marker is a minority
    event at any given pixel across many frames). That gradient is quantized once;
    everything else -- the real data, which is what matters most and varies frame to
    frame -- is quantized fresh per frame, each into its own disjoint block of palette
    indices, then the two index arrays are pasted together directly. Every pixel's
    final index comes straight from whichever quantization already decided it, so
    there is no second round of nearest-color matching where one side could win over
    the other.

    The marker itself would otherwise be erased by this (every frame would just show
    the shared, marker-free gradient), so it is restored separately: wherever a
    frame's own strip pixels deviate strongly from the marker-free median -- which,
    by construction, only happens at that frame's own marker lines -- those pixels are
    forced to the reserved pure-white index instead (the marker is drawn in solid
    white), rather than left in contention with the rest of the palette.
    """
    w, h = frames[0].size
    bbox = _render_lambda_slice_chrome_bbox(w, h)

    black_white = [(0, 0, 0), (255, 255, 255)]
    static_colors: list[tuple[int, int, int]] = []
    static_mask = np.zeros((h, w), dtype=bool)
    static_idx_full = None
    if bbox is not None:
        top, bottom, left, right = bbox
        static_mask[top:bottom, left:right] = True
        stack = np.stack([np.asarray(f, dtype=np.int16)[top:bottom, left:right] for f in frames], axis=0)
        median_rgb = np.median(stack, axis=0).astype(np.uint8)  # (bbox_h, bbox_w, 3), marker-free

        n_static = min(n_static_colors, 256 - len(black_white))
        static_src = Image.fromarray(median_rgb, mode="RGB")
        static_q = static_src.quantize(colors=n_static, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        static_bytes = static_q.getpalette()[: 3 * n_static]
        static_colors = [tuple(static_bytes[i : i + 3]) for i in range(0, len(static_bytes), 3)]
        static_idx_full = np.zeros((h, w), dtype=np.int32)
        static_idx_full[top:bottom, left:right] = np.asarray(static_q)

    offset_static = len(black_white)
    offset_data = offset_static + len(static_colors)
    n_data_colors = 256 - offset_data

    quantized = []
    for f in frames:
        arr = np.asarray(f, dtype=np.uint8).copy()
        if static_idx_full is not None:
            arr[static_mask] = 0  # exclude the shared chrome from this frame's own data palette
        # Dithering: 256 colors can't render a smooth, high-dynamic-range glow (e.g. a
        # galaxy bulge) without banding when quantized flat. Floyd-Steinberg trades
        # that banding for fine dither noise instead -- but only where there is a real
        # gradient to smooth: an exact palette match (solid black outside the
        # footprint, in particular) propagates zero error, so it stays exactly flat.
        data_q = Image.fromarray(arr, mode="RGB").quantize(
            colors=n_data_colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG
        )
        data_bytes = data_q.getpalette()[: 3 * n_data_colors]
        data_colors = [tuple(data_bytes[i : i + 3]) for i in range(0, len(data_bytes), 3)]
        data_idx = np.asarray(data_q, dtype=np.int32) + offset_data

        final_idx = data_idx if static_idx_full is None else np.where(static_mask, static_idx_full + offset_static, data_idx)

        if static_idx_full is not None:
            crop = np.asarray(f, dtype=np.int16)[top:bottom, left:right]
            marker_crop = np.abs(crop - median_rgb.astype(np.int16)).sum(axis=-1) > 80
            if marker_crop.any():
                marker_full = np.zeros((h, w), dtype=bool)
                marker_full[top:bottom, left:right] = marker_crop
                final_idx = np.where(marker_full, 1, final_idx)  # index 1 == pure white

        flat = [v for rgb in (black_white + static_colors + data_colors) for v in rgb]
        flat += [0] * (768 - len(flat))
        p_img = Image.fromarray(final_idx.astype(np.uint8), mode="P")
        p_img.putpalette(flat)
        quantized.append(p_img)
    return quantized
