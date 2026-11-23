"""Color composites from up to three already-built feature maps (e.g. H2O tau in
red, CO2 tau in green, PAH emission in blue) -- not three wavelengths of one map
the way :func:`spherexstack.animate.render_rgb_composite` does, but three
different *feature* products, each already reduced to its own tau/MJy-per-sr
map by :mod:`spherex_ice.tau`, stacked as color channels on their shared grid.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


def _load_channel(path: Path | None) -> tuple[np.ndarray | None, WCS | None, dict[str, Any]]:
    if path is None:
        return None, None, {}
    with fits.open(path, memmap=False) as hdul:
        tau = np.asarray(hdul["TAU"].data, dtype=float)
        hdr = hdul["TAU"].header
        wcs = WCS(hdr)
        meta = {
            "feature": hdr.get("FEATURE") or hdr.get("LABEL"),
            "kind": hdr.get("KIND"),
        }
    return tau, wcs, meta


def compose_rgb(
    red_fits: Path | None,
    green_fits: Path | None,
    blue_fits: Path | None,
    red_vmax: float | None = None,
    green_vmax: float | None = None,
    blue_vmax: float | None = None,
) -> dict[str, Any]:
    """Build an (ny, nx, 3) RGB array in [0, 1] from up to three ``tau_epochNN.fits``
    files (each a :func:`spherex_ice.tau.build_epoch_tau_map` output), one per channel.

    At least one of ``red_fits``/``green_fits``/``blue_fits`` must be given; any left
    as None contributes a channel of all zeros (black). All given files must have the
    same pixel grid (same ROI/pixscale) -- this does not reproject between them.

    Each channel is independently clipped to ``[0, vmax]`` and scaled to ``[0, 1]``;
    a channel's own ``vmax`` defaults to the 99th percentile of its finite values
    if not given.

    Returns a dict with ``rgb`` (the composite array), ``wcs`` (from whichever
    channel was given, preferring red > green > blue), ``shape``, and each
    channel's resolved ``*_vmax``/``*_meta`` for the caller to label things with.
    """
    channels = []
    wcs = None
    shape = None
    for path, vmax in ((red_fits, red_vmax), (green_fits, green_vmax), (blue_fits, blue_vmax)):
        tau, w, meta = _load_channel(path)
        if tau is not None:
            if shape is None:
                shape = tau.shape
            elif tau.shape != shape:
                raise ValueError(
                    f"Channel grids don't match: {path} is {tau.shape}, expected {shape} -- "
                    "all inputs must come from the same ROI/pixel scale."
                )
            if wcs is None:
                wcs = w
        channels.append((tau, vmax, meta))

    if shape is None:
        raise ValueError("Give at least one of red_fits/green_fits/blue_fits.")

    planes = []
    resolved_vmax = []
    metas = []
    for tau, vmax, meta in channels:
        if tau is None:
            planes.append(np.zeros(shape, dtype=float))
            resolved_vmax.append(None)
            metas.append(meta)
            continue
        finite = tau[np.isfinite(tau)]
        v = vmax if vmax is not None else (float(np.percentile(finite, 99)) if finite.size else 1.0)
        v = v if v > 0 else 1.0
        plane = np.where(np.isfinite(tau), np.clip(tau, 0.0, v), 0.0) / v
        planes.append(plane)
        resolved_vmax.append(v)
        metas.append(meta)

    rgb = np.stack(planes, axis=-1)
    return {"rgb": rgb, "wcs": wcs, "shape": shape, "vmax": resolved_vmax, "meta": metas}


def save_rgb_fits(path: Path, result: dict[str, Any]) -> None:
    """Write the composite as a 3-plane cube (R, G, B as separate HDUs), with the
    shared WCS, so it can be reused (reprojected, regridded, ...) elsewhere.
    """
    hdr = result["wcs"].to_header() if result["wcs"] is not None else fits.Header()
    names = ("R", "G", "B")
    hdul = [fits.PrimaryHDU()]
    for i, name in enumerate(names):
        h = hdr.copy()
        vmax = result["vmax"][i]
        meta = result["meta"][i]
        if vmax is not None:
            h["VMAX"] = (float(vmax), f"{name} channel clip value (0=black)")
        if meta.get("feature"):
            h["FEATURE"] = str(meta["feature"])
        if meta.get("kind"):
            h["KIND"] = str(meta["kind"])
        hdul.append(fits.ImageHDU(data=result["rgb"][..., i].astype(np.float32), header=h, name=name))
    fits.HDUList(hdul).writeto(path, overwrite=True)


def compose_blend(
    paths: list[Path],
    cmaps: list[Any] | None = None,
    vmaxes: list[float | None] | None = None,
) -> dict[str, Any]:
    """Blend 2-3 already-built feature maps, each through its *own* perceptual
    colormap (e.g. H2O's black-blue-white scale, PAH's black-red-yellow-white
    "heat" scale -- see each preset's ``"cmap"`` in
    :data:`spherex_ice.tau.FEATURE_PRESETS`), via a "screen" blend:
    ``1 - (1-a)*(1-b)`` per RGB channel, the natural way to combine several
    bright-on-black glow-style maps without one washing out the other --
    unlike :func:`compose_rgb`, which stacks up to three maps as literal
    R/G/B primaries instead of each keeping its own color.

    ``cmaps[i]`` left as ``None`` falls back to that file's own FEATURE header
    matched against :data:`spherex_ice.tau.FEATURE_PRESETS`, then to plain
    grayscale ("gray") if that feature has no preset colormap either.
    """
    from matplotlib import colormaps as mpl_colormaps

    from spherex_ice.tau import FEATURE_PRESETS

    label_to_cmap = {p["label"]: p.get("cmap") for p in FEATURE_PRESETS.values()}

    if len(paths) < 2:
        raise ValueError("compose_blend needs at least 2 feature maps to blend.")
    if len(paths) > 3:
        raise ValueError("compose_blend takes at most 3 feature maps.")
    cmaps = list(cmaps) if cmaps is not None else [None] * len(paths)
    vmaxes = list(vmaxes) if vmaxes is not None else [None] * len(paths)

    shape = None
    wcs = None
    rgb = None
    resolved_vmax = []
    metas = []
    resolved_cmaps = []
    for path, cmap, vmax in zip(paths, cmaps, vmaxes, strict=True):
        tau, w, meta = _load_channel(path)
        if shape is None:
            shape = tau.shape
            wcs = w
        elif tau.shape != shape:
            raise ValueError(
                f"Channel grids don't match: {path} is {tau.shape}, expected {shape} -- "
                "all inputs must come from the same ROI/pixel scale."
            )

        resolved_cmap = cmap if cmap is not None else label_to_cmap.get(meta.get("feature")) or "gray"
        finite = tau[np.isfinite(tau)]
        v = vmax if vmax is not None else (float(np.percentile(finite, 99)) if finite.size else 1.0)
        v = v if v > 0 else 1.0
        norm = np.clip(np.where(np.isfinite(tau), tau, 0.0), 0.0, v) / v

        cmap_obj = mpl_colormaps[resolved_cmap] if isinstance(resolved_cmap, str) else resolved_cmap
        layer = cmap_obj(norm)[..., :3]
        rgb = layer if rgb is None else 1.0 - (1.0 - rgb) * (1.0 - layer)

        resolved_vmax.append(v)
        metas.append(meta)
        resolved_cmaps.append(resolved_cmap)

    return {
        "rgb": rgb, "wcs": wcs, "shape": shape,
        "vmax": resolved_vmax, "meta": metas, "cmaps": resolved_cmaps,
    }


def render_blend_png(
    result: dict[str, Any],
    out_png: Path,
    labels: list[str | None] | None = None,
    fov_deg: tuple[float, float] | None = None,
    mark_target: bool = True,
) -> None:
    """Render a :func:`compose_blend` result -- always against a black background
    (these are false-color glow composites, not the plain grayscale field image
    ``invert`` toggles between white/black for -- see
    :func:`spherex_ice.render.render_tau_map`).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spherex_ice.render import _draw_target_reticle, _format_fov, _make_image_axes

    fig = plt.figure(figsize=(5.2, 5.6), dpi=160)
    fig.patch.set_facecolor("black")
    ax = _make_image_axes(fig, result["wcs"], "black", "white", top=0.90)
    ax.imshow(result["rgb"], origin="lower", interpolation="nearest")
    if mark_target:
        _draw_target_reticle(ax, result["shape"], color="white")

    labels = labels or [None] * len(result["meta"])
    legend_bits = [label for label in labels if label]
    if legend_bits:
        fig.text(0.5, 0.975, "  +  ".join(legend_bits), color="white", fontsize=8.5, ha="center", va="top")
    if fov_deg is not None:
        ax.text(
            0.98, 0.98, f"FOV {_format_fov(*fov_deg)}",
            transform=ax.transAxes, color="white", fontsize=9, ha="right", va="top",
            bbox=dict(facecolor="black", alpha=0.4, edgecolor="none", pad=2),
        )
    fig.savefig(out_png, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)


def render_rgb_png(
    result: dict[str, Any],
    out_png: Path,
    labels: tuple[str | None, str | None, str | None] = (None, None, None),
    invert: bool = True,
    fov_deg: tuple[float, float] | None = None,
    mark_target: bool = True,
) -> None:
    """Render the composite as a plain RGB PNG, with a small legend naming each
    channel's feature (when given) and real RA/Dec tick labels (from ``result["wcs"]``).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spherex_ice.render import _draw_target_reticle, _format_fov, _make_image_axes

    bg_color = "white" if invert else "black"
    fg_color = "black" if invert else "white"
    rgb = result["rgb"]
    if invert:
        rgb = 1.0 - rgb

    fig = plt.figure(figsize=(5.2, 5.6), dpi=160)
    fig.patch.set_facecolor(bg_color)
    ax = _make_image_axes(fig, result["wcs"], bg_color, fg_color, top=0.90)
    ax.imshow(rgb, origin="lower", interpolation="nearest")
    if mark_target:
        _draw_target_reticle(ax, result["shape"])

    channel_names = ("R", "G", "B")
    legend_bits = [
        f"{channel_names[i]}: {labels[i]}" for i in range(3) if labels[i]
    ]
    if legend_bits:
        fig.text(
            0.5, 0.975, "   ".join(legend_bits), color=fg_color, fontsize=8.5, ha="center", va="top",
        )
    if fov_deg is not None:
        ax.text(
            0.98, 0.98, f"FOV {_format_fov(*fov_deg)}",
            transform=ax.transAxes, color=fg_color, fontsize=9, ha="right", va="top",
            bbox=dict(facecolor=bg_color, alpha=0.4, edgecolor="none", pad=2),
        )
    fig.savefig(out_png, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
