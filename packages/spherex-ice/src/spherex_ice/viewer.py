"""Interactive double-click-to-see-spectrum viewer, for maps already built by
``spherex-ice-map`` (or ``-get``/``-from-cube``) with their raw exposure cutouts still
on disk.

Shows two or more feature median maps (e.g. H2O and PAH) side by side, sharing one
window; zooming/panning either one (e.g. via the toolbar's zoom-rectangle tool) applies
the same pixel view to all the others, since they all share one pixel grid. A single
click there is left alone for that (zooming is a click-and-drag); double-clicking a
position instead marks it (a small circle, in its own color) on every map, and adds
that position's full SPHEREx spectrum -- flux density (Fnu, SPHEREx's native MJy/sr by
default, or a background-subtracted circular-aperture flux in mJy with ``aperture_px``
set) vs. wavelength on a log Y axis, one point per raw exposure that covers it, see
:mod:`spherex_ice.spectrum` -- to a second, running plot, in the same color as its
marker. Double-clicking elsewhere adds another point/spectrum in a new color, so
several positions can be compared; a "Guardar" button writes every spectrum shown so
far to its own CSV, plus a PNG of each window, under ``--out_dir``, and "Borrar todo"
clears every marker/spectrum to start over.

Meant to be run once everything of interest is already downloaded (the raw cutouts take
the vast majority of the disk space and download time; this tool only reads them).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from spherex_ice import spectrum as spectrum_mod
from spherex_ice import tau as tau_mod


def _load_map(name: str, path: Path) -> dict[str, Any]:
    with fits.open(path) as hdul:
        data = np.asarray(hdul["TAU"].data, dtype=float)
        header = hdul["TAU"].header
    kind = header.get("KIND", "absorption")
    finite = data[np.isfinite(data)]
    vmax = float(np.percentile(finite, tau_mod.default_vmax_percentile(kind))) if finite.size else 1.0
    preset = tau_mod.FEATURE_PRESETS.get(name)
    return {
        "name": name,
        "data": data,
        "wcs": WCS(header),
        "vmax": vmax,
        "cmap": (preset or {}).get("cmap", "gray_r"),
        "label": (preset or {}).get("label", name),
    }


def run_viewer(
    map_paths: dict[str, Path],
    cutouts_dir: Path,
    bad_bits: list[str],
    out_dir: Path,
    aperture_px: float | None = None,
    annulus_in_px: float | None = None,
    annulus_out_px: float | None = None,
) -> None:
    """Open the two-window viewer. Blocks (``plt.show()``) until both windows are closed.

    By default each double-click extracts a single-pixel spectrum. Pass ``aperture_px``
    to switch to a background-subtracted circular-aperture spectrum instead (see
    :func:`spherex_ice.spectrum.extract_aperture_spectrum_at_position`), useful when a
    single pixel is too noisy to judge a feature by.
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button

    cutout_paths = spectrum_mod.list_cutout_files(cutouts_dir)
    if not cutout_paths:
        raise SystemExit(f"[ERROR] No .fits cutouts found in {cutouts_dir}")
    print(f"[INFO] {len(cutout_paths)} raw cutouts available for spectrum extraction.")

    maps = {name: _load_map(name, path) for name, path in map_paths.items()}

    fig_maps = plt.figure(figsize=(5.2 * len(maps), 5.6))
    fig_maps.suptitle("Double-click a position on any map to see its spectrum")
    axes: dict[str, Any] = {}
    for i, (name, m) in enumerate(maps.items(), start=1):
        ax = fig_maps.add_subplot(1, len(maps), i, projection=m["wcs"])
        clipped = np.clip(m["data"], 0.0, m["vmax"])
        ax.imshow(clipped, origin="lower", interpolation="nearest", cmap=m["cmap"], vmin=0.0, vmax=m["vmax"])
        ax.set_title(m["label"], fontsize=10)
        ax.coords[0].set_axislabel("RA")
        ax.coords[1].set_axislabel("Dec")
        axes[name] = ax
    fig_maps.tight_layout()

    # Every map shares one pixel grid (the same --out_dir run's ref_wcs), so a pixel-space
    # view (xlim/ylim) from one applies unchanged to the others -- keep them all in lock
    # step whenever any one is zoomed/panned. The "syncing" guard stops the resulting
    # set_xlim/set_ylim calls on the other axes from re-triggering this right back.
    syncing = {"active": False}

    def _sync_limits(changed_ax):
        if syncing["active"]:
            return
        syncing["active"] = True
        try:
            xlim, ylim = changed_ax.get_xlim(), changed_ax.get_ylim()
            for ax in axes.values():
                if ax is not changed_ax:
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
            fig_maps.canvas.draw_idle()
        finally:
            syncing["active"] = False

    for ax in axes.values():
        ax.callbacks.connect("xlim_changed", _sync_limits)
        ax.callbacks.connect("ylim_changed", _sync_limits)

    aperture_mode = aperture_px is not None
    flux_label = "Fν (mJy, aperture)" if aperture_mode else "Fν (MJy/sr)"
    flux_csv_col = "flux_mjy" if aperture_mode else "flux_mjy_sr"

    fig_spec, ax_spec = plt.subplots(figsize=(7.5, 4.8))
    fig_spec.subplots_adjust(bottom=0.2)

    def _reset_spec_axes():
        ax_spec.set_xlabel("Wavelength (µm)")
        ax_spec.set_ylabel(flux_label)
        ax_spec.set_yscale("log")
        title = "Double-click a map to extract its SPHEREx spectrum"
        if aperture_mode:
            title += f" (r={aperture_px:g} px aperture)"
        ax_spec.set_title(title)

    _reset_spec_axes()

    out_dir = Path(out_dir)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    clicks: list[dict[str, Any]] = []

    def on_click(event):
        if not event.dblclick:
            return  # a plain click/drag is for the zoom/pan tools, not for marking a spot
        clicked_name = next((name for name, ax in axes.items() if event.inaxes is ax), None)
        if clicked_name is None or event.xdata is None:
            return
        sky = maps[clicked_name]["wcs"].pixel_to_world(event.xdata, event.ydata)
        ra, dec = float(sky.ra.deg), float(sky.dec.deg)

        print(f"[INFO] Click #{len(clicks) + 1}: RA={ra:.5f} Dec={dec:+.5f} "
              f"-- scanning {len(cutout_paths)} cutouts...")
        if aperture_mode:
            spec = spectrum_mod.extract_aperture_spectrum_at_position(
                ra, dec, cutout_paths, bad_bits, aperture_px, annulus_in_px, annulus_out_px
            )
        else:
            spec = spectrum_mod.extract_spectrum_at_position(ra, dec, cutout_paths, bad_bits)
        if spec["lam"].size == 0:
            print("[WARN] No valid flux at that position in any cutout -- ignored.")
            return

        color = color_cycle[len(clicks) % len(color_cycle)]
        idx = len(clicks) + 1
        clicks.append({"ra": ra, "dec": dec, "color": color, **spec})

        for name, ax in axes.items():
            xw, yw = maps[name]["wcs"].world_to_pixel(sky)
            ax.plot(xw, yw, marker="o", mfc="none", mec=color, mew=1.6, ms=9)
        fig_maps.canvas.draw_idle()

        positive = spec["flux"] > 0
        ax_spec.plot(
            spec["lam"][positive], spec["flux"][positive], "o-", color=color, ms=3, lw=0.9,
            label=f"#{idx}  RA={ra:.4f}  Dec={dec:+.4f}",
        )
        ax_spec.legend(fontsize=7, loc="best")
        fig_spec.canvas.draw_idle()

    def on_save(_event):
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx, c in enumerate(clicks, start=1):
            csv_path = out_dir / f"spectrum_{idx:02d}_ra{c['ra']:.4f}_dec{c['dec']:+.4f}.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["lambda_um", flux_csv_col, "source_file"])
                writer.writerows(zip(c["lam"], c["flux"], c["files"], strict=True))
        fig_spec.savefig(out_dir / "spectra.png", dpi=150)
        fig_maps.savefig(out_dir / "maps_with_markers.png", dpi=150)
        print(f"[DONE] Saved {len(clicks)} spectra (.csv) + spectra.png + maps_with_markers.png -> {out_dir}")

    def on_clear(_event):
        for ax in axes.values():
            for line in list(ax.lines):
                line.remove()
        ax_spec.cla()
        _reset_spec_axes()
        clicks.clear()
        fig_maps.canvas.draw_idle()
        fig_spec.canvas.draw_idle()
        print("[INFO] Cleared all markers and spectra.")

    clear_ax = fig_spec.add_axes((0.64, 0.03, 0.16, 0.07))
    clear_button = Button(clear_ax, "Borrar todo")
    clear_button.on_clicked(on_clear)

    save_ax = fig_spec.add_axes((0.82, 0.03, 0.15, 0.07))
    save_button = Button(save_ax, "Guardar")
    save_button.on_clicked(on_save)

    fig_maps.canvas.mpl_connect("button_press_event", on_click)
    plt.show()
