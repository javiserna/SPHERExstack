"""Console-script entry points: ``spherex-ice-epochs``, ``spherex-ice-map``, ``spherex-ice-get``.

Reuses :mod:`spherexstack` (query, download, roi, io, mosaic) for everything
up to "a common pixel grid of reprojected fluxes"; this package only adds
the parts specific to water-ice mapping: grouping exposures into observation
epochs (:mod:`spherex_ice.epochs`) and the per-pixel continuum-interpolation
optical depth calculation (:mod:`spherex_ice.tau`).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
from spherexstack import download as download_mod
from spherexstack import io as sx_io
from spherexstack import mosaic as mosaic_mod
from spherexstack import query as query_mod
from spherexstack import roi as roi_mod

from spherex_ice import combine as combine_mod
from spherex_ice import contour as contour_mod
from spherex_ice import cube as cube_mod
from spherex_ice import epochs as epochs_mod
from spherex_ice import render as render_mod
from spherex_ice import rgb as rgb_mod
from spherex_ice import tau as tau_mod
from spherex_ice import viewer as viewer_mod


def _bad_bits_from_arg(arg: str) -> list[str]:
    return [b.strip() for b in arg.split(",") if b.strip()]


def _add_cleaning_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--no-clean", action="store_true", help="Disable FLAGS-based hot pixel/outlier cleaning.")
    ap.add_argument(
        "--bad-bits",
        default=",".join(sx_io.DEFAULT_BAD_BITS),
        help=f"Comma list of FLAGS bits to treat as defects. Available: {','.join(sx_io.FLAG_BITS.keys())}.",
    )


def _add_epoch_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--max_gap_days", type=float, default=30.0,
        help="Start a new epoch when the gap since the previous exposure exceeds this (days). A "
        "single visit can have its own internal scheduling gaps of a couple of weeks, so this "
        "needs to be noticeably larger than that but well below the gap between true revisits "
        "(often 100+ days) -- check spherex-ice-epochs' output and adjust if it looks wrong.",
    )
    ap.add_argument(
        "--max_span_days", type=float, default=45.0,
        help="Safety cap: force a new epoch rather than let one grow past this total span (days).",
    )


def _add_tau_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--feature", nargs="+", type=_feature_name_type, default=["h2o"],
        metavar="FEATURE",
        help="One or more named spectral features (space-separated, case-insensitive), e.g. "
        "--feature h2o pah co2 -- one map (+ contour) is built for each, in its own "
        "subdirectory when more than one is given. Choices: "
        f"{', '.join([*tau_mod.FEATURE_PRESETS.keys(), 'custom'])}. 'custom' uses whatever "
        "--lam_line/--dlambda_line/--cont_*/kind you pass instead of a preset.",
    )
    ap.add_argument(
        "--lam_line", type=float, default=tau_mod.H2O_LAMBDA_UM,
        help="Line center (um). Ignored unless --feature custom.",
    )
    ap.add_argument(
        "--dlambda_line", type=float, default=0.05,
        help="Half-width (um) of the line band. Ignored unless --feature custom.",
    )
    ap.add_argument("--cont_blue_lo", type=float, default=2.50, help="Ignored unless --feature custom.")
    ap.add_argument("--cont_blue_hi", type=float, default=2.75, help="Ignored unless --feature custom.")
    ap.add_argument("--cont_red_lo", type=float, default=3.75, help="Ignored unless --feature custom.")
    ap.add_argument("--cont_red_hi", type=float, default=4.00, help="Ignored unless --feature custom.")
    ap.add_argument(
        "--no-match-background", action="store_true", help="Disable incremental background offset matching."
    )
    ap.add_argument("--min_overlap_px", type=int, default=20)
    ap.add_argument("--weighting", choices=mosaic_mod.WEIGHTING_CHOICES, default="mean")
    ap.add_argument(
        "--no-reject-outliers", action="store_true",
        help="Disable automatic rejection of exposures whose median in-band flux is off by more "
        f"than {tau_mod.OUTLIER_FACTOR:.0f}x from their peers (see tau.OUTLIER_FACTOR docstring). "
        "With very few exposures per band, one such exposure can otherwise bias the whole epoch's "
        "tau map, not just its own small footprint.",
    )
    ap.add_argument(
        "--min_weight", type=float, default=1.0,
        help="Mask out any pixel not backed by at least this much per-pixel weight in ALL THREE "
        "bands (under --weighting mean, that's simply exposure count). Default 1.0 keeps every "
        "pixel with any coverage at all; raising it to 2 drops single-exposure pixels, which is "
        "where leftover band-to-band registration mismatches near stars show up as isolated "
        "bright/dark pixel pairs, at the cost of some coverage.",
    )
    ap.add_argument(
        "--vmax", type=float, default=None,
        help="Value that saturates the display. Default: the 99.9th percentile of that epoch's "
        "own finite tau values for an absorption feature, or the 95th percentile for an "
        "emission feature (no fixed physical scale applies to either).",
    )
    ap.add_argument(
        "--dark", action="store_true",
        help="Render with the classic dark theme (black background, white text/ticks) instead "
        "of the default: white background, black text/ticks, data shown as a photographic "
        "negative (higher signal = darker).",
    )
    ap.add_argument(
        "--no-contour", action="store_true",
        help="Skip the contour product (tau_epochNN_contour.png + .fits) that's otherwise built "
        "alongside every map -- a reusable FITS mask (with the same WCS) of where the signal is "
        "at or above --contour_level, plus a PNG showing that boundary drawn on the map.",
    )
    ap.add_argument(
        "--contour_level", type=float, default=None,
        help="Threshold for the contour product. Default: the chosen --feature's own "
        "'contour_level' (0.5 tau for h2o, 0.2 tau for co2/co -- see tau.FEATURE_PRESETS); for "
        "an emission feature, or --feature custom, 30%% of that epoch's own 95th-percentile "
        "signal level (no single physical cutoff applies across lines of very different "
        "brightness).",
    )
    ap.add_argument(
        "--no-median", action="store_true",
        help="Skip the extra tau_median.fits/.png (+ contour) product that's otherwise built "
        "alongside the per-epoch maps whenever more than one epoch is processed: the per-pixel "
        "median of tau across all those epochs. More robust than any single epoch to a one-off "
        "artifact (an outlier exposure, a registration mismatch), at the cost of no longer "
        "showing real time variability -- keep both: epochs to check stability, the median for "
        "the field's overall most-trustworthy structure.",
    )
    ap.add_argument(
        "--median-only", action="store_true",
        help="Build the median (+ its contour) but don't write each epoch's own "
        "tau_epochNN.fits/.png/_contour.* -- for when you only want the combined map and don't "
        "need the per-epoch files cluttering the output directory. Each epoch's data is still "
        "used to compute the median; only the per-epoch files are skipped. Not compatible with "
        "--no-median (there would be nothing left to write).",
    )
    ap.add_argument(
        "--auto-rgb", action="store_true",
        help="With exactly 2 or 3 --feature values, also build an RGB composite from their "
        "median maps (first=red, second=green, third=blue if given), saved as composite.png/"
        ".fits alongside the per-feature subdirectories. Requires the median to actually be "
        "built (not compatible with --no-median).",
    )
    ap.add_argument(
        "--cube", action="store_true",
        help="Also save one compact FITS cube per epoch (cubes/epochNN_cube.fits), bundling "
        "every band already computed for the --feature(s) you asked for this run -- at no "
        "extra cost, since those bands are already built along the way. Use "
        "spherex-ice-from-cube afterward to derive any of those same features' maps straight "
        "from the cube, without ever touching the (often much larger, more numerous) raw "
        "per-exposure FITS cutouts again.",
    )


# ---------------------------------------------------------------------------
# spherex-ice-epochs
# ---------------------------------------------------------------------------
def epochs() -> None:
    ap = argparse.ArgumentParser(
        prog="spherex-ice-epochs",
        description="Group a list of SPHEREx exposures into observation epochs by MJD-OBS.",
    )
    ap.add_argument("--list", required=True, help="Path to list.txt (one FITS path per line).")
    ap.add_argument("--out", default="epochs.json")
    _add_epoch_args(ap)
    args = ap.parse_args()

    list_path = Path(args.list).expanduser().resolve()
    files = sx_io.resolve_list_paths(list_path)
    print(f"[INFO] Reading MJD-OBS from {len(files)} exposures...")
    eps = epochs_mod.group_into_epochs(
        files, max_gap_days=args.max_gap_days, max_span_days=args.max_span_days
    )

    n_skipped = len(files) - sum(len(e.files) for e in eps)
    if n_skipped:
        print(f"[WARN] {n_skipped} exposures had no readable MJD-OBS and were skipped.")

    print(f"[INFO] {len(eps)} epoch(s):")
    out_data = []
    for e in eps:
        print(f"[INFO]   epoch {e.index}: {len(e.files)} files, MJD {e.mjd_min:.2f}-{e.mjd_max:.2f} "
              f"(span {e.span_days:.1f} d)")
        out_data.append(
            {
                "index": e.index,
                "mjd_min": e.mjd_min,
                "mjd_max": e.mjd_max,
                "files": [str(fp) for fp in e.files],
            }
        )
    Path(args.out).write_text(json.dumps(out_data, indent=2))
    print(f"[DONE] {args.out}")


def _load_epochs(path: str) -> list[epochs_mod.Epoch]:
    data = json.loads(Path(path).read_text())
    return [
        epochs_mod.Epoch(
            index=d["index"], files=[Path(f) for f in d["files"]], mjd_min=d["mjd_min"], mjd_max=d["mjd_max"]
        )
        for d in data
    ]


def _feature_name_type(s: str) -> str:
    """``argparse`` ``type=`` for one ``--feature`` token: lower-cased, validated against
    ``tau.FEATURE_PRESETS`` (plus ``"custom"``) so ``--feature H2O PAH`` and the like work.
    """
    name = s.strip().lower()
    valid = {*tau_mod.FEATURE_PRESETS.keys(), "custom"}
    if name not in valid:
        choices = ", ".join(sorted(valid))
        raise argparse.ArgumentTypeError(f"'{s}' is not a known feature (choices: {choices})")
    return name


def _resolve_feature(args: argparse.Namespace, feature_name: str) -> dict:
    """Band/kind/label settings for ``feature_name``: from ``tau.FEATURE_PRESETS``, or
    straight from the individual --lam_line/--cont_* flags if ``feature_name == "custom"``.
    """
    if feature_name != "custom":
        preset = tau_mod.FEATURE_PRESETS[feature_name]
        return dict(preset)
    return {
        "label": f"{args.lam_line:.2f} um custom", "kind": "absorption",
        "symbol": r"$\tau_{peak}$",
        "lam_line": args.lam_line, "dlambda_line": args.dlambda_line,
        "cont_blue_lo": args.cont_blue_lo, "cont_blue_hi": args.cont_blue_hi,
        "cont_red_lo": args.cont_red_lo, "cont_red_hi": args.cont_red_hi,
    }


def _save_and_render_result(
    result: dict,
    feature: dict,
    args: argparse.Namespace,
    ref_wcs,
    ref_shape,
    out_dir: Path,
    index,
    mjd_min: float,
    mjd_max: float,
    file_prefix: str,
    write_outputs: bool = True,
) -> dict:
    """Save/render one already-built result (one epoch, or the median across several) --
    shared by the raw-exposure path (:func:`_build_and_save_epoch_map`) and the
    compact-cube path (``spherex-ice-from-cube``), since both produce the same kind of
    result dict (see :func:`spherex_ice.tau.build_epoch_tau_map`).

    ``index`` is an epoch index (int) or ``"median"``; ``file_prefix`` names the output
    files, e.g. ``"tau_epoch00"`` or ``"tau_median"``.
    """
    print(f"[INFO] {file_prefix}: coverage={100 * result['coverage']:.1f}%")
    if not write_outputs:
        print(f"[INFO] {file_prefix}: --median-only, not writing its own files.")
        return result

    vmax = args.vmax
    if vmax is None:
        finite = result["tau"][np.isfinite(result["tau"])]
        pct = tau_mod.default_vmax_percentile(feature["kind"])
        vmax = float(np.percentile(finite, pct)) if finite.size else 1.0

    units = "tau" if feature["kind"] == "absorption" else "MJy/sr"
    fits_path = out_dir / f"{file_prefix}.fits"
    png_path = out_dir / f"{file_prefix}.png"
    render_mod.save_tau_fits(
        fits_path, result, ref_wcs, index, mjd_min, mjd_max, feature_label=feature["label"]
    )
    render_mod.render_tau_map(
        result["tau"], png_path, index, mjd_min, mjd_max, result["coverage"],
        vmax=vmax, fov_deg=render_mod.fov_deg_from_wcs(ref_wcs, ref_shape), invert=not args.dark,
        title=feature["symbol"], wcs=ref_wcs, cmap=feature.get("cmap"),
        reticle_color=feature.get("reticle_color", "red"),
    )
    print(f"[DONE] {fits_path}")
    print(f"[DONE] {png_path}")

    if feature.get("cmap") is not None:
        bw_png_path = out_dir / f"{file_prefix}_bw.png"
        render_mod.render_tau_map(
            result["tau"], bw_png_path, index, mjd_min, mjd_max, result["coverage"],
            vmax=vmax, fov_deg=render_mod.fov_deg_from_wcs(ref_wcs, ref_shape), invert=not args.dark,
            title=feature["symbol"], wcs=ref_wcs,
        )
        print(f"[DONE] {bw_png_path}")

    if not args.no_contour:
        level = args.contour_level
        if level is None:
            level = feature.get("contour_level")
        if level is None:
            level = contour_mod.default_contour_level(result["tau"], feature["kind"])
        mask = contour_mod.compute_contour_mask(result["tau"], level)
        contour_fits_path = out_dir / f"{file_prefix}_contour.fits"
        contour_png_path = out_dir / f"{file_prefix}_contour.png"
        contour_mod.save_contour_fits(
            contour_fits_path, mask, ref_wcs, level, feature["label"], feature["kind"], units,
            index, mjd_min, mjd_max,
        )
        contour_mod.render_contour_map(
            result["tau"], level, contour_png_path, index, mjd_min, mjd_max,
            result["coverage"], units, vmax=vmax, fov_deg=render_mod.fov_deg_from_wcs(ref_wcs, ref_shape),
            invert=not args.dark, title=feature["symbol"], wcs=ref_wcs, cmap=feature.get("cmap"),
            reticle_color=feature.get("reticle_color", "red"),
        )
        print(f"[DONE] {contour_fits_path} (level={level:.3g} {units}, "
              f"{100 * float(mask.sum()) / mask.size:.1f}% of pixels at/above it)")
        print(f"[DONE] {contour_png_path}")

        if feature.get("cmap") is not None:
            bw_contour_png_path = out_dir / f"{file_prefix}_contour_bw.png"
            contour_mod.render_contour_map(
                result["tau"], level, bw_contour_png_path, index, mjd_min, mjd_max,
                result["coverage"], units, vmax=vmax,
                fov_deg=render_mod.fov_deg_from_wcs(ref_wcs, ref_shape),
                invert=not args.dark, title=feature["symbol"], wcs=ref_wcs,
            )
            print(f"[DONE] {bw_contour_png_path}")

    return result


def _build_and_save_epoch_map(
    epoch: epochs_mod.Epoch,
    bad_bits: list[str],
    args: argparse.Namespace,
    ref_wcs,
    ref_shape,
    out_dir: Path,
    feature: dict,
    write_outputs: bool = True,
) -> dict:
    print(f"[INFO] --- Epoch {epoch.index} ({len(epoch.files)} files, "
          f"MJD {epoch.mjd_min:.2f}-{epoch.mjd_max:.2f}) -- {feature['label']} ({feature['kind']}) ---")
    result = tau_mod.build_epoch_tau_map(
        epoch.files,
        bad_bits,
        clean=not args.no_clean,
        ref_wcs=ref_wcs,
        ref_shape=ref_shape,
        lam_line=feature["lam_line"],
        dlambda_line=feature["dlambda_line"],
        cont_blue_lo=feature["cont_blue_lo"],
        cont_blue_hi=feature["cont_blue_hi"],
        cont_red_lo=feature["cont_red_lo"],
        cont_red_hi=feature["cont_red_hi"],
        match_background=not args.no_match_background,
        min_overlap_px=args.min_overlap_px,
        weighting=args.weighting,
        reject_outliers=not args.no_reject_outliers,
        min_weight=args.min_weight,
        kind=feature["kind"],
    )
    return _save_and_render_result(
        result, feature, args, ref_wcs, ref_shape, out_dir,
        epoch.index, epoch.mjd_min, epoch.mjd_max, f"tau_epoch{epoch.index:02d}", write_outputs,
    )


def _build_and_save_median_map(
    results_with_meta: list[tuple[int, float, float, dict]],
    args: argparse.Namespace,
    ref_wcs,
    ref_shape,
    out_dir: Path,
    feature: dict,
) -> Path:
    """Combine several epochs' already-built results into one median map (see
    :mod:`spherex_ice.combine`) and save it the same way a single epoch's map is saved.

    ``results_with_meta`` is a list of ``(epoch_index, mjd_min, mjd_max, result)``
    (works the same whether those results came from raw exposures or a compact cube).

    Returns the path to the saved ``tau_median.fits``.
    """
    results = [r for _i, _mn, _mx, r in results_with_meta]
    epoch_indices = [i for i, _mn, _mx, _r in results_with_meta]
    mjd_min = min(mn for _i, mn, _mx, _r in results_with_meta)
    mjd_max = max(mx for _i, _mn, mx, _r in results_with_meta)

    print(f"[INFO] --- Median across epochs {epoch_indices} -- {feature['label']} ({feature['kind']}) ---")
    combined = combine_mod.combine_epochs_median(results)
    print(f"[INFO] Median map: coverage={100 * combined['coverage']:.1f}%")

    vmax = args.vmax
    if vmax is None:
        finite = combined["tau"][np.isfinite(combined["tau"])]
        pct = tau_mod.default_vmax_percentile(feature["kind"])
        vmax = float(np.percentile(finite, pct)) if finite.size else 1.0

    units = "tau" if feature["kind"] == "absorption" else "MJy/sr"
    title = feature["symbol"]

    fits_path = out_dir / "tau_median.fits"
    png_path = out_dir / "tau_median.png"
    combine_mod.save_median_fits(
        fits_path, combined, ref_wcs, feature["label"], feature["kind"], units,
        epoch_indices, mjd_min, mjd_max,
    )
    render_mod.render_tau_map(
        combined["tau"], png_path, "median", mjd_min, mjd_max, combined["coverage"],
        vmax=vmax, fov_deg=render_mod.fov_deg_from_wcs(ref_wcs, ref_shape), invert=not args.dark, title=title,
        wcs=ref_wcs, cmap=feature.get("cmap"), reticle_color=feature.get("reticle_color", "red"),
    )
    print(f"[DONE] {fits_path}")
    print(f"[DONE] {png_path}")

    if feature.get("cmap") is not None:
        bw_png_path = out_dir / "tau_median_bw.png"
        render_mod.render_tau_map(
            combined["tau"], bw_png_path, "median", mjd_min, mjd_max, combined["coverage"],
            vmax=vmax, fov_deg=render_mod.fov_deg_from_wcs(ref_wcs, ref_shape), invert=not args.dark,
            title=title, wcs=ref_wcs,
        )
        print(f"[DONE] {bw_png_path}")

    if not args.no_contour:
        level = args.contour_level
        if level is None:
            level = feature.get("contour_level")
        if level is None:
            level = contour_mod.default_contour_level(combined["tau"], feature["kind"])
        mask = contour_mod.compute_contour_mask(combined["tau"], level)
        contour_fits_path = out_dir / "tau_median_contour.fits"
        contour_png_path = out_dir / "tau_median_contour.png"
        contour_mod.save_contour_fits(
            contour_fits_path, mask, ref_wcs, level, feature["label"], feature["kind"], units,
            "median", mjd_min, mjd_max,
        )
        contour_mod.render_contour_map(
            combined["tau"], level, contour_png_path, "median", mjd_min, mjd_max,
            combined["coverage"], units, vmax=vmax, fov_deg=render_mod.fov_deg_from_wcs(ref_wcs, ref_shape),
            invert=not args.dark, title=title, wcs=ref_wcs, cmap=feature.get("cmap"),
            reticle_color=feature.get("reticle_color", "red"),
        )
        print(f"[DONE] {contour_fits_path} (level={level:.3g} {units}, "
              f"{100 * float(mask.sum()) / mask.size:.1f}% of pixels at/above it)")
        print(f"[DONE] {contour_png_path}")

        if feature.get("cmap") is not None:
            bw_contour_png_path = out_dir / "tau_median_contour_bw.png"
            contour_mod.render_contour_map(
                combined["tau"], level, bw_contour_png_path, "median", mjd_min, mjd_max,
                combined["coverage"], units, vmax=vmax,
                fov_deg=render_mod.fov_deg_from_wcs(ref_wcs, ref_shape),
                invert=not args.dark, title=title, wcs=ref_wcs,
            )
            print(f"[DONE] {bw_contour_png_path}")

    return fits_path


def _process_all_features(
    epochs_to_process: list[epochs_mod.Epoch],
    bad_bits: list[str],
    args: argparse.Namespace,
    ref_wcs,
    ref_shape,
    base_out_dir: Path,
) -> dict[str, Path | None]:
    """Build every requested ``--feature``'s map(s) (+ median), each in its own
    subdirectory of ``base_out_dir`` when more than one feature was requested (otherwise
    straight in ``base_out_dir``, same as before --feature took more than one value).

    Returns ``{feature_name: tau_median.fits path, or None if no median was built}``.
    """
    multi = len(args.feature) > 1
    median_paths: dict[str, Path | None] = {}
    # epoch.index -> list of band dicts, accumulated across features at zero extra
    # reprojection cost (each feature's blue/line/red mosaics are already computed
    # for its own map) -- only used if --cube was given; see cube.py.
    cube_bands_by_epoch: dict[int, list[dict]] = {e.index: [] for e in epochs_to_process}

    for name in args.feature:
        feature = _resolve_feature(args, name)
        out_dir = (base_out_dir / name) if multi else base_out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        if multi:
            print(f"[INFO] === Feature '{name}' -> {out_dir} ===")
        epoch_results = [
            (
                epoch,
                _build_and_save_epoch_map(
                    epoch, bad_bits, args, ref_wcs, ref_shape, out_dir, feature,
                    write_outputs=not args.median_only,
                ),
            )
            for epoch in epochs_to_process
        ]
        if getattr(args, "cube", False):
            for epoch, result in epoch_results:
                cube_bands_by_epoch[epoch.index].extend(cube_mod.bands_from_result(name, feature, result))
        median_path = None
        if not args.no_median and (len(epoch_results) > 1 or args.median_only):
            median_path = _build_and_save_median_map(
                [(e.index, e.mjd_min, e.mjd_max, r) for e, r in epoch_results],
                args, ref_wcs, ref_shape, out_dir, feature,
            )
        median_paths[name] = median_path

    if getattr(args, "cube", False):
        cube_dir = base_out_dir / "cubes"
        cube_dir.mkdir(parents=True, exist_ok=True)
        for epoch in epochs_to_process:
            bands = cube_bands_by_epoch[epoch.index]
            cube_path = cube_dir / f"epoch{epoch.index:02d}_cube.fits"
            cube_mod.save_epoch_cube_fits(
                cube_path, bands, ref_wcs, epoch.index, epoch.mjd_min, epoch.mjd_max
            )
            print(f"[DONE] {cube_path} ({len(bands)} bands, compact reusable cube)")

    return median_paths


def _maybe_build_auto_rgb(
    args: argparse.Namespace, median_paths: dict[str, Path | None], base_out_dir: Path
) -> None:
    """If ``--auto-rgb`` was given, build a composite from each requested feature's median
    map (first=red, second=green, third=blue if given) -- see :mod:`spherex_ice.rgb`.
    """
    if not args.auto_rgb:
        return
    names = args.feature
    if len(names) not in (2, 3):
        print(f"[WARN] --auto-rgb needs exactly 2 or 3 --feature values, got {len(names)}; skipping.")
        return
    paths = [median_paths.get(n) for n in names]
    if any(p is None for p in paths):
        print("[WARN] --auto-rgb requires the median map for every feature (check --no-median); skipping.")
        return
    while len(paths) < 3:
        paths.append(None)

    result = rgb_mod.compose_rgb(*paths)
    fov_deg = render_mod.fov_deg_from_wcs(result["wcs"], result["shape"]) if result["wcs"] else None
    out_png = base_out_dir / "composite.png"
    out_fits = base_out_dir / "composite.fits"
    labels = tuple(result["meta"][i].get("feature") for i in range(3))
    rgb_mod.render_rgb_png(result, out_png, labels=labels, invert=not args.dark, fov_deg=fov_deg)
    rgb_mod.save_rgb_fits(out_fits, result)
    print(f"[DONE] {out_png}")
    print(f"[DONE] {out_fits}")


# ---------------------------------------------------------------------------
# spherex-ice-map
# ---------------------------------------------------------------------------
def map_() -> None:
    ap = argparse.ArgumentParser(
        prog="spherex-ice-map",
        description="Build a 3.05 um H2O ice peak optical depth map for one or all observation epochs.",
    )
    ap.add_argument("--epochs", required=True, help="epochs.json produced by spherex-ice-epochs.")
    roi_mod.add_roi_cli_args(ap)
    ap.add_argument("--pixscale", type=float, default=6.15, help='Output pixel scale, arcsec/px.')
    ap.add_argument("--epoch", default="all", help="Epoch index to process, or 'all' (default).")
    ap.add_argument("--out_dir", default="tau_maps")
    _add_cleaning_args(ap)
    _add_tau_args(ap)
    args = ap.parse_args()
    if args.median_only and args.no_median:
        raise SystemExit("[ERROR] --median-only and --no-median together would write nothing.")
    if args.auto_rgb and args.no_median:
        raise SystemExit("[ERROR] --auto-rgb needs the median map, which --no-median disables.")

    bad_bits = _bad_bits_from_arg(args.bad_bits)
    all_epochs = _load_epochs(args.epochs)
    targets = all_epochs if args.epoch == "all" else [e for e in all_epochs if e.index == int(args.epoch)]
    if not targets:
        raise SystemExit(f"[ERROR] No epoch '{args.epoch}' in {args.epochs}.")

    roi_box = roi_mod.roi_from_args(args)
    ref_wcs, ref_shape = mosaic_mod.build_reference_wcs(
        roi_box["ra_center"], roi_box["dec_center"],
        roi_box["width_deg"], roi_box["height_deg"], args.pixscale,
    )
    print(f"[INFO] Grid {ref_shape[1]}x{ref_shape[0]} px ({args.pixscale}\"/px).")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    median_paths = _process_all_features(targets, bad_bits, args, ref_wcs, ref_shape, out_dir)
    _maybe_build_auto_rgb(args, median_paths, out_dir)


# ---------------------------------------------------------------------------
# spherex-ice-get
# ---------------------------------------------------------------------------
def _default_out_dir(target: str | None, ra: float, dec: float) -> str:
    if target:
        slug = re.sub(r"[^a-z0-9]+", "_", target.strip().lower()).strip("_")
        if slug:
            return slug
    return f"ice_ra{ra:.4f}_dec{dec:.4f}"


def get() -> None:
    ap = argparse.ArgumentParser(
        prog="spherex-ice-get",
        description="One-command pipeline: query -> download -> group into epochs -> per-epoch "
        "3.05 um H2O ice optical depth map, for one small field.",
    )
    ap.add_argument("--target", default=None, help="Object name, resolved via SIMBAD (needs internet).")
    ap.add_argument("--ra_center", type=float, default=None)
    ap.add_argument("--dec_center", type=float, default=None)
    ap.add_argument("--size_deg", type=float, required=True, help="Square field size, in degrees.")
    ap.add_argument("--out", default=None, help="Output directory. Default: derived from --target or RA/Dec.")
    ap.add_argument("--workers", type=int, default=1, help="Parallel download workers.")
    ap.add_argument("--pixscale", type=float, default=6.15, help='Output pixel scale, arcsec/px.')
    _add_cleaning_args(ap)
    _add_epoch_args(ap)
    _add_tau_args(ap)
    ap.add_argument(
        "--delete-cutouts-after-cube", action="store_true",
        help="Once --cube has successfully written every epoch's cube, delete the raw "
        "downloaded cutouts (.spherex_ice/cutouts/) to actually reclaim the disk space -- "
        "--cube alone keeps both. The query manifest/roi.json/epochs.json/list.txt are kept "
        "either way (tiny, and needed to re-download the same field later). Requires --cube.",
    )
    args = ap.parse_args()
    if args.median_only and args.no_median:
        raise SystemExit("[ERROR] --median-only and --no-median together would write nothing.")
    if args.auto_rgb and args.no_median:
        raise SystemExit("[ERROR] --auto-rgb needs the median map, which --no-median disables.")
    if args.delete_cutouts_after_cube and not args.cube:
        raise SystemExit("[ERROR] --delete-cutouts-after-cube requires --cube.")

    if args.target:
        from spherexstack.target import resolve_target

        try:
            ra, dec, main_id = resolve_target(args.target)
        except (ValueError, RuntimeError) as exc:
            raise SystemExit(f"[ERROR] {exc}") from exc
        print(f"[INFO] Resolved '{args.target}' -> {main_id} at RA={ra:.5f} Dec={dec:.5f}")
    elif args.ra_center is not None and args.dec_center is not None:
        ra, dec = args.ra_center, args.dec_center
    else:
        raise SystemExit("[ERROR] Give --target NAME, or both --ra_center/--dec_center.")

    out_dir = Path(args.out or _default_out_dir(args.target, ra, dec)).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    # Everything that isn't a map/contour/composite -- raw downloads, the query
    # manifest, roi/epochs.json -- goes in a hidden subdirectory, so a plain `ls`
    # of --out shows only the actual products (tau_maps/...). Kept, not deleted,
    # since spherex-ice-map can reuse it later without re-downloading.
    cache_dir = out_dir / ".spherex_ice"
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directory: {out_dir}  (raw downloads/metadata in {cache_dir})")

    roi_box = roi_mod.resolve_roi(ra_center=ra, dec_center=dec, size_deg=args.size_deg)
    roi_mod.save_roi(cache_dir / "roi.json", roi_box)
    print(
        f"[INFO] ROI: RA [{roi_box['ra_min']:.4f}, {roi_box['ra_max']:.4f}]  "
        f"Dec [{roi_box['dec_min']:.4f}, {roi_box['dec_max']:.4f}]"
    )

    # Computed now (not just after downloading) so the size/cost warning below
    # lands before any network traffic, not after.
    ref_wcs, ref_shape = mosaic_mod.build_reference_wcs(
        roi_box["ra_center"], roi_box["dec_center"],
        roi_box["width_deg"], roi_box["height_deg"], args.pixscale,
    )
    print(f"[INFO] Grid {ref_shape[1]}x{ref_shape[0]} px ({args.pixscale}\"/px).")
    if args.size_deg > query_mod.NATIVE_FOV_DEG:
        print(
            f"[WARN] --size_deg {args.size_deg:.2f} is larger than one SPHEREx detector's "
            f"native FOV (~{query_mod.NATIVE_FOV_DEG} deg) -- there is no hard limit enforced "
            "here (IRSA's API has none either, unlike its web form's 5x5 deg cap), but beyond "
            "this point /ibe/cutout stops cropping each file down (every exposure comes back "
            "close to its full ~72 MB native size instead of a small crop), AND a bigger area "
            "also means more exposures from the all-sky survey -- so both the download and the "
            "per-band background-matching step (which scales roughly with the square of the "
            "exposure count) can get slow. Consider a smaller --size_deg, or --workers/patience "
            "for a large one.",
        )

    # IRSA's TAP query filters on each exposure's reported POINTING CENTER
    # (s_ra/s_dec) falling inside the query box -- not on whether the query
    # box overlaps the exposure's actual footprint. A single SPHEREx detector
    # covers ~3.47 deg across, so an exposure can easily cover a small target
    # field while its own center sits well outside a --size_deg-sized query
    # box. To not miss those, query a box at least one native FOV wide
    # (centered on the target), then still download/crop only --size_deg.
    query_size_deg = max(args.size_deg, query_mod.NATIVE_FOV_DEG)
    query_box = roi_mod.resolve_roi(ra_center=ra, dec_center=dec, size_deg=query_size_deg)
    print(f"[INFO] Querying IRSA for exposures covering this field (query box {query_size_deg:.2f} deg)...")
    csv_text = query_mod.run_tap_query(
        query_box["ra_min"], query_box["ra_max"], query_box["dec_min"], query_box["dec_max"]
    )
    import csv as _csv

    rows = list(_csv.DictReader(csv_text.splitlines()))
    if not rows:
        raise SystemExit("[ERROR] No exposures found for this field.")
    print(f"[INFO] {len(rows)} exposures found.")
    manifest_path = cache_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    cutouts_dir = cache_dir / "cutouts"
    cutouts_dir.mkdir(exist_ok=True)
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from spherexstack.progress import ProgressBar

    print(
        f"[INFO] Downloading {len(rows)} cutouts ({args.size_deg:.3f} deg) with {args.workers} worker(s)..."
    )
    bar = ProgressBar(total=len(rows), prefix="Downloading", unit="files")
    ok, failed = 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [
            ex.submit(
                download_mod.download_one, row, roi_box["ra_center"], roi_box["dec_center"], args.size_deg,
                cutouts_dir, download_mod.KEEP_HDUS_DEFAULT,
            )
            for row in rows
        ]
        for fut in as_completed(futures):
            _path, status, _nbytes = fut.result()
            if status in ("ok", "skip"):
                ok += 1
            else:
                failed += 1
                bar.interrupt(f"[WARN] {status}")
            bar.update()
    bar.close()
    print(f"[DONE] Downloaded/skipped {ok}, failed {failed}.")

    all_files = sorted(cutouts_dir.glob("*.fits"))
    list_path = cache_dir / "list.txt"
    list_path.write_text("\n".join(str(p) for p in all_files) + "\n")

    print("[INFO] Grouping exposures into observation epochs...")
    eps = epochs_mod.group_into_epochs(
        all_files, max_gap_days=args.max_gap_days, max_span_days=args.max_span_days
    )
    for e in eps:
        print(f"[INFO]   epoch {e.index}: {len(e.files)} files, MJD {e.mjd_min:.2f}-{e.mjd_max:.2f} "
              f"(span {e.span_days:.1f} d)")
    if not eps:
        raise SystemExit("[ERROR] No exposures had a readable MJD-OBS -- cannot group into epochs.")
    epochs_path = cache_dir / "epochs.json"
    epochs_path.write_text(
        json.dumps(
            [
                {
                    "index": e.index, "mjd_min": e.mjd_min, "mjd_max": e.mjd_max,
                    "files": [str(fp) for fp in e.files],
                }
                for e in eps
            ],
            indent=2,
        )
    )

    bad_bits = _bad_bits_from_arg(args.bad_bits)
    tau_dir = out_dir / "tau_maps"
    tau_dir.mkdir(exist_ok=True)
    median_paths = _process_all_features(eps, bad_bits, args, ref_wcs, ref_shape, tau_dir)
    _maybe_build_auto_rgb(args, median_paths, tau_dir)

    print(f"[DONE] {len(eps)} epoch(s) x {len(args.feature)} feature(s) in {tau_dir}")

    if args.delete_cutouts_after_cube:
        cube_paths = [tau_dir / "cubes" / f"epoch{e.index:02d}_cube.fits" for e in eps]
        missing = [p for p in cube_paths if not p.exists()]
        if missing:
            print(
                f"[WARN] --delete-cutouts-after-cube: {len(missing)} expected cube file(s) are "
                f"missing ({missing[0].name}, ...) -- NOT deleting {cutouts_dir} to be safe."
            )
        else:
            bad = []
            for p in cube_paths:
                try:
                    cube_mod.load_cube(p)
                except Exception as exc:  # noqa: BLE001 - any failure here means "don't delete"
                    bad.append((p, exc))
            if bad:
                print(
                    f"[WARN] --delete-cutouts-after-cube: {bad[0][0].name} failed to re-load "
                    f"({bad[0][1]}) -- NOT deleting {cutouts_dir} to be safe."
                )
            else:
                n_files = sum(1 for _ in cutouts_dir.glob("*.fits"))
                shutil.rmtree(cutouts_dir)
                print(
                    f"[DONE] Verified all {len(cube_paths)} cube(s) load correctly; deleted "
                    f"{n_files} raw cutout(s) from {cutouts_dir} (manifest/roi/epochs.json kept, "
                    "so the same field can be re-downloaded later if needed)."
                )


# ---------------------------------------------------------------------------
# spherex-ice-rgb
# ---------------------------------------------------------------------------
def rgb() -> None:
    ap = argparse.ArgumentParser(
        prog="spherex-ice-rgb",
        description="Combine up to three already-built feature maps (tau_epochNN.fits from "
        "spherex-ice-map/spherex-ice-get, each a different feature -- e.g. H2O tau, CO2 tau, "
        "PAH emission) into one RGB color composite. All inputs must share the same pixel "
        "grid (built from the same --roi/--pixscale).",
    )
    ap.add_argument("--red", default=None, help="tau_epochNN.fits to use as the red channel.")
    ap.add_argument("--green", default=None, help="tau_epochNN.fits to use as the green channel.")
    ap.add_argument("--blue", default=None, help="tau_epochNN.fits to use as the blue channel.")
    ap.add_argument(
        "--red_label", default=None,
        help="Legend text for the red channel. Default: that file's own FEATURE header (e.g. "
        "'3.05 um H2O ice') -- no units, just the feature name.",
    )
    ap.add_argument(
        "--green_label", default=None, help="Legend text for the green channel (see --red_label)."
    )
    ap.add_argument(
        "--blue_label", default=None, help="Legend text for the blue channel (see --red_label)."
    )
    ap.add_argument(
        "--red_vmax", type=float, default=None,
        help="Red channel clip value. Default: its own 99th percentile.",
    )
    ap.add_argument(
        "--green_vmax", type=float, default=None,
        help="Green channel clip value. Default: its own 99th percentile.",
    )
    ap.add_argument(
        "--blue_vmax", type=float, default=None,
        help="Blue channel clip value. Default: its own 99th percentile.",
    )
    ap.add_argument("--out", default="composite.png")
    ap.add_argument(
        "--out_fits", default=None,
        help="If given, also save the composite as a 3-plane FITS cube (R/G/B HDUs, shared WCS).",
    )
    ap.add_argument(
        "--dark", action="store_true",
        help="Render with a black background instead of the default white one.",
    )
    args = ap.parse_args()

    if not (args.red or args.green or args.blue):
        raise SystemExit("[ERROR] Give at least one of --red/--green/--blue.")

    result = rgb_mod.compose_rgb(
        Path(args.red) if args.red else None,
        Path(args.green) if args.green else None,
        Path(args.blue) if args.blue else None,
        red_vmax=args.red_vmax, green_vmax=args.green_vmax, blue_vmax=args.blue_vmax,
    )
    print(f"[INFO] Composite grid {result['shape'][1]}x{result['shape'][0]} px.")
    for name, vmax in zip(("red", "green", "blue"), result["vmax"], strict=True):
        if vmax is not None:
            print(f"[INFO] {name} channel clipped at {vmax:.4g}")

    given_labels = (args.red_label, args.green_label, args.blue_label)
    labels = tuple(
        given if given is not None else result["meta"][i].get("feature")
        for i, given in enumerate(given_labels)
    )

    fov_deg = render_mod.fov_deg_from_wcs(result["wcs"], result["shape"]) if result["wcs"] else None
    out_png = Path(args.out).expanduser().resolve()
    rgb_mod.render_rgb_png(
        result, out_png, labels=labels,
        invert=not args.dark, fov_deg=fov_deg,
    )
    print(f"[DONE] {out_png}")

    if args.out_fits:
        out_fits = Path(args.out_fits).expanduser().resolve()
        rgb_mod.save_rgb_fits(out_fits, result)
        print(f"[DONE] {out_fits}")


# ---------------------------------------------------------------------------
# spherex-ice-blend
# ---------------------------------------------------------------------------
def blend() -> None:
    ap = argparse.ArgumentParser(
        prog="spherex-ice-blend",
        description="Blend 2-3 already-built feature maps (tau_epochNN.fits / tau_median.fits "
        "from spherex-ice-map/spherex-ice-get), each through its own perceptual colormap (e.g. "
        "H2O's blue ice scale, PAH's red/orange/yellow heat scale -- see FEATURE_PRESETS), into "
        "one composite -- unlike spherex-ice-rgb, which stacks maps as literal R/G/B primaries, "
        "this keeps each feature's own color and combines them with a 'screen' blend. All inputs "
        "must share the same pixel grid (built from the same --roi/--pixscale).",
    )
    ap.add_argument("--map1", required=True, help="First feature map FITS file.")
    ap.add_argument("--map2", required=True, help="Second feature map FITS file.")
    ap.add_argument("--map3", default=None, help="Optional third feature map FITS file.")
    ap.add_argument(
        "--cmap1", default=None,
        help="Colormap for --map1. Default: looked up from its own FEATURE header "
        "against FEATURE_PRESETS, else plain grayscale.",
    )
    ap.add_argument("--cmap2", default=None, help="Colormap for --map2 (see --cmap1).")
    ap.add_argument("--cmap3", default=None, help="Colormap for --map3 (see --cmap1).")
    ap.add_argument("--vmax1", type=float, default=None, help="Clip value for --map1 (default: 99th pct).")
    ap.add_argument("--vmax2", type=float, default=None, help="Clip value for --map2 (default: 99th pct).")
    ap.add_argument("--vmax3", type=float, default=None, help="Clip value for --map3 (default: 99th pct).")
    ap.add_argument("--label1", default=None, help="Legend text for --map1 (default: its FEATURE header).")
    ap.add_argument("--label2", default=None, help="Legend text for --map2 (see --label1).")
    ap.add_argument("--label3", default=None, help="Legend text for --map3 (see --label1).")
    ap.add_argument("--out", default="blend.png")
    args = ap.parse_args()

    paths = [Path(args.map1), Path(args.map2)]
    cmaps = [args.cmap1, args.cmap2]
    vmaxes = [args.vmax1, args.vmax2]
    given_labels = [args.label1, args.label2]
    if args.map3:
        paths.append(Path(args.map3))
        cmaps.append(args.cmap3)
        vmaxes.append(args.vmax3)
        given_labels.append(args.label3)

    result = rgb_mod.compose_blend(paths, cmaps=cmaps, vmaxes=vmaxes)
    print(f"[INFO] Blend grid {result['shape'][1]}x{result['shape'][0]} px.")
    for path, vmax, cmap in zip(paths, result["vmax"], result["cmaps"], strict=True):
        cmap_name = cmap if isinstance(cmap, str) else getattr(cmap, "name", "custom")
        print(f"[INFO] {path}: cmap={cmap_name}, clipped at {vmax:.4g}")

    labels = [
        given if given is not None else meta.get("feature")
        for given, meta in zip(given_labels, result["meta"], strict=True)
    ]
    fov_deg = render_mod.fov_deg_from_wcs(result["wcs"], result["shape"]) if result["wcs"] else None
    out_png = Path(args.out).expanduser().resolve()
    rgb_mod.render_blend_png(result, out_png, labels=labels, fov_deg=fov_deg)
    print(f"[DONE] {out_png}")


# ---------------------------------------------------------------------------
# spherex-ice-from-cube
# ---------------------------------------------------------------------------
def from_cube() -> None:
    ap = argparse.ArgumentParser(
        prog="spherex-ice-from-cube",
        description="Derive any built-in feature's map(s) (+ contour, + median, + auto-rgb) "
        "from an already-built compact cube (spherex-ice-get/spherex-ice-map --cube) -- the "
        "raw per-exposure FITS cutouts are never touched.",
    )
    ap.add_argument(
        "--input_cube", required=True,
        help="A single epochNN_cube.fits, or a directory containing several (e.g. the "
        "'cubes' folder --cube wrote) -- all must share the same pixel grid.",
    )
    ap.add_argument("--out_dir", default="tau_maps_from_cube")
    _add_tau_args(ap)
    args = ap.parse_args()
    if args.median_only and args.no_median:
        raise SystemExit("[ERROR] --median-only and --no-median together would write nothing.")
    if args.auto_rgb and args.no_median:
        raise SystemExit("[ERROR] --auto-rgb needs the median map, which --no-median disables.")

    cube_path = Path(args.input_cube).expanduser().resolve()
    cube_files = sorted(cube_path.glob("epoch*_cube.fits")) if cube_path.is_dir() else [cube_path]
    if not cube_files:
        raise SystemExit(f"[ERROR] No cube FITS found at {cube_path}.")

    cubes = [(fp, cube_mod.load_cube(fp)) for fp in cube_files]
    ref_wcs = cubes[0][1]["wcs"]
    ref_shape = cubes[0][1]["flux_cube"].shape[1:]
    print(f"[INFO] Loaded {len(cubes)} cube(s) from {cube_path}, grid {ref_shape[1]}x{ref_shape[0]} px.")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    multi = len(args.feature) > 1
    median_paths: dict[str, Path | None] = {}

    for name in args.feature:
        feature = _resolve_feature(args, name)
        feat_out_dir = (out_dir / name) if multi else out_dir
        feat_out_dir.mkdir(parents=True, exist_ok=True)
        if multi:
            print(f"[INFO] === Feature '{name}' -> {feat_out_dir} ===")

        results_with_meta = []
        for fp, cube in cubes:
            try:
                result = cube_mod.tau_from_cube(cube, name, feature["kind"], min_weight=args.min_weight)
            except ValueError as exc:
                raise SystemExit(f"[ERROR] {exc}") from exc
            idx = cube["epoch_index"]
            prefix = f"tau_epoch{idx:02d}" if isinstance(idx, int) else f"tau_{idx}"
            print(f"[INFO] --- {fp.name} -- {feature['label']} ({feature['kind']}) ---")
            result = _save_and_render_result(
                result, feature, args, ref_wcs, ref_shape, feat_out_dir,
                idx, cube["mjd_min"], cube["mjd_max"], prefix, write_outputs=not args.median_only,
            )
            results_with_meta.append((idx, cube["mjd_min"], cube["mjd_max"], result))

        median_path = None
        if not args.no_median and (len(results_with_meta) > 1 or args.median_only):
            median_path = _build_and_save_median_map(
                results_with_meta, args, ref_wcs, ref_shape, feat_out_dir, feature
            )
        median_paths[name] = median_path

    _maybe_build_auto_rgb(args, median_paths, out_dir)


# ---------------------------------------------------------------------------
# spherex-ice-inspect
# ---------------------------------------------------------------------------
def inspect() -> None:
    ap = argparse.ArgumentParser(
        prog="spherex-ice-inspect",
        description="Interactive viewer: click a position on one or more already-built "
        "median maps (e.g. H2O and PAH) to see its full SPHEREx spectrum (ln(flux) vs. "
        "wavelength), pulled on the spot from the raw exposure cutouts -- run this once "
        "everything is already downloaded (see spherex-ice-get/-map). Each click adds a "
        "colored marker to every map and the matching spectrum curve to a second window; "
        "a 'Guardar' button there saves every spectrum shown so far as .csv, plus a .png "
        "of each window.",
    )
    ap.add_argument(
        "--tau_dir", required=True,
        help="The --out_dir a previous spherex-ice-map/-get/-from-cube run used -- "
        "tau_median.fits for each --feature below is found under here.",
    )
    ap.add_argument(
        "--feature", nargs="+", default=["h2o"], metavar="FEATURE",
        help="Which built median map(s) to show side by side, e.g. --feature h2o pah. Each "
        "must already exist as <tau_dir>/<feature>/tau_median.fits (several features) or "
        "<tau_dir>/tau_median.fits (just one) -- i.e. whatever that prior run's own "
        "--feature produced.",
    )
    ap.add_argument(
        "--cutouts", required=True,
        help="Directory of raw exposure cutouts to extract spectra from, e.g. the "
        "<out>/.spherex_ice/cutouts that spherex-ice-get left behind.",
    )
    ap.add_argument(
        "--bad-bits",
        default=",".join(sx_io.DEFAULT_BAD_BITS),
        help="Comma list of FLAGS bits that make a cutout's pixel skipped rather than used "
        f"for a clicked position's spectrum. Available: {','.join(sx_io.FLAG_BITS.keys())}.",
    )
    ap.add_argument("--out_dir", default="spectra", help="Where the 'Guardar' button writes its files.")
    ap.add_argument(
        "--aperture_px",
        type=float,
        default=None,
        help="Switch from a single-pixel spectrum to a background-subtracted circular-"
        "aperture spectrum (radius in SPHEREx native pixels), the same aperture-plus-"
        "annulus recipe IRSA's own SPHEREx mosaic-cube tutorial uses. Useful when one "
        "pixel is too noisy to judge a feature by (an extended, diffuse source, say). "
        "Reported in mJy (converted from MJy/sr), not the default MJy/sr.",
    )
    ap.add_argument(
        "--annulus_in_px", type=float, default=None,
        help="[aperture mode] Background annulus inner radius, px. Default: 2x --aperture_px.",
    )
    ap.add_argument(
        "--annulus_out_px", type=float, default=None,
        help="[aperture mode] Background annulus outer radius, px. Default: 3x --aperture_px.",
    )
    args = ap.parse_args()

    multi = len(args.feature) > 1
    tau_dir = Path(args.tau_dir).expanduser().resolve()
    map_paths = {}
    for name in args.feature:
        path = (tau_dir / name / "tau_median.fits") if multi else (tau_dir / "tau_median.fits")
        if not path.exists():
            raise SystemExit(f"[ERROR] {path} not found -- build it first "
                              f"(spherex-ice-map --feature {name}).")
        map_paths[name] = path

    bad_bits = _bad_bits_from_arg(args.bad_bits)
    viewer_mod.run_viewer(
        map_paths,
        Path(args.cutouts).expanduser().resolve(),
        bad_bits,
        Path(args.out_dir),
        aperture_px=args.aperture_px,
        annulus_in_px=args.annulus_in_px,
        annulus_out_px=args.annulus_out_px,
    )
