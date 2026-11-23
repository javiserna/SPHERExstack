"""Console-script entry points: ``spherex-get``, ``spherex-query``, ``spherex-download``,
``spherex-mosaic``, ``spherex-merge``, ``spherex-cube``, ``spherex-animate``, ``spherex-rgb``.

``query`` and ``download`` delegate straight to :mod:`spherexstack.query` and
:mod:`spherexstack.download`, which own their own argument parsing. ``mosaic``,
``merge``, ``cube``, ``animate`` and ``rgb`` wire together
:mod:`spherexstack.mosaic`, :mod:`spherexstack.cube` and
:mod:`spherexstack.animate` to reproduce (and extend, for ``ivar`` weighting)
the behavior of the original standalone scripts. ``get`` is a higher-level
convenience wrapper for a basic user: given just a target (by name, resolved
via :mod:`spherexstack.target`, or by RA/Dec) and a field size, it runs the
query -> download -> cube steps (and optionally ``--animate``/``--rgb``) in
one call, choosing its own manifest/ROI/list file names inside an output
directory instead of asking the user to name them.
"""

from __future__ import annotations

import argparse
import csv
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

from spherexstack import cube as cube_mod
from spherexstack import download as download_mod
from spherexstack import io
from spherexstack import mosaic as mosaic_mod
from spherexstack import query as query_mod
from spherexstack import roi as roi_mod
from spherexstack import units as units_mod
from spherexstack.progress import ProgressBar


def query() -> None:
    query_mod.main()


def download() -> None:
    download_mod.main()


def _bad_bits_from_arg(arg: str) -> list[str]:
    return [b.strip() for b in arg.split(",") if b.strip()]


def _resolved_files(list_path_str: str) -> list[Path]:
    list_path = Path(list_path_str).expanduser().resolve()
    return io.resolve_list_paths(list_path)


def _default_out_dir(target: str | None, ra: float, dec: float) -> str:
    if target:
        slug = re.sub(r"[^a-z0-9]+", "_", target.strip().lower()).strip("_")
        if slug:
            return slug
    return f"field_ra{ra:.4f}_dec{dec:.4f}"


def _add_cleaning_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--no-clean", action="store_true", help="Disable FLAGS-based hot pixel/outlier cleaning."
    )
    ap.add_argument(
        "--bad-bits",
        default=",".join(io.DEFAULT_BAD_BITS),
        help=f"Comma list of FLAGS bits to treat as defects. Available: {','.join(io.FLAG_BITS.keys())}.",
    )


def _add_weighting_arg(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--weighting",
        choices=mosaic_mod.WEIGHTING_CHOICES,
        default="mean",
        help="Per-pixel coaddition weighting. 'mean' (default) is an unweighted average, "
        "matching the original behavior. 'ivar' weights by 1/VARIANCE (falls back to 'mean' "
        "for exposures without a usable VARIANCE HDU -- re-download with --keep-hdus including "
        "VARIANCE to use this).",
    )


def _add_units_arg(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--units",
        choices=units_mod.UNIT_CHOICES,
        default="mjy_sr",
        help="Flux unit for the FITS output. 'mjy_sr' (default) keeps SPHEREx's native MJy/sr "
        "surface brightness. 'mjy' converts to a per-pixel flux density in mJy, the same "
        "conversion IRSA's own SPHEREx mosaic-cube tutorial uses before aperture photometry.",
    )


def _add_grayscale_arg(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--grayscale",
        action="store_true",
        help="Render the image itself in plain black/white intensity instead of tinting it by "
        "wavelength (the colorbar, with its moving marker, still shows where you are in the "
        "spectrum either way). The eye often picks out faint structure more easily in grayscale "
        "than in a single hue.",
    )
    ap.add_argument(
        "--invert",
        action="store_true",
        help="Invert the rendered image like a photographic negative: what would be black is "
        "white and what would be white is black (background, text and the colorbar's own ticks "
        "swap too; the colorbar's rainbow gradient itself is left unchanged). Combine with "
        "--grayscale for a classic black-on-white (or white-on-black) look.",
    )


# ---------------------------------------------------------------------------
# spherex-mosaic
# ---------------------------------------------------------------------------
def mosaic() -> None:
    ap = argparse.ArgumentParser(
        prog="spherex-mosaic", description="Build a single-wavelength mosaic (FITS + PNG preview)."
    )
    ap.add_argument("--list", required=True, help="Path to list.txt (one FITS path per line)")
    ap.add_argument("--out", default="mosaic_lambda.png")
    ap.add_argument(
        "--out_fits",
        default=None,
        help="If given, also save the mosaic (+ weight/coverage map + WCS) as FITS -- needed to "
        "merge it later with spherex-merge alongside other sub-regions.",
    )
    roi_mod.add_roi_cli_args(ap)
    ap.add_argument(
        "--pixscale", type=float, default=6.15, help='Output pixel scale, arcsec/px (default: native ~6.15"/px).'
    )
    ap.add_argument("--lambda_um", type=float, required=True, help="Target wavelength, in microns (e.g. 3.4 for PAH).")
    ap.add_argument(
        "--dlambda",
        type=float,
        default=0.05,
        help="Half-width (um) of the synthetic band around --lambda_um. Must be >= the --dlambda_um "
        "used when filtering the manifest, if any.",
    )
    ap.add_argument("--p_lo", type=float, default=1.0)
    ap.add_argument("--p_hi", type=float, default=99.0)
    ap.add_argument(
        "--no-match-background", action="store_true", help="Disable incremental background offset matching."
    )
    ap.add_argument("--min_overlap_px", type=int, default=30)
    _add_cleaning_args(ap)
    _add_weighting_arg(ap)
    _add_grayscale_arg(ap)
    _add_units_arg(ap)
    args = ap.parse_args()

    bad_bits = _bad_bits_from_arg(args.bad_bits)
    resolved = _resolved_files(args.list)

    roi_box = roi_mod.roi_from_args(args)
    ref_wcs, ref_shape = mosaic_mod.build_reference_wcs(
        roi_box["ra_center"], roi_box["dec_center"], roi_box["width_deg"], roi_box["height_deg"], args.pixscale
    )
    ram_gb = ref_shape[0] * ref_shape[1] * 8 / 1e9  # 1 bin: float32 sum + float32 weight
    print(
        f"[INFO] Grid {ref_shape[1]}x{ref_shape[0]} px ({args.pixscale}\"/px), "
        f"mosaic at lambda={args.lambda_um} +/- {args.dlambda} um. Accumulator RAM (1 bin): ~{ram_gb:.2f} GB."
    )

    bin_centers = np.array([args.lambda_um])
    mosaics, coverages, weights = mosaic_mod.build_lambda_mosaics_streaming(
        resolved,
        bad_bits,
        clean=not args.no_clean,
        ref_wcs=ref_wcs,
        ref_shape=ref_shape,
        bin_centers=bin_centers,
        dlambda=args.dlambda,
        match_background=not args.no_match_background,
        min_overlap_px=args.min_overlap_px,
        weighting=args.weighting,
    )
    m, cov, w = mosaics[0], coverages[0], weights[0]
    if cov == 0.0:
        raise SystemExit(
            f"[ERROR] Zero coverage at lambda={args.lambda_um} um -- no exposure contributed a "
            "single pixel. Check that the FITS list, ROI and --lambda_um/--dlambda overlap."
        )

    m, bunit = units_mod.convert_units(m, args.units, ref_wcs)

    from spherexstack import animate

    vmin, vmax = animate.robust_limits([m], p_lo=args.p_lo, p_hi=args.p_hi)
    print(f"[INFO] coverage={100 * cov:.1f}%  robust vmin/vmax={vmin:.4g}/{vmax:.4g}")

    out_path = Path(args.out).expanduser().resolve()
    animate.render_lambda_slice(
        m,
        out_path,
        vmin=vmin,
        vmax=vmax,
        lam_center=args.lambda_um,
        dlambda=args.dlambda,
        coverage_frac=cov,
        fov_deg=animate.fov_deg_from_wcs(ref_wcs, ref_shape),
        grayscale=args.grayscale,
        invert=args.invert,
    )
    print(f"[DONE] {out_path}")

    if args.out_fits:
        fits_path = Path(args.out_fits).expanduser().resolve()
        mosaic_mod.save_mosaic_fits(
            fits_path,
            m,
            w,
            ref_wcs,
            lambda_um=args.lambda_um,
            dlambda=args.dlambda,
            extra_header={
                "RAMIN": roi_box["ra_min"],
                "RAMAX": roi_box["ra_max"],
                "DECMIN": roi_box["dec_min"],
                "DECMAX": roi_box["dec_max"],
                "COVERAGE": cov,
            },
            bunit=bunit,
        )
        print(f"[DONE] {fits_path} (MOSAIC + WEIGHT, {bunit}, for merging with spherex-merge)")


# ---------------------------------------------------------------------------
# spherex-merge
# ---------------------------------------------------------------------------
def merge() -> None:
    ap = argparse.ArgumentParser(prog="spherex-merge", description="Merge partial mosaics into one final mosaic.")
    ap.add_argument(
        "--tiles", nargs="+", required=True, help="Paths to the partial mosaic FITS (--out_fits of spherex-mosaic)."
    )
    roi_mod.add_roi_cli_args(ap)
    ap.add_argument("--pixscale", type=float, default=6.15, help="Must match the --pixscale used for the tiles.")
    ap.add_argument("--out", required=True, help="Final merged FITS (MOSAIC + WEIGHT).")
    ap.add_argument("--out_png", default=None, help="Optional preview PNG of the final mosaic.")
    ap.add_argument("--p_lo", type=float, default=1.0)
    ap.add_argument("--p_hi", type=float, default=99.0)
    ap.add_argument(
        "--no-match-background",
        action="store_true",
        help="Disable background pedestal matching between tiles (enabled by default).",
    )
    ap.add_argument(
        "--min_overlap_px",
        type=int,
        default=200,
        help="Minimum overlap pixels between tiles to trust their relative offset.",
    )
    _add_grayscale_arg(ap)
    args = ap.parse_args()

    roi_box = roi_mod.roi_from_args(args)
    ref_wcs, ref_shape = mosaic_mod.build_reference_wcs(
        roi_box["ra_center"], roi_box["dec_center"], roi_box["width_deg"], roi_box["height_deg"], args.pixscale
    )
    ny, nx = ref_shape
    print(f"[INFO] Final grid {nx}x{ny} px ({args.pixscale}\"/px). Accumulator RAM: ~{ny * nx * 8 / 1e9:.2f} GB.")

    tile_paths = [Path(tp).expanduser().resolve() for tp in args.tiles]
    final_mosaic, weight_acc, lambda_um, dlambda_um, coverage = mosaic_mod.build_merged_mosaic(
        tile_paths,
        ref_wcs,
        ref_shape,
        match_background=not args.no_match_background,
        min_overlap_px=args.min_overlap_px,
    )
    print(f"[INFO] Final coverage: {100 * coverage:.1f}%")
    if coverage == 0.0:
        raise SystemExit("[ERROR] Zero coverage after merging -- no tile overlapped the final grid.")

    out_path = Path(args.out).expanduser().resolve()
    mosaic_mod.save_mosaic_fits(
        out_path,
        final_mosaic,
        weight_acc,
        ref_wcs,
        lambda_um=lambda_um or 0.0,
        dlambda=dlambda_um or 0.0,
        extra_header={
            "RAMIN": roi_box["ra_min"],
            "RAMAX": roi_box["ra_max"],
            "DECMIN": roi_box["dec_min"],
            "DECMAX": roi_box["dec_max"],
            "COVERAGE": coverage,
            "NTILES": len(args.tiles),
        },
    )
    print(f"[DONE] {out_path}")

    if args.out_png:
        from spherexstack import animate

        vmin, vmax = animate.robust_limits([final_mosaic], p_lo=args.p_lo, p_hi=args.p_hi)
        png_path = Path(args.out_png).expanduser().resolve()
        animate.render_lambda_slice(
            final_mosaic,
            png_path,
            vmin=vmin,
            vmax=vmax,
            lam_center=lambda_um or 0.0,
            dlambda=dlambda_um or 0.0,
            coverage_frac=coverage,
            fov_deg=animate.fov_deg_from_wcs(ref_wcs, ref_shape),
            grayscale=args.grayscale,
            invert=args.invert,
        )
        print(f"[DONE] {png_path}")


# ---------------------------------------------------------------------------
# spherex-rgb
# ---------------------------------------------------------------------------
def rgb() -> None:
    ap = argparse.ArgumentParser(
        prog="spherex-rgb", description="Build an RGB composite image from three wavelengths."
    )
    ap.add_argument("--list", required=True, help="Path to list.txt (one FITS path per line)")
    ap.add_argument("--out", default="rgb.png")
    roi_mod.add_roi_cli_args(ap)
    ap.add_argument("--pixscale", type=float, default=6.15, help='Output pixel scale, arcsec/px (default ~6.15"/px).')
    ap.add_argument("--lambda_r", type=float, required=True, help="Wavelength (um) mapped to the red channel.")
    ap.add_argument("--lambda_g", type=float, required=True, help="Wavelength (um) mapped to the green channel.")
    ap.add_argument("--lambda_b", type=float, required=True, help="Wavelength (um) mapped to the blue channel.")
    ap.add_argument(
        "--dlambda", type=float, default=0.1, help="Half-width (um) of the synthetic band around each wavelength."
    )
    ap.add_argument("--p_lo", type=float, default=1.0, help="Low percentile for each channel's own stretch.")
    ap.add_argument("--p_hi", type=float, default=99.0, help="High percentile for each channel's own stretch.")
    ap.add_argument("--no-match-background", action="store_true")
    ap.add_argument("--min_overlap_px", type=int, default=30)
    _add_cleaning_args(ap)
    _add_weighting_arg(ap)
    args = ap.parse_args()

    bad_bits = _bad_bits_from_arg(args.bad_bits)
    resolved = _resolved_files(args.list)

    roi_box = roi_mod.roi_from_args(args)
    ref_wcs, ref_shape = mosaic_mod.build_reference_wcs(
        roi_box["ra_center"], roi_box["dec_center"], roi_box["width_deg"], roi_box["height_deg"], args.pixscale
    )

    bin_centers = np.array([args.lambda_r, args.lambda_g, args.lambda_b])
    mosaics, coverages, _weights = mosaic_mod.build_lambda_mosaics_streaming(
        resolved,
        bad_bits,
        clean=not args.no_clean,
        ref_wcs=ref_wcs,
        ref_shape=ref_shape,
        bin_centers=bin_centers,
        dlambda=args.dlambda,
        match_background=not args.no_match_background,
        min_overlap_px=args.min_overlap_px,
        weighting=args.weighting,
    )
    for channel_name, lam, cov in zip(("R", "G", "B"), bin_centers, coverages):
        print(f"[INFO] {channel_name} lambda={lam:.4f} um  coverage={100 * cov:.1f}%")
        if cov == 0.0:
            print(f"[WARN] Zero coverage for the {channel_name} channel -- it will render fully black.")

    from spherexstack import animate

    out_path = Path(args.out).expanduser().resolve()
    animate.render_rgb_composite(
        mosaics[0],
        mosaics[1],
        mosaics[2],
        out_path,
        wavelengths_um=(args.lambda_r, args.lambda_g, args.lambda_b),
        p_lo=args.p_lo,
        p_hi=args.p_hi,
        fov_deg=animate.fov_deg_from_wcs(ref_wcs, ref_shape),
    )
    print(f"[DONE] {out_path}")


def _roi_given(args: argparse.Namespace) -> bool:
    return bool(args.roi) or any(
        v is not None
        for v in (args.ra_center, args.dec_center, args.size_deg, args.ra_min, args.ra_max, args.dec_min, args.dec_max)
    )


def _parse_target(target_str: str) -> SkyCoord:
    return SkyCoord(target_str, frame="icrs", unit=(u.hourangle, u.deg))


# ---------------------------------------------------------------------------
# spherex-cube
# ---------------------------------------------------------------------------
def cube() -> None:
    ap = argparse.ArgumentParser(
        prog="spherex-cube", description="Build a (RA, Dec, lambda) spectral cube FITS from SPHEREx cutouts."
    )
    ap.add_argument("--list", required=True, help="Path to list.txt (one FITS path per line)")
    ap.add_argument("--out", required=True, help="Output cube FITS path (CUBE + WEIGHT + COVERAGE HDUs).")

    ap.add_argument(
        "--target",
        default=None,
        help='Point-target mode: e.g. "4h41m0.00s +25d52m00.0s". Crops a --cutout px window around '
        "this point from each FITS (full frames).",
    )
    ap.add_argument("--cutout", type=int, default=300, help="[point-target mode] cutout size in pixels.")

    roi_mod.add_roi_cli_args(ap)
    ap.add_argument(
        "--pixscale", type=float, default=6.15, help='[ROI mode] output pixel scale, arcsec/px (default ~6.15"/px).'
    )

    ap.add_argument("--n_bins", type=int, default=40, help="Number of wavelength slices to reconstruct.")
    ap.add_argument(
        "--dlambda",
        type=float,
        default=0.1,
        help="Half-width (um) of the synthetic band around each target wavelength. The default "
        "(40 bins, 0.1 um) is chosen to map the cube smoothly for a typical ROI.",
    )
    ap.add_argument(
        "--no-match-background", action="store_true", help="Disable background offset matching between exposures."
    )
    ap.add_argument("--min_overlap_px", type=int, default=30)
    _add_cleaning_args(ap)
    _add_weighting_arg(ap)
    _add_units_arg(ap)
    args = ap.parse_args()

    bad_bits = _bad_bits_from_arg(args.bad_bits)
    resolved = _resolved_files(args.list)

    roi_given = _roi_given(args)
    if roi_given and args.target is not None:
        raise SystemExit("[ERROR] Don't mix --target (point mode) with an ROI (box mode).")
    if not roi_given and args.target is None:
        raise SystemExit("[ERROR] Give --target (point mode) or --roi/--ra_center.../--ra_min... (ROI mode).")

    extra_header = {}
    if roi_given:
        roi_box = roi_mod.roi_from_args(args)
        ref_wcs, ref_shape = mosaic_mod.build_reference_wcs(
            roi_box["ra_center"], roi_box["dec_center"], roi_box["width_deg"], roi_box["height_deg"], args.pixscale
        )
        print(
            f"[INFO] ROI mode: reference grid {ref_shape[1]}x{ref_shape[0]} px ({args.pixscale}\"/px) "
            f"centered at RA={roi_box['ra_center']:.4f}, Dec={roi_box['dec_center']:.4f}, "
            f"{roi_box['width_deg']:.2f} x {roi_box['height_deg']:.2f} deg."
        )
        extra_header = {
            "RAMIN": roi_box["ra_min"],
            "RAMAX": roi_box["ra_max"],
            "DECMIN": roi_box["dec_min"],
            "DECMAX": roi_box["dec_max"],
        }

        print(f"[INFO] Scanning wavelength range ({len(resolved)} FITS, WCS-WAVE only)...")
        lam_ranges = [io.quick_lambda_range(fp) for fp in resolved]
        lam_ranges = [r for r in lam_ranges if r is not None]
        if not lam_ranges:
            raise SystemExit("[ERROR] No FITS file has a readable WCS-WAVE.")
        global_lam_min = min(r[0] for r in lam_ranges)
        global_lam_max = max(r[1] for r in lam_ranges)
        print(f"[INFO] Global observed lambda range: {global_lam_min:.4f}-{global_lam_max:.4f} um")
        bin_centers = np.linspace(global_lam_min, global_lam_max, args.n_bins)

        print(
            f"[INFO] Accumulating {len(resolved)} exposures on the {ref_shape[1]}x{ref_shape[0]} px grid, "
            f"{args.n_bins} wavelength bins (accumulator RAM: "
            f"~{args.n_bins * ref_shape[0] * ref_shape[1] * 8 / 1e9:.2f} GB)..."
        )
        cube_arr, weight_cube, coverage = cube_mod.build_cube(
            resolved,
            bad_bits,
            clean=not args.no_clean,
            ref_wcs=ref_wcs,
            ref_shape=ref_shape,
            bin_centers=bin_centers,
            dlambda=args.dlambda,
            match_background=not args.no_match_background,
            min_overlap_px=args.min_overlap_px,
            weighting=args.weighting,
        )
    else:
        target = _parse_target(args.target)
        mosaics, coverages, weights, ref_wcs, _ref_shape, bin_centers, _all_lam = (
            mosaic_mod.build_lambda_mosaics_legacy(
                resolved,
                bad_bits,
                clean=not args.no_clean,
                target=target,
                cutout_size=args.cutout,
                bin_centers=None,
                n_bins=args.n_bins,
                dlambda=args.dlambda,
                match_background=not args.no_match_background,
                min_overlap_px=args.min_overlap_px,
                weighting=args.weighting,
            )
        )
        cube_arr, weight_cube, coverage = cube_mod.assemble_cube(mosaics, coverages, weights)

    cube_arr, bunit = units_mod.convert_units(cube_arr, args.units, ref_wcs)

    out_path = Path(args.out).expanduser().resolve()
    cube_mod.save_cube_fits(
        out_path, cube_arr, weight_cube, ref_wcs, bin_centers, dlambda=args.dlambda, coverage=coverage,
        extra_header=extra_header, bunit=bunit,
    )
    n_empty = int(np.sum(coverage == 0))
    if n_empty:
        print(f"[WARN] {n_empty}/{len(coverage)} wavelength bins have zero coverage (see the COVERAGE table).")
    print(f"[DONE] {out_path} ({bunit})")


# ---------------------------------------------------------------------------
# spherex-animate
# ---------------------------------------------------------------------------
def animate() -> None:
    ap = argparse.ArgumentParser(
        prog="spherex-animate",
        description="GIF/MP4 of wavelength slices, from a cube FITS or directly from cutouts.",
    )
    ap.add_argument("--out", default="spherex_lambda_slices.gif", help="Output .gif (or .mp4 with --mp4)")
    ap.add_argument("--fps", type=float, default=6.0)
    ap.add_argument(
        "--sec_per_frame", type=float, default=None, help="Seconds per frame (GIF only). Overrides --fps if given."
    )
    ap.add_argument("--mp4", action="store_true")
    ap.add_argument("--p_lo", type=float, default=1.0)
    ap.add_argument("--p_hi", type=float, default=99.0)

    ap.add_argument(
        "--mode",
        choices=("binned", "raw"),
        default="binned",
        help="'binned' (default): wavelength-binned monochromatic mosaics (from --from_cube, or built "
        "on the fly from --list, matching the original lambda-slice-gif script). 'raw': one frame per "
        "exposure with no coaddition, colored row-by-row by wavelength, sorted by the target's "
        "wavelength (matching the original cutout-gif-clean script); requires --target and --list.",
    )
    ap.add_argument(
        "--from_cube",
        default=None,
        help="[binned mode] Path to a cube FITS written by spherex-cube. If given, --list/ROI/--target "
        "are not needed: the cube's own slices and wavelengths are animated directly.",
    )

    ap.add_argument("--list", default=None, help="Path to list.txt (one FITS path per line)")
    ap.add_argument(
        "--target",
        default=None,
        help='Point-target mode (both --mode values): e.g. "4h41m0.00s +25d52m00.0s" for binned mode, '
        'or "RA, Dec" (comma-separated) for raw mode, matching each original script\'s format.',
    )
    ap.add_argument("--cutout", type=int, default=300, help="[point-target mode] cutout size in pixels.")

    roi_mod.add_roi_cli_args(ap)
    ap.add_argument("--pixscale", type=float, default=6.15)

    ap.add_argument("--n_bins", type=int, default=40, help="[binned mode, built from --list] number of frames.")
    ap.add_argument("--dlambda", type=float, default=0.05)
    ap.add_argument(
        "--min_coverage",
        type=float,
        default=0.0,
        help="[binned mode] Skip (with a warning) any frame whose coverage is below this fraction (0-1).",
    )
    ap.add_argument("--no-match-background", action="store_true")
    ap.add_argument("--min_overlap_px", type=int, default=30)
    _add_cleaning_args(ap)
    _add_weighting_arg(ap)
    _add_grayscale_arg(ap)
    args = ap.parse_args()

    from spherexstack import animate as animate_mod

    bad_bits = _bad_bits_from_arg(args.bad_bits)

    if args.mode == "raw":
        if args.list is None or args.target is None:
            raise SystemExit("[ERROR] --mode raw needs both --list and --target.")
        _animate_raw(args, bad_bits, animate_mod)
        return

    if args.from_cube:
        _animate_from_cube(args, animate_mod)
        return

    if args.list is None:
        raise SystemExit("[ERROR] Give --from_cube, or --list plus --target/--roi (to build the cube on the fly).")
    _animate_binned_from_list(args, bad_bits, animate_mod)


def _animate_from_cube(args: argparse.Namespace, animate_mod) -> None:
    cube_path = Path(args.from_cube).expanduser().resolve()
    cube_arr, _weight_cube, wcs, bin_centers, coverage, dlambda_um = cube_mod.load_cube_fits(cube_path)
    dlambda = dlambda_um if dlambda_um is not None else args.dlambda
    _render_and_write(
        args, animate_mod, list(cube_arr), list(coverage), bin_centers, dlambda, out_dir=cube_path.parent, wcs=wcs
    )


def _animate_binned_from_list(args: argparse.Namespace, bad_bits: list[str], animate_mod) -> None:
    list_path = Path(args.list).expanduser().resolve()
    resolved = io.resolve_list_paths(list_path)
    out_dir = list_path.parent / "_frames_lambda_slices"

    roi_given = _roi_given(args)
    if roi_given:
        if args.target is not None:
            raise SystemExit("[ERROR] Don't mix --target with an ROI.")
        roi_box = roi_mod.roi_from_args(args)
        ref_wcs, ref_shape = mosaic_mod.build_reference_wcs(
            roi_box["ra_center"], roi_box["dec_center"], roi_box["width_deg"], roi_box["height_deg"], args.pixscale
        )
        print(f"[INFO] ROI mode: reference grid {ref_shape[1]}x{ref_shape[0]} px ({args.pixscale}\"/px).")

        lam_ranges = [io.quick_lambda_range(fp) for fp in resolved]
        lam_ranges = [r for r in lam_ranges if r is not None]
        if not lam_ranges:
            raise SystemExit("[ERROR] No FITS file has a readable WCS-WAVE.")
        global_lam_min = min(r[0] for r in lam_ranges)
        global_lam_max = max(r[1] for r in lam_ranges)
        bin_centers = np.linspace(global_lam_min, global_lam_max, args.n_bins)
        print(f"[INFO] Global observed lambda range: {global_lam_min:.4f}-{global_lam_max:.4f} um")

        mosaics, coverages, _weights = mosaic_mod.build_lambda_mosaics_streaming(
            resolved,
            bad_bits,
            clean=not args.no_clean,
            ref_wcs=ref_wcs,
            ref_shape=ref_shape,
            bin_centers=bin_centers,
            dlambda=args.dlambda,
            match_background=not args.no_match_background,
            min_overlap_px=args.min_overlap_px,
            weighting=args.weighting,
        )
    else:
        if args.target is None:
            raise SystemExit("[ERROR] Give --target or an ROI (--roi/--ra_center.../--ra_min...).")
        target = _parse_target(args.target)
        mosaics, coverages, _weights, ref_wcs, _ref_shape, bin_centers, _all_lam = (
            mosaic_mod.build_lambda_mosaics_legacy(
                resolved,
                bad_bits,
                clean=not args.no_clean,
                target=target,
                cutout_size=args.cutout,
                bin_centers=None,
                n_bins=args.n_bins,
                dlambda=args.dlambda,
                match_background=not args.no_match_background,
                min_overlap_px=args.min_overlap_px,
                weighting=args.weighting,
            )
        )

    _render_and_write(args, animate_mod, mosaics, coverages, bin_centers, args.dlambda, out_dir=out_dir, wcs=ref_wcs)


def _render_and_write(args, animate_mod, mosaics, coverages, bin_centers, dlambda, out_dir: Path, wcs=None) -> None:
    vmin, vmax = animate_mod.robust_limits(mosaics, p_lo=args.p_lo, p_hi=args.p_hi)
    print(f"[INFO] Global robust vmin/vmax: {vmin:.6g}, {vmax:.6g}")

    fov_deg = animate_mod.fov_deg_from_wcs(wcs, mosaics[0].shape) if wcs is not None else None

    out_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = []
    for i, (lam0, mosaic_slice, cov) in enumerate(zip(bin_centers, mosaics, coverages)):
        if cov < args.min_coverage:
            print(f"[WARN] lambda={lam0:.4f} um coverage {100 * cov:.1f}% < min_coverage, skipping.")
            continue
        out_png = out_dir / f"lslice_{i:04d}.png"
        animate_mod.render_lambda_slice(
            mosaic_slice,
            out_png,
            vmin=vmin,
            vmax=vmax,
            lam_center=float(lam0),
            dlambda=dlambda,
            coverage_frac=cov,
            fov_deg=fov_deg,
            grayscale=args.grayscale,
            invert=args.invert,
        )
        frame_paths.append(out_png)

    if not frame_paths:
        raise SystemExit("[ERROR] No frame passed min_coverage; nothing to animate.")

    out_path = Path(args.out).expanduser().resolve()
    animate_mod.write_animation(frame_paths, out_path, fps=args.fps, sec_per_frame=args.sec_per_frame, mp4=args.mp4)
    print(f"[INFO] PNG frames in: {out_dir}")


def _animate_raw(args: argparse.Namespace, bad_bits: list[str], animate_mod) -> None:
    """Raw per-exposure animation: one frame per exposure, no wavelength binning/coaddition."""
    list_path = Path(args.list).expanduser().resolve()
    resolved = io.resolve_list_paths(list_path)
    out_dir = list_path.parent / "_frames_raw"

    ra_str, dec_str = (s.strip() for s in args.target.split(","))
    target = SkyCoord(ra_str, dec_str, frame="icrs", unit=(u.hourangle, u.deg))

    lambdas = [io.central_lambda_um(fp) for fp in resolved]
    lam_valid = np.array([lam for lam in lambdas if np.isfinite(lam)], dtype=float)
    if lam_valid.size == 0:
        lam_min, lam_max = 0.0, 1.0
    else:
        lam_min, lam_max = float(np.nanmin(lam_valid)), float(np.nanmax(lam_valid))
        if lam_min == lam_max:
            lam_max = lam_min + 1e-6

    cutouts = []
    cutout_wcs_list = []
    cutout_objs = []
    total_bad = 0
    total_px = 0

    for fp in resolved:
        img, w = io.get_image_and_wcs(fp, image_ext="IMAGE")
        if not args.no_clean:
            flags = io.get_flags(fp, flags_ext="FLAGS")
            if flags is not None and flags.shape == img.shape:
                bad_mask = io.bad_pixel_mask_from_flags(flags, bad_bits)
                img, n_bad = io.clean_hot_pixels(img, bad_mask)
                total_bad += n_bad
                total_px += img.size
            else:
                print(f"[WARN] No compatible FLAGS HDU in {fp.name}; not cleaning this frame.")

        from astropy.nddata import Cutout2D

        co = Cutout2D(img, position=target, size=(args.cutout, args.cutout), wcs=w, mode="partial", fill_value=np.nan)
        cutouts.append(co.data)
        cutout_wcs_list.append(co.wcs)
        cutout_objs.append(co)

    if not args.no_clean and total_px:
        print(
            f"[INFO] FLAGS cleaning: {total_bad}/{total_px} px flagged as defects "
            f"({100.0 * total_bad / total_px:.2f}%) using bits: {','.join(bad_bits)}"
        )

    ref_data = cutouts[0]
    ref_wcs = cutout_wcs_list[0]
    ref_shape = ref_data.shape

    ny_out = cutouts[0].shape[0]
    lam_y_list = []
    for fp, co in zip(resolved, cutout_objs):
        lam_y = np.full(ny_out, np.nan, dtype=float)
        ysl_orig = co.slices_original[0]
        ysl_cut = co.slices_cutout[0]
        y_pix_abs = np.arange(ysl_orig.start, ysl_orig.stop, dtype=float)
        lam_part = io.lambda_axis_from_wcs_wave(fp, ny_out=len(y_pix_abs), y_pix=y_pix_abs)
        lam_y[ysl_cut.start : ysl_cut.stop] = lam_part
        lam_y_list.append(lam_y)

    def target_lam_key(lam_y: np.ndarray) -> float:
        lam_y = np.asarray(lam_y, dtype=float)
        if lam_y.size == 0:
            return np.inf
        y0 = lam_y.size // 2
        v = lam_y[y0]
        if np.isfinite(v):
            return float(v)
        if np.any(np.isfinite(lam_y)):
            idx = np.where(np.isfinite(lam_y))[0]
            j = idx[np.argmin(np.abs(idx - y0))]
            return float(lam_y[j])
        return np.inf

    keys = [target_lam_key(ly) for ly in lam_y_list]
    order = sorted(range(len(resolved)), key=lambda i: (keys[i], resolved[i].name))
    resolved = [resolved[i] for i in order]
    lam_y_list = [lam_y_list[i] for i in order]
    cutouts = [cutouts[i] for i in order]
    cutout_wcs_list = [cutout_wcs_list[i] for i in order]

    all_lam = np.concatenate([ly[np.isfinite(ly)] for ly in lam_y_list if np.any(np.isfinite(ly))])
    if all_lam.size == 0:
        global_lam_min, global_lam_max = 0.0, 1.0
    else:
        global_lam_min = float(np.nanmin(all_lam))
        global_lam_max = float(np.nanmax(all_lam))
        if global_lam_min == global_lam_max:
            global_lam_max = global_lam_min + 1e-6
    print(f"[INFO] global lambda(y) range: {global_lam_min:.4f}-{global_lam_max:.4f} um")

    from reproject import reproject_interp

    cutouts_reproj = []
    footprints = []
    for data, wcs in zip(cutouts, cutout_wcs_list):
        reproj, footprint = reproject_interp((data, wcs), ref_wcs, shape_out=ref_shape)
        reproj = np.where(footprint > 0, reproj, np.nan)
        cutouts_reproj.append(reproj)
        footprints.append(footprint)
    cutouts = cutouts_reproj

    vmin, vmax = animate_mod.robust_limits(cutouts, p_lo=args.p_lo, p_hi=args.p_hi)
    print(f"[INFO] Robust global vmin/vmax: {vmin:.6g}, {vmax:.6g}")
    print(f"[INFO] lambda range from WCS-WAVE: {lam_min:.4f}-{lam_max:.4f} um")

    out_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = []
    for i, (im2d, foot, lam_y) in enumerate(zip(cutouts, footprints, lam_y_list)):
        out_png = out_dir / f"frame_{i:04d}.png"
        animate_mod.render_frame_lambda_by_row(
            im2d,
            out_png=out_png,
            vmin=vmin,
            vmax=vmax,
            lam_y=lam_y,
            lam_min=global_lam_min,
            lam_max=global_lam_max,
            cover_mask=foot,
            cmap_name="rainbow",
            context_gray=True,
            show_lambda_colorbar=True,
            lambda_colorbar_style="top",
            grayscale=args.grayscale,
            invert=args.invert,
        )
        frame_paths.append(out_png)

    out_path = Path(args.out).expanduser().resolve()
    animate_mod.write_animation(frame_paths, out_path, fps=args.fps, sec_per_frame=args.sec_per_frame, mp4=args.mp4)
    print(f"[INFO] PNG frames in: {out_dir}")


# ---------------------------------------------------------------------------
# spherex-get: one-command pipeline for a basic user
# ---------------------------------------------------------------------------
def get() -> None:
    ap = argparse.ArgumentParser(
        prog="spherex-get",
        description="One-command pipeline: resolve a target (by name or RA/Dec) plus a field "
        "size, then query IRSA, download cutouts, and build a spectral cube -- with no "
        "manifest/ROI file names to juggle yourself. For full control over each step "
        "(rectangular boxes, custom manifest/output names, tiling large regions, ...), use "
        "spherex-query / spherex-download / spherex-cube directly instead.",
    )
    ap.add_argument(
        "--target", default=None, help="Object name or identifier, resolved via SIMBAD (e.g. 'M31', 'NGC 224')."
    )
    ap.add_argument("--ra", type=float, default=None, help="Right ascension, degrees. Use instead of --target.")
    ap.add_argument("--dec", type=float, default=None, help="Declination, degrees. Use instead of --target.")
    ap.add_argument("--size_deg", type=float, required=True, help="Square field of view, degrees.")
    ap.add_argument(
        "--out", default=None, help="Output directory (default: named after --target or the coordinates)."
    )
    ap.add_argument("--n_bins", type=int, default=40, help="Wavelength bins in the cube.")
    ap.add_argument(
        "--dlambda",
        type=float,
        default=0.1,
        help="Half-width (um) of each bin's synthetic band. The default (40 bins, 0.1 um) maps "
        "the cube smoothly for a typical field.",
    )
    ap.add_argument("--pixscale", type=float, default=6.15, help='Output pixel scale, arcsec/px (default ~6.15"/px).')
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel download workers. Default is 1 (gentle, predictable); raise it (e.g. 6) "
        "for faster downloads once you're comfortable with the pipeline.",
    )
    ap.add_argument("--no-match-background", action="store_true")
    ap.add_argument("--min_overlap_px", type=int, default=30)
    _add_cleaning_args(ap)
    _add_weighting_arg(ap)
    _add_grayscale_arg(ap)
    ap.add_argument("--animate", action="store_true", help="Also render a GIF sweeping the cube's wavelength bins.")
    ap.add_argument("--fps", type=float, default=6.0)
    ap.add_argument("--sec_per_frame", type=float, default=0.4)
    ap.add_argument(
        "--min_coverage", type=float, default=0.0, help="[--animate] Skip frames with less coverage than this (0-1)."
    )
    ap.add_argument(
        "--rgb",
        nargs=3,
        type=float,
        metavar=("LAMBDA_R", "LAMBDA_G", "LAMBDA_B"),
        default=None,
        help="Also render an RGB composite from these three wavelengths (um), e.g. --rgb 4.5 3.0 1.2.",
    )
    ap.add_argument(
        "--mosaic",
        type=float,
        default=None,
        metavar="LAMBDA_UM",
        help="Also render a single-wavelength mosaic at this wavelength (um) -- e.g. --mosaic 3.3 for "
        "the 3.3 um PAH (polycyclic aromatic hydrocarbon) emission feature. Uses --dlambda as its "
        "band half-width, same as the cube. For just this one image with no cube at all (faster, "
        "fewer exposures to download), use spherex-query --lambda_um/spherex-download/spherex-mosaic "
        "directly instead.",
    )
    args = ap.parse_args()

    if args.target is not None and (args.ra is not None or args.dec is not None):
        raise SystemExit("[ERROR] Give --target or --ra/--dec, not both.")
    if args.target is None and (args.ra is None or args.dec is None):
        raise SystemExit("[ERROR] Give --target, or both --ra and --dec.")

    if args.target is not None:
        from spherexstack.target import resolve_target

        try:
            ra, dec, main_id = resolve_target(args.target)
        except (ValueError, RuntimeError) as exc:
            raise SystemExit(f"[ERROR] {exc}") from exc
        print(f"[INFO] Resolved '{args.target}' -> {main_id}  RA={ra:.5f}  Dec={dec:.5f}")
    else:
        ra, dec = args.ra, args.dec

    out_dir = (
        Path(args.out).expanduser().resolve() if args.out else Path(_default_out_dir(args.target, ra, dec)).resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directory: {out_dir}")

    bad_bits = _bad_bits_from_arg(args.bad_bits)
    roi_box = roi_mod.resolve_roi(ra_center=ra, dec_center=dec, size_deg=args.size_deg)
    roi_path = out_dir / "roi.json"
    roi_mod.save_roi(roi_path, roi_box)

    print("[INFO] Step 1/3: querying IRSA for exposures...")
    csv_text = query_mod.run_tap_query(roi_box["ra_min"], roi_box["ra_max"], roi_box["dec_min"], roi_box["dec_max"])
    rows = list(csv.DictReader(csv_text.splitlines()))
    if not rows:
        raise SystemExit(
            "[ERROR] No exposures found for this field. SPHEREx's sky coverage is still patchy -- "
            "try a larger --size_deg, or a different target."
        )
    manifest_path = out_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] {len(rows)} exposures found.")

    print(f"[INFO] Step 2/3: downloading cutouts ({args.workers} worker(s), resumable)...")
    cutouts_dir = out_dir / "cutouts"
    cutouts_dir.mkdir(exist_ok=True)
    keep_hdus = download_mod.KEEP_HDUS_DEFAULT
    if args.weighting == "ivar" and "VARIANCE" not in keep_hdus:
        keep_hdus = (*keep_hdus, "VARIANCE")
        print("[INFO] --weighting ivar: also keeping the VARIANCE HDU.")
    cutout_size_deg = max(roi_box["width_deg"], roi_box["height_deg"])

    ok, failed = 0, 0
    bar = ProgressBar(total=len(rows), prefix="Downloading", unit="files")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [
            ex.submit(
                download_mod.download_one,
                row,
                roi_box["ra_center"],
                roi_box["dec_center"],
                cutout_size_deg,
                cutouts_dir,
                keep_hdus,
            )
            for row in rows
        ]
        for fut in as_completed(futures):
            out_path, status, _nbytes = fut.result()
            if status in ("ok", "skip"):
                ok += 1
            else:
                failed += 1
                bar.interrupt(f"[WARN] {out_path.name}: {status}")
            bar.update()
    bar.close()
    msg = f"[INFO] {ok}/{len(rows)} cutouts ready"
    print(msg + (f", {failed} failed -- rerun spherex-get with the same --out to retry." if failed else "."))

    all_files = sorted(str(p) for p in cutouts_dir.glob("*.fits"))
    if not all_files:
        raise SystemExit("[ERROR] No cutouts downloaded successfully.")
    list_path = out_dir / "list.txt"
    with open(list_path, "w") as f:
        f.write("\n".join(all_files) + "\n")

    print("[INFO] Step 3/3: building the spectral cube...")
    resolved = io.resolve_list_paths(list_path)
    ref_wcs, ref_shape = mosaic_mod.build_reference_wcs(
        roi_box["ra_center"], roi_box["dec_center"], roi_box["width_deg"], roi_box["height_deg"], args.pixscale
    )
    lam_ranges = [io.quick_lambda_range(fp) for fp in resolved]
    lam_ranges = [r for r in lam_ranges if r is not None]
    if not lam_ranges:
        raise SystemExit("[ERROR] No downloaded FITS file has a readable WCS-WAVE.")
    global_lam_min = min(r[0] for r in lam_ranges)
    global_lam_max = max(r[1] for r in lam_ranges)
    bin_centers = np.linspace(global_lam_min, global_lam_max, args.n_bins)
    print(f"[INFO] Wavelength range: {global_lam_min:.4f}-{global_lam_max:.4f} um, {args.n_bins} bins")

    cube_arr, weight_cube, coverage = cube_mod.build_cube(
        resolved,
        bad_bits,
        clean=not args.no_clean,
        ref_wcs=ref_wcs,
        ref_shape=ref_shape,
        bin_centers=bin_centers,
        dlambda=args.dlambda,
        match_background=not args.no_match_background,
        min_overlap_px=args.min_overlap_px,
        weighting=args.weighting,
    )
    cube_path = out_dir / "cube.fits"
    cube_mod.save_cube_fits(
        cube_path,
        cube_arr,
        weight_cube,
        ref_wcs,
        bin_centers,
        dlambda=args.dlambda,
        coverage=coverage,
        extra_header={
            "RAMIN": roi_box["ra_min"],
            "RAMAX": roi_box["ra_max"],
            "DECMIN": roi_box["dec_min"],
            "DECMAX": roi_box["dec_max"],
        },
    )
    n_empty = int(np.sum(coverage == 0))
    if n_empty:
        print(f"[WARN] {n_empty}/{len(coverage)} wavelength bins have zero coverage.")
    print(f"[DONE] Cube: {cube_path}")

    if args.animate:
        from spherexstack import animate

        frames_dir = out_dir / "_frames"
        frames_dir.mkdir(exist_ok=True)
        vmin, vmax = animate.robust_limits(list(cube_arr), p_lo=1.0, p_hi=99.0)
        fov_deg = animate.fov_deg_from_wcs(ref_wcs, ref_shape)
        frame_paths = []
        for i, (lam0, mosaic_slice, cov) in enumerate(zip(bin_centers, cube_arr, coverage)):
            if cov < args.min_coverage:
                continue
            out_png = frames_dir / f"lslice_{i:04d}.png"
            animate.render_lambda_slice(
                mosaic_slice,
                out_png,
                vmin=vmin,
                vmax=vmax,
                lam_center=float(lam0),
                dlambda=args.dlambda,
                coverage_frac=cov,
                fov_deg=fov_deg,
                grayscale=args.grayscale,
                invert=args.invert,
            )
            frame_paths.append(out_png)
        if frame_paths:
            gif_path = out_dir / "slices.gif"
            animate.write_animation(frame_paths, gif_path, fps=args.fps, sec_per_frame=args.sec_per_frame)
            print(f"[DONE] Animation: {gif_path}")
        else:
            print("[WARN] No frame passed --min_coverage; skipping the animation.")

    if args.rgb:
        from spherexstack import animate

        lam_r, lam_g, lam_b = args.rgb
        print(f"[INFO] Building RGB composite (R={lam_r} G={lam_g} B={lam_b} um)...")
        rgb_mosaics, rgb_coverages, _weights = mosaic_mod.build_lambda_mosaics_streaming(
            resolved,
            bad_bits,
            clean=not args.no_clean,
            ref_wcs=ref_wcs,
            ref_shape=ref_shape,
            bin_centers=np.array([lam_r, lam_g, lam_b]),
            dlambda=args.dlambda,
            match_background=not args.no_match_background,
            min_overlap_px=args.min_overlap_px,
            weighting=args.weighting,
        )
        for channel_name, lam, cov in zip(("R", "G", "B"), (lam_r, lam_g, lam_b), rgb_coverages):
            if cov == 0.0:
                print(f"[WARN] Zero coverage for the {channel_name} channel ({lam} um) -- it will render fully black.")
        rgb_path = out_dir / "rgb.png"
        animate.render_rgb_composite(
            rgb_mosaics[0],
            rgb_mosaics[1],
            rgb_mosaics[2],
            rgb_path,
            wavelengths_um=(lam_r, lam_g, lam_b),
            fov_deg=animate.fov_deg_from_wcs(ref_wcs, ref_shape),
        )
        print(f"[DONE] RGB composite: {rgb_path}")

    if args.mosaic is not None:
        from spherexstack import animate

        lam = args.mosaic
        print(f"[INFO] Building single-wavelength mosaic at lambda={lam} um...")
        single_mosaics, single_coverages, _weights = mosaic_mod.build_lambda_mosaics_streaming(
            resolved,
            bad_bits,
            clean=not args.no_clean,
            ref_wcs=ref_wcs,
            ref_shape=ref_shape,
            bin_centers=np.array([lam]),
            dlambda=args.dlambda,
            match_background=not args.no_match_background,
            min_overlap_px=args.min_overlap_px,
            weighting=args.weighting,
        )
        mosaic_slice, cov = single_mosaics[0], single_coverages[0]
        if cov == 0.0:
            print(
                f"[WARN] Zero coverage at lambda={lam} um -- no exposure contributed a single pixel. "
                "Try a wider --dlambda or a bigger --size_deg."
            )
        else:
            vmin, vmax = animate.robust_limits([mosaic_slice], p_lo=1.0, p_hi=99.0)
            mosaic_path = out_dir / f"mosaic_{lam:.2f}um.png"
            animate.render_lambda_slice(
                mosaic_slice,
                mosaic_path,
                vmin=vmin,
                vmax=vmax,
                lam_center=lam,
                dlambda=args.dlambda,
                coverage_frac=cov,
                fov_deg=animate.fov_deg_from_wcs(ref_wcs, ref_shape),
                grayscale=args.grayscale,
                invert=args.invert,
            )
            print(f"[DONE] Single-wavelength mosaic: {mosaic_path} (coverage {100 * cov:.1f}%)")

    print(f"[INFO] For other commands, reuse: --list {list_path} --roi {roi_path}")
