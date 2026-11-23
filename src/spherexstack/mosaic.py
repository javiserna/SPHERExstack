"""Reprojection, background matching, streaming accumulation, and mosaic merging.

This module builds monochromatic 2D mosaics: for each target wavelength, pixels
from many exposures whose local wavelength (from ``WCS-WAVE``) falls within a
tolerance of the target are reprojected onto a common sky grid and averaged.
This works because SPHEREx revisits the same field with different pointings
(dithers), so a given sky pixel lands on a different detector row -- and
therefore a different wavelength -- on each visit; combining many visits can
fill in wavelength coverage that no single exposure has on its own.

Two accumulation strategies are provided:

* :func:`build_lambda_mosaics_streaming` -- processes one exposure at a time and
  only touches the sub-window of the reference grid that exposure overlaps,
  accumulating ``(sum, weight)`` per wavelength bin. Memory scales with
  ``n_bins x grid size``, not with the number of exposures, which is what lets
  it scale to large regions. Background matching here is incremental: each new
  exposure is compared only against the average accumulated so far in the
  overlap region.
* :func:`build_lambda_mosaics_legacy` -- loads every exposure into memory at
  once and solves a single joint background correction per wavelength bin
  (:func:`background_corrections_for_bin`) seeing all exposures simultaneously.
  Used for the point-target mode, where the output grid is small.

No plotting dependency is required to build a mosaic or a cube; rendering
lives in :mod:`spherexstack.animate`.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
from astropy.wcs import WCS
from reproject import reproject_interp
from reproject.mosaicking.background import solve_corrections_sgd

from spherexstack import io

WEIGHTING_CHOICES = ("mean", "ivar")


def build_reference_wcs(
    ra_center: float, dec_center: float, width_deg: float, height_deg: float,
    pixscale_arcsec: float,
) -> tuple[WCS, tuple[int, int]]:
    """Reference WCS for ROI mode: a TAN grid centered on ``(ra_center, dec_center)``.

    All exposures (already cropped to the ROI by ``spherex-download``) are
    reprojected onto this one common grid, instead of onto a cutout WCS
    around a point target.

    Returns
    -------
    wcs : astropy.wcs.WCS
    shape : (ny, nx)
    """
    nx = int(round((width_deg * 3600.0) / pixscale_arcsec))
    ny = int(round((height_deg * 3600.0) / pixscale_arcsec))
    w = WCS(naxis=2)
    w.wcs.crpix = [nx / 2.0 + 0.5, ny / 2.0 + 0.5]
    w.wcs.cdelt = [-pixscale_arcsec / 3600.0, pixscale_arcsec / 3600.0]
    w.wcs.crval = [ra_center, dec_center]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w, (ny, nx)


def _ref_pixel_bbox(
    wcs_native: WCS, shape: tuple[int, int], ref_wcs: WCS, ref_shape: tuple[int, int], pad: int = 2
) -> tuple[int, int, int, int] | None:
    """Pixel box ``(y0, y1, x0, x1)`` in ``ref_wcs`` covering a native image's footprint.

    Clipped to ``ref_shape``. Returns None if there is no overlap.
    """
    ny_n, nx_n = shape
    corners_x = [0, nx_n - 1, 0, nx_n - 1]
    corners_y = [0, 0, ny_n - 1, ny_n - 1]
    sky = wcs_native.pixel_to_world(corners_x, corners_y)
    xs, ys = ref_wcs.world_to_pixel(sky)
    if not np.all(np.isfinite(xs)) or not np.all(np.isfinite(ys)):
        return None
    x0 = int(np.floor(np.min(xs))) - pad
    x1 = int(np.ceil(np.max(xs))) + 1 + pad
    y0 = int(np.floor(np.min(ys))) - pad
    y1 = int(np.ceil(np.max(ys))) + 1 + pad
    ny_ref, nx_ref = ref_shape
    x0c, x1c = max(0, x0), min(nx_ref, x1)
    y0c, y1c = max(0, y0), min(ny_ref, y1)
    if x0c >= x1c or y0c >= y1c:
        return None
    return y0c, y1c, x0c, x1c


def _shifted_wcs(ref_wcs: WCS, x0: int, y0: int) -> WCS:
    """Copy of ``ref_wcs`` whose pixel (0, 0) corresponds to ``(x0, y0)`` in ``ref_wcs``."""
    local = ref_wcs.deepcopy()
    local.wcs.crpix = [ref_wcs.wcs.crpix[0] - x0, ref_wcs.wcs.crpix[1] - y0]
    return local


def background_corrections_for_bin(
    data_masked: np.ndarray, min_overlap_px: int = 30
) -> np.ndarray:
    """Per-exposure additive background offset that removes seams between exposures.

    Uses the same technique as ``reproject.mosaicking.reproject_and_coadd(...,
    match_background=True)``: for each pair of exposures that overlap (same
    valid pixels on the common grid, for this wavelength bin), the median
    difference in the overlap region is taken; the set of per-exposure offsets
    that best explains that difference matrix is then solved by stochastic
    gradient descent (:func:`reproject.mosaicking.background.solve_corrections_sgd`).
    Exposures with no overlap with any other get a correction of 0 (nothing
    to align them against).

    Parameters
    ----------
    data_masked : ndarray, shape (N, ny, nx)
        NaN outside this bin's wavelength band.

    Returns
    -------
    ndarray, shape (N,)
        Per-exposure correction (0 for exposures with no data in this bin).
    """
    n = data_masked.shape[0]
    valid = np.isfinite(data_masked)
    has_data = valid.reshape(n, -1).any(axis=1)
    active_idx = np.where(has_data)[0]

    corrections = np.zeros(n)
    if active_idx.size <= 1:
        return corrections

    k = active_idx.size
    offset_matrix = np.full((k, k), np.nan)
    for a in range(k):
        i = active_idx[a]
        for b in range(a + 1, k):
            j = active_idx[b]
            both = valid[i] & valid[j]
            n_overlap = int(both.sum())
            if n_overlap >= min_overlap_px:
                diff = data_masked[i][both] - data_masked[j][both]
                med = float(np.median(diff))
                offset_matrix[a, b] = med
                offset_matrix[b, a] = -med

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        active_corrections = solve_corrections_sgd(offset_matrix)
    active_corrections = np.nan_to_num(active_corrections, nan=0.0)

    corrections[active_idx] = active_corrections
    return corrections


def _pixel_weight(
    valid: np.ndarray,
    reproj_var: np.ndarray | None,
    weighting: str,
    shape: tuple[int, int],
) -> np.ndarray:
    """Per-pixel weight array for one exposure's contribution to one bin."""
    if weighting == "ivar" and reproj_var is not None:
        pix_weight = np.zeros(shape, dtype=np.float32)
        good_var = valid & np.isfinite(reproj_var) & (reproj_var > 0)
        pix_weight[good_var] = (1.0 / reproj_var[good_var]).astype(np.float32)
        pix_weight[valid & ~good_var] = 1.0
        return pix_weight
    return np.where(valid, 1.0, 0.0).astype(np.float32)


def build_lambda_mosaics_streaming(
    resolved: list[Path],
    bad_bits: list[str],
    clean: bool,
    ref_wcs: WCS,
    ref_shape: tuple[int, int],
    bin_centers: np.ndarray,
    dlambda: float | np.ndarray,
    match_background: bool,
    min_overlap_px: int,
    weighting: str = "mean",
    bin_sources: list[set[Path]] | None = None,
) -> tuple[list[np.ndarray], list[float], list[np.ndarray]]:
    """Streaming ROI-mode mosaic builder: scalable in memory with the grid, not with exposure count.

    Processes one exposure at a time and reprojects only the sub-region of the
    reference grid that exposure actually touches, accumulating
    ``(weighted sum, total weight)`` per wavelength bin. RAM use depends on
    ``n_bins x grid_size``, not on ``n_exposures x grid_size`` -- this is what
    allows scaling to large regions (thousands of exposures, grids of several
    thousand pixels per side) without holding everything in memory at once,
    unlike :func:`build_lambda_mosaics_legacy`.

    ``dlambda`` is either one half-width shared by every bin, or an array with
    one half-width per ``bin_centers`` entry (e.g. a feature's narrower line
    band next to its wider continuum bands -- see
    :func:`spherex_ice.tau.build_epoch_tau_map`).

    ``bin_sources``, if given, is one set of allowed exposure paths per bin
    (same length as ``bin_centers``): a file not in ``bin_sources[b]`` is
    skipped for bin ``b`` specifically, even though it's still read once and
    may still contribute to other bins. This is what lets several bins that
    each need their own exposure exclusions (e.g. independent outlier
    rejection per band in :mod:`spherex_ice.tau`) share one pass over
    ``resolved`` (the union of every bin's own files) -- each file is read,
    cleaned and wavelength-mapped once no matter how many bins it touches --
    instead of every bin re-reading/re-cleaning every file from scratch in its
    own separate call. Default (None): every file in ``resolved`` contributes
    to every bin, exactly as before this parameter existed.

    Returns
    -------
    mosaics : list of ndarray
        One averaged 2D mosaic per bin (NaN where there is no coverage).
    coverages : list of float
        Fraction of finite pixels per mosaic.
    weights : list of ndarray
        Per-pixel total weight per bin (exposure count, under ``weighting="mean"``).
    """
    if weighting not in WEIGHTING_CHOICES:
        raise ValueError(f"Unknown weighting '{weighting}', expected one of {WEIGHTING_CHOICES}.")
    need_variance = weighting == "ivar"

    n_bins = len(bin_centers)
    dlambda_per_bin = np.broadcast_to(np.asarray(dlambda, dtype=float), (n_bins,))
    if bin_sources is not None and len(bin_sources) != n_bins:
        raise ValueError(f"bin_sources has {len(bin_sources)} entries, expected {n_bins} (one per bin_centers).")
    ny, nx = ref_shape
    sum_acc = np.zeros((n_bins, ny, nx), dtype=np.float32)
    weight_acc = np.zeros((n_bins, ny, nx), dtype=np.float32)
    exps_per_bin = np.zeros(n_bins, dtype=int)

    n_used = 0
    ivar_fallback_warned = False
    for i, fp in enumerate(resolved):
        result = io.load_roi_frame(fp, bad_bits, clean, need_variance=need_variance)
        if result is None:
            print(f"[WARN] No valid data in {fp.name}, skipping.")
            continue
        data, wcs_native, lam, variance = result

        if need_variance and variance is None and not ivar_fallback_warned:
            print(
                f"[WARN] {fp.name} has no VARIANCE HDU (re-download with "
                "--keep-hdus including VARIANCE); exposures without it fall "
                "back to weight=1 for ivar weighting."
            )
            ivar_fallback_warned = True

        bbox = _ref_pixel_bbox(wcs_native, data.shape, ref_wcs, ref_shape)
        if bbox is None:
            continue
        y0, y1, x0, x1 = bbox
        local_wcs = _shifted_wcs(ref_wcs, x0, y0)
        local_shape = (y1 - y0, x1 - x0)

        lam_fin = lam[np.isfinite(lam)]
        if lam_fin.size == 0:
            continue
        lam_lo, lam_hi = float(lam_fin.min()), float(lam_fin.max())
        n_used += 1

        for b, lam0 in enumerate(bin_centers):
            if bin_sources is not None and fp not in bin_sources[b]:
                continue
            db = dlambda_per_bin[b]
            if lam0 + db < lam_lo or lam0 - db > lam_hi:
                continue
            mask = np.abs(lam - lam0) <= db
            if not mask.any():
                continue
            data_masked = np.where(mask, data, np.nan)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=AstropyWarning)
                reproj, footprint = reproject_interp(
                    (data_masked, wcs_native), local_wcs, shape_out=local_shape
                )
            valid = (footprint > 0) & np.isfinite(reproj)
            if not valid.any():
                continue

            reproj_var = None
            if need_variance and variance is not None:
                var_masked = np.where(mask, variance, np.nan)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=AstropyWarning)
                    reproj_var, _ = reproject_interp(
                        (var_masked, wcs_native), local_wcs, shape_out=local_shape
                    )
            pix_weight = _pixel_weight(valid, reproj_var, weighting, local_shape)

            if match_background:
                cur_weight = weight_acc[b, y0:y1, x0:x1]
                cur_sum = sum_acc[b, y0:y1, x0:x1]
                overlap = valid & (cur_weight > 0)
                if int(overlap.sum()) >= min_overlap_px:
                    cur_mean = cur_sum[overlap] / cur_weight[overlap]
                    offset = float(np.median(reproj[overlap] - cur_mean))
                    reproj = reproj - offset

            reproj_filled = np.where(valid, reproj, 0.0).astype(np.float32)
            sum_acc[b, y0:y1, x0:x1] += reproj_filled * pix_weight
            weight_acc[b, y0:y1, x0:x1] += pix_weight
            exps_per_bin[b] += 1

        if (i + 1) % 20 == 0 or (i + 1) == len(resolved):
            print(f"[INFO]   {i + 1}/{len(resolved)} exposures processed...")

    print(f"[INFO] {n_used}/{len(resolved)} exposures overlap the reference grid.")

    mosaics, coverages, weights = [], [], []
    for b in range(n_bins):
        with np.errstate(invalid="ignore", divide="ignore"):
            mosaic = sum_acc[b] / weight_acc[b]
        mosaic = np.where(weight_acc[b] > 0, mosaic, np.nan).astype(np.float64)
        coverage = float(np.isfinite(mosaic).sum()) / mosaic.size
        mosaics.append(mosaic)
        coverages.append(coverage)
        weights.append(weight_acc[b].astype(np.float32))
        print(
            f"[INFO] lambda={bin_centers[b]:.4f} um  coverage={100 * coverage:.1f}%  "
            f"n_exposures_used={int(exps_per_bin[b])}"
        )

    return mosaics, coverages, weights


def build_lambda_mosaics_legacy(
    resolved: list[Path],
    bad_bits: list[str],
    clean: bool,
    target: Any,
    cutout_size: int,
    bin_centers: np.ndarray | None,
    dlambda: float,
    match_background: bool,
    min_overlap_px: int,
    weighting: str = "mean",
    n_bins: int | None = None,
) -> tuple[
    list[np.ndarray], list[float], list[np.ndarray], WCS, tuple[int, int], np.ndarray, np.ndarray
]:
    """Point-target mode: load every exposure into memory and jointly solve background offsets.

    All exposures are cropped to a ``cutout_size`` x ``cutout_size`` window
    around ``target`` and reprojected onto a common WCS taken from the first
    valid cutout; for each wavelength bin, a joint per-exposure background
    correction is solved from all exposures at once
    (:func:`background_corrections_for_bin`), then averaged with
    ``numpy.nanmean``. Memory scales with ``n_exposures x grid_size``; use
    :func:`build_lambda_mosaics_streaming` for large ROIs.

    Only ``weighting="mean"`` is supported in this mode; ``"ivar"`` falls back
    to ``"mean"`` with a warning (point-target cutouts are small and this mode
    is meant for quick single-target looks, not the primary weighted pipeline).

    If ``bin_centers`` is None, it is computed as ``n_bins`` evenly spaced
    values spanning the wavelength range actually observed across the loaded
    exposures (matching the original script's behavior, where the range is
    only known once the data has been read).

    Returns
    -------
    mosaics : list of ndarray
    coverages : list of float
    weights : list of ndarray
        Per-pixel exposure count contributing to each bin (diagnostic; the
        mosaic itself is an unweighted ``nanmean``, not a weighted sum).
    ref_wcs : astropy.wcs.WCS
    ref_shape : (ny, nx)
    bin_centers : ndarray, shape (n_bins,)
    all_lam : ndarray, shape (N, ny, nx)
        Per-pixel wavelength map for every loaded exposure (for diagnostics).
    """
    if weighting == "ivar":
        print("[WARN] weighting='ivar' is not supported in point-target mode; using 'mean'.")

    print(f"[INFO] Reading, cleaning and cropping {len(resolved)} FITS files...")
    ref_wcs = None
    ref_shape = None
    all_data = []
    all_lam = []

    for i, fp in enumerate(resolved):
        result = io.load_target_cutout_and_lambda(fp, target, cutout_size, bad_bits, clean=clean)
        if result is None:
            print(f"[WARN] No overlap/valid data in {fp.name}, skipping.")
            continue
        data, wcs_cut, lam = result

        if ref_wcs is None:
            ref_wcs = wcs_cut
            ref_shape = data.shape

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=AstropyWarning)
            reproj_data, footprint = reproject_interp((data, wcs_cut), ref_wcs, shape_out=ref_shape)
            reproj_lam, footprint_lam = reproject_interp((lam, wcs_cut), ref_wcs, shape_out=ref_shape)

        reproj_data = np.where(footprint > 0, reproj_data, np.nan)
        reproj_lam = np.where(footprint_lam > 0, reproj_lam, np.nan)

        all_data.append(reproj_data)
        all_lam.append(reproj_lam)

        if (i + 1) % 20 == 0:
            print(f"[INFO]   {i + 1}/{len(resolved)} processed...")

    if not all_data:
        raise ValueError("No FITS file overlapped the requested target/cutout.")

    all_data_arr = np.array(all_data)  # (N, ny, nx)
    all_lam_arr = np.array(all_lam)

    if bin_centers is None:
        if n_bins is None:
            raise ValueError("Give either bin_centers or n_bins.")
        lam_finite = all_lam_arr[np.isfinite(all_lam_arr)]
        global_lam_min = float(np.nanmin(lam_finite))
        global_lam_max = float(np.nanmax(lam_finite))
        print(
            f"[INFO] {all_data_arr.shape[0]} valid exposures. "
            f"Global observed lambda range: {global_lam_min:.4f}-{global_lam_max:.4f} um"
        )
        bin_centers = np.linspace(global_lam_min, global_lam_max, n_bins)

    mosaics = []
    coverages = []
    weights = []
    for lam0 in bin_centers:
        match = np.abs(all_lam_arr - lam0) <= dlambda
        data_masked = np.where(match, all_data_arr, np.nan)

        if match_background:
            corrections = background_corrections_for_bin(data_masked, min_overlap_px=min_overlap_px)
            if np.any(corrections != 0):
                data_masked = data_masked - corrections[:, None, None]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mosaic = np.nanmean(data_masked, axis=0)
        coverage = float(np.isfinite(mosaic).sum()) / mosaic.size
        mosaics.append(mosaic)
        coverages.append(coverage)
        weights.append(np.isfinite(data_masked).sum(axis=0).astype(np.float32))
        print(
            f"[INFO] lambda={lam0:.4f} um  coverage={100 * coverage:.1f}%  "
            f"n_exposures_used={int(match.any(axis=(1, 2)).sum())}"
        )

    return mosaics, coverages, weights, ref_wcs, ref_shape, bin_centers, all_lam_arr


def save_mosaic_fits(
    path: Path,
    mosaic: np.ndarray,
    weight: np.ndarray,
    ref_wcs: WCS,
    lambda_um: float,
    dlambda: float,
    extra_header: dict[str, Any] | None = None,
    bunit: str = "MJy/sr",
) -> None:
    """Write a streaming mosaic (+ its per-pixel weight map) to FITS.

    Intended for *partial* mosaics (one sub-region) that are later combined
    with :func:`build_merged_mosaic` into a larger final mosaic, without
    needing the original exposures (which may already have been deleted from
    disk).
    """
    hdr = ref_wcs.to_header()
    hdr["LAMBDA_UM"] = (float(lambda_um), "Target wavelength (microns)")
    hdr["DLAMBDA_UM"] = (float(dlambda), "Half-width of the synthetic band (microns)")
    if extra_header:
        for k, v in extra_header.items():
            hdr[k] = v

    mosaic_hdr = hdr.copy()
    mosaic_hdr["BUNIT"] = (bunit, "Surface brightness (native) or flux density (--units mjy)")
    weight_hdr = hdr.copy()
    weight_hdr["BUNIT"] = ("", "Total weight per pixel (exposure count under weighting=mean)")

    mosaic_hdu = fits.ImageHDU(data=mosaic.astype(np.float32), header=mosaic_hdr, name="MOSAIC")
    weight_hdu = fits.ImageHDU(data=weight.astype(np.float32), header=weight_hdr, name="WEIGHT")
    fits.HDUList([fits.PrimaryHDU(), mosaic_hdu, weight_hdu]).writeto(path, overwrite=True)


def load_mosaic_fits(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, WCS, dict[str, float | None]]:
    """Inverse of :func:`save_mosaic_fits`: returns ``(mosaic, weight, wcs, meta)``."""
    with fits.open(path, memmap=False) as hdul:
        mosaic = np.asarray(hdul["MOSAIC"].data, dtype=float)
        weight = np.asarray(hdul["WEIGHT"].data, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=AstropyWarning)
            wcs = WCS(hdul["MOSAIC"].header)
        meta = {
            "lambda_um": hdul["MOSAIC"].header.get("LAMBDA_UM"),
            "dlambda_um": hdul["MOSAIC"].header.get("DLAMBDA_UM"),
        }
    mosaic = np.where(weight > 0, mosaic, np.nan)
    return mosaic, weight, wcs, meta


def build_merged_mosaic(
    tile_paths: list[Path],
    ref_wcs: WCS,
    ref_shape: tuple[int, int],
    match_background: bool = True,
    min_overlap_px: int = 200,
) -> tuple[np.ndarray, np.ndarray, float | None, float | None, float]:
    """Combine several partial mosaic FITS files onto one common grid.

    Each input FITS carries ``MOSAIC`` (an already-averaged value) and
    ``WEIGHT`` (how much weight contributed to each pixel). Each tile is
    reprojected onto the final grid and accumulated as a weighted average:
    ``final = sum(mosaic_i * weight_i) / sum(weight_i)`` -- the same operation
    :func:`build_lambda_mosaics_streaming` performs over individual exposures,
    just with a whole partial mosaic as the unit being combined.

    Returns
    -------
    final_mosaic, weight_acc, lambda_um, dlambda_um, coverage
    """
    ny, nx = ref_shape
    sum_acc = np.zeros(ref_shape, dtype=np.float32)
    weight_acc = np.zeros(ref_shape, dtype=np.float32)

    lambda_um = None
    dlambda_um = None

    for tile_path in tile_paths:
        mosaic, weight, tile_wcs, meta = load_mosaic_fits(tile_path)
        if lambda_um is None:
            lambda_um = meta.get("lambda_um")
            dlambda_um = meta.get("dlambda_um")
        elif meta.get("lambda_um") is not None and abs(meta["lambda_um"] - lambda_um) > 1e-6:
            print(
                f"[WARN] {tile_path.name} has LAMBDA_UM={meta['lambda_um']}, "
                f"different from the rest ({lambda_um}). Merging anyway; "
                "check that this is intentional."
            )

        bbox = _ref_pixel_bbox(tile_wcs, mosaic.shape, ref_wcs, ref_shape)
        if bbox is None:
            print(f"[WARN] {tile_path.name} does not overlap the final grid, skipping.")
            continue
        y0, y1, x0, x1 = bbox
        local_wcs = _shifted_wcs(ref_wcs, x0, y0)
        local_shape = (y1 - y0, x1 - x0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=AstropyWarning)
            reproj_mosaic, footprint = reproject_interp((mosaic, tile_wcs), local_wcs, shape_out=local_shape)
            reproj_weight, _ = reproject_interp((weight, tile_wcs), local_wcs, shape_out=local_shape)

        valid = (footprint > 0) & np.isfinite(reproj_mosaic) & np.isfinite(reproj_weight) & (reproj_weight > 0)
        n_valid = int(valid.sum())
        corr_info = ""
        if match_background:
            cur_weight = weight_acc[y0:y1, x0:x1]
            cur_sum = sum_acc[y0:y1, x0:x1]
            overlap = valid & (cur_weight > 0)
            if int(overlap.sum()) >= min_overlap_px:
                cur_mean = cur_sum[overlap] / cur_weight[overlap]
                offset = float(np.median(reproj_mosaic[overlap] - cur_mean))
                reproj_mosaic = reproj_mosaic - offset
                corr_info = f"  background_offset={offset:+.4g}"
        print(
            f"[INFO] {tile_path.name}: {n_valid} valid px on the final grid "
            f"(bbox {local_shape[1]}x{local_shape[0]} at x=[{x0},{x1}) y=[{y0},{y1})){corr_info}"
        )
        if n_valid == 0:
            continue

        sum_acc[y0:y1, x0:x1][valid] += (reproj_mosaic[valid] * reproj_weight[valid]).astype(np.float32)
        weight_acc[y0:y1, x0:x1][valid] += reproj_weight[valid].astype(np.float32)

    with np.errstate(invalid="ignore", divide="ignore"):
        final_mosaic = sum_acc / weight_acc
    final_mosaic = np.where(weight_acc > 0, final_mosaic, np.nan)
    coverage = float(np.isfinite(final_mosaic).sum()) / final_mosaic.size

    return final_mosaic, weight_acc, lambda_um, dlambda_um, coverage
