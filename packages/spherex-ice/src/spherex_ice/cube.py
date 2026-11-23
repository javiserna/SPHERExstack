"""A compact, per-epoch spectral data cube: the handful of narrow bands any of this
package's built-in features actually needs (blue continuum, line, red continuum --
see :data:`spherex_ice.tau.FEATURE_PRESETS`), already reprojected onto the shared
pixel grid and reduced through the same outlier-rejection/background-matching
pipeline as a single feature map, bundled into ONE small FITS file per epoch.

The point: the expensive part of this whole package is reading and reprojecting
potentially hundreds of raw per-exposure cutouts. Do that once (building whichever
``--feature``s you asked for, with ``--cube``), and the compact cube that comes out
of it is enough to re-derive any of those same features' maps afterward --
instantly, and without the raw cutouts on hand at all (see
:func:`tau_from_cube` and the ``spherex-ice-from-cube`` command).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


def bands_from_result(
    feature_name: str, feature: dict[str, Any], result: dict[str, Any]
) -> list[dict[str, Any]]:
    """The 3 cube bands (blue/line/red) already sitting inside a
    :func:`spherex_ice.tau.build_epoch_tau_map` result for ``feature_name`` -- free,
    since that function already built exactly these three mosaics for its own map.
    """
    dlambda_blue = 0.5 * (feature["cont_blue_hi"] - feature["cont_blue_lo"])
    dlambda_red = 0.5 * (feature["cont_red_hi"] - feature["cont_red_lo"])
    return [
        {
            "name": f"{feature_name}_blue", "feature": feature_name, "role": "blue",
            "lam_center": result["lam_blue"], "dlambda": dlambda_blue,
            "flux": result["f_blue"], "weight": result["w_blue"], "coverage": result["cov_blue"],
        },
        {
            "name": f"{feature_name}_line", "feature": feature_name, "role": "line",
            "lam_center": result["lam_line"], "dlambda": feature["dlambda_line"],
            "flux": result["f_line"], "weight": result["w_line"], "coverage": result["cov_line"],
        },
        {
            "name": f"{feature_name}_red", "feature": feature_name, "role": "red",
            "lam_center": result["lam_red"], "dlambda": dlambda_red,
            "flux": result["f_red"], "weight": result["w_red"], "coverage": result["cov_red"],
        },
    ]


def save_epoch_cube_fits(
    path: Path,
    bands: list[dict[str, Any]],
    ref_wcs: WCS,
    epoch_index: int,
    mjd_min: float,
    mjd_max: float,
) -> None:
    """Write ``bands`` (as built by :func:`bands_from_result`, across one or several
    features) to one compact FITS file: a ``FLUX`` and a ``WEIGHT`` cube (one slice per
    band), plus a ``BANDS`` table recording what each slice actually is.
    """
    if not bands:
        raise ValueError("No bands to write -- was --feature empty, or --cube given with nothing computed?")

    hdr = ref_wcs.to_header()
    hdr["SXEPOCH"] = (int(epoch_index), "Observation epoch index (spherex_ice)")
    hdr["MJD_MIN"] = (float(mjd_min), "First exposure MJD-OBS in this epoch")
    hdr["MJD_MAX"] = (float(mjd_max), "Last exposure MJD-OBS in this epoch")
    hdr["NBANDS"] = (len(bands), "Number of band slices -- see the BANDS table")
    hdr["COMMENT"] = "Axis 3 (band slices) is NOT a linear wavelength WCS axis -- see the BANDS table."

    flux_cube = np.stack([b["flux"] for b in bands], axis=0).astype(np.float32)
    weight_cube = np.stack([b["weight"] for b in bands], axis=0).astype(np.float32)

    bands_hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="NAME", format="24A", array=[b["name"] for b in bands]),
            fits.Column(name="FEATURE", format="16A", array=[b["feature"] for b in bands]),
            fits.Column(name="ROLE", format="8A", array=[b["role"] for b in bands]),
            fits.Column(name="LAM_UM", format="E", array=[b["lam_center"] for b in bands]),
            fits.Column(name="DLAMBDA_UM", format="E", array=[b["dlambda"] for b in bands]),
            fits.Column(name="COVERAGE", format="E", array=[b["coverage"] for b in bands]),
        ],
        name="BANDS",
    )

    hdul = fits.HDUList(
        [
            fits.PrimaryHDU(header=hdr),
            fits.ImageHDU(data=flux_cube, header=hdr.copy(), name="FLUX"),
            fits.ImageHDU(data=weight_cube, header=hdr.copy(), name="WEIGHT"),
            bands_hdu,
        ]
    )
    hdul.writeto(path, overwrite=True)


def load_cube(path: Path) -> dict[str, Any]:
    """Inverse of :func:`save_epoch_cube_fits`."""
    with fits.open(path, memmap=False) as hdul:
        flux_cube = np.asarray(hdul["FLUX"].data, dtype=float)
        weight_cube = np.asarray(hdul["WEIGHT"].data, dtype=float)
        table = hdul["BANDS"].data
        bands = [
            {
                "name": str(table["NAME"][i]).strip(),
                "feature": str(table["FEATURE"][i]).strip(),
                "role": str(table["ROLE"][i]).strip(),
                "lam_center": float(table["LAM_UM"][i]),
                "dlambda": float(table["DLAMBDA_UM"][i]),
                "coverage": float(table["COVERAGE"][i]),
            }
            for i in range(len(table))
        ]
        wcs = WCS(hdul["FLUX"].header, naxis=2)
        hdr0 = hdul[0].header
        epoch_index = hdr0.get("SXEPOCH")
        mjd_min = float(hdr0["MJD_MIN"])
        mjd_max = float(hdr0["MJD_MAX"])
    return {
        "flux_cube": flux_cube, "weight_cube": weight_cube, "bands": bands, "wcs": wcs,
        "epoch_index": epoch_index, "mjd_min": mjd_min, "mjd_max": mjd_max,
    }


def tau_from_cube(
    cube: dict[str, Any], feature_name: str, kind: str, min_weight: float = 1.0
) -> dict[str, Any]:
    """Derive one feature's map directly from an already-loaded compact cube (see
    :func:`load_cube`) -- no raw exposures needed. Returns the same shape of dict as
    :func:`spherex_ice.tau.build_epoch_tau_map`, so it's a drop-in for the usual
    save/render helpers.
    """
    by_role = {b["role"]: i for i, b in enumerate(cube["bands"]) if b["feature"] == feature_name}
    if not {"blue", "line", "red"} <= by_role.keys():
        available = sorted({b["feature"] for b in cube["bands"]})
        raise ValueError(f"This cube has no '{feature_name}' bands (it has: {', '.join(available)})")

    i_blue, i_line, i_red = by_role["blue"], by_role["line"], by_role["red"]
    flux_cube, weight_cube = cube["flux_cube"], cube["weight_cube"]
    f_blue, f_line, f_red = flux_cube[i_blue], flux_cube[i_line], flux_cube[i_red]
    w_blue, w_line, w_red = weight_cube[i_blue], weight_cube[i_line], weight_cube[i_red]
    lam_blue = cube["bands"][i_blue]["lam_center"]
    lam_line = cube["bands"][i_line]["lam_center"]
    lam_red = cube["bands"][i_red]["lam_center"]

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
        "tau": tau, "kind": kind, "coverage": coverage,
        "f_blue": f_blue, "f_line": f_line, "f_red": f_red, "f_cont": f_cont,
        "w_blue": w_blue, "w_line": w_line, "w_red": w_red,
        "cov_blue": cube["bands"][i_blue]["coverage"], "cov_line": cube["bands"][i_line]["coverage"],
        "cov_red": cube["bands"][i_red]["coverage"],
        "lam_blue": lam_blue, "lam_line": lam_line, "lam_red": lam_red,
    }
