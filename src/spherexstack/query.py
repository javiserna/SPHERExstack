"""Query IRSA's TAP/ADQL service for SPHEREx Level 2 exposures covering a sky box.

Lists every SPHEREx Level 2 (science) exposure that falls within an RA/Dec
box, without downloading anything. There is no 5x5 degree limit like IRSA's
web form -- that limit belongs to the form, not the API.

Returns a manifest CSV with ``obs_id``, ``s_ra``, ``s_dec``,
``energy_bandpassname``, ``t_min``, ``uri`` (relative path of the full FITS
file), ``contentlength`` (bytes of the full FITS, before any cutout/HDU
trimming); and a ``roi.json`` with the resolved box, so
:mod:`spherexstack.download` and the mosaic/cube builders use exactly the
same ROI without re-typing numbers.
"""

from __future__ import annotations

import argparse
import csv
import urllib.parse
import urllib.request

from spherexstack.roi import add_roi_cli_args, roi_from_args, save_roi

TAP_URL = "https://irsa.ipac.caltech.edu/TAP/sync"

QUERY_TEMPLATE = """
SELECT oc.obs_id, oc.s_ra, oc.s_dec, oc.energy_bandpassname, oc.t_min,
       a.uri, a.contentlength
FROM spherex.obscore oc
JOIN spherex.plane p ON oc.obs_publisher_did = p.obs_publisher_did
JOIN spherex.artifact a ON p.planeid = a.planeid
WHERE oc.calib_level = 2
  AND oc.dataproduct_type = 'image'
  AND a.producttype = 'science'
  AND oc.s_ra BETWEEN {ra_min:.6f} AND {ra_max:.6f}
  AND oc.s_dec BETWEEN {dec_min:.6f} AND {dec_max:.6f}
{lambda_clause}
"""

# em_min/em_max in spherex.obscore are in METERS (ObsCore standard).
LAMBDA_CLAUSE_TEMPLATE = """  AND oc.em_min <= {em_max_m:.9e}
  AND oc.em_max >= {em_min_m:.9e}
"""

# Native field of view of one SPHEREx detector, in degrees (approximate).
NATIVE_FOV_DEG = 3.47


def run_tap_query(
    ra_min: float,
    ra_max: float,
    dec_min: float,
    dec_max: float,
    lambda_um: float | None = None,
    dlambda_um: float = 0.1,
) -> str:
    """Run the ADQL query against IRSA's TAP sync endpoint, returning the raw CSV response."""
    lambda_clause = ""
    if lambda_um is not None:
        em_min_m = (lambda_um - dlambda_um) * 1e-6
        em_max_m = (lambda_um + dlambda_um) * 1e-6
        lambda_clause = LAMBDA_CLAUSE_TEMPLATE.format(em_min_m=em_min_m, em_max_m=em_max_m)
    query = QUERY_TEMPLATE.format(
        ra_min=ra_min, ra_max=ra_max, dec_min=dec_min, dec_max=dec_max, lambda_clause=lambda_clause
    )
    params = urllib.parse.urlencode({"query": query, "format": "csv"}).encode()
    req = urllib.request.Request(TAP_URL, data=params)
    with urllib.request.urlopen(req, timeout=180) as resp:  # noqa: S310 - fixed, trusted IRSA host
        return resp.read().decode("utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    add_roi_cli_args(ap)
    ap.add_argument("--out", default="manifest.csv")
    ap.add_argument(
        "--roi_out",
        default=None,
        help="Where to save the resolved roi.json. Default: <out> with a .roi.json extension",
    )
    ap.add_argument(
        "--lambda_um",
        type=float,
        default=None,
        help="If given, only fetch exposures whose spectral range (em_min/em_max) covers this "
        "wavelength (um) -- for monochromatic mosaics, see spherex-mosaic. Reduces the number "
        "of exposures a lot (each SPHEREx detector only covers part of 0.75-5 um).",
    )
    ap.add_argument(
        "--dlambda_um", type=float, default=0.1, help="Tolerance (um) around --lambda_um for the spectral filter."
    )
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    roi = roi_from_args(args)
    roi_out = args.roi_out or (args.out.rsplit(".", 1)[0] + ".roi.json")

    print(
        f"[INFO] ROI: RA [{roi['ra_min']:.4f}, {roi['ra_max']:.4f}]  "
        f"Dec [{roi['dec_min']:.4f}, {roi['dec_max']:.4f}]  "
        f"(center {roi['ra_center']:.4f},{roi['dec_center']:.4f}, "
        f"{roi['width_deg']:.2f} x {roi['height_deg']:.2f} deg)"
    )
    if args.lambda_um is not None:
        print(f"[INFO] Spectral filter: only exposures covering lambda={args.lambda_um} +/- {args.dlambda_um} um")

    print("[INFO] Querying IRSA's TAP service (spherex.obscore + plane + artifact)...")
    csv_text = run_tap_query(
        roi["ra_min"], roi["ra_max"], roi["dec_min"], roi["dec_max"], lambda_um=args.lambda_um, dlambda_um=args.dlambda_um
    )
    rows = list(csv.DictReader(csv_text.splitlines()))

    if not rows:
        print("[WARN] The query returned no results. Check the ROI.")
        return

    sizes = [int(r["contentlength"]) for r in rows if r.get("contentlength")]
    total_gb = sum(sizes) / 1e9
    fits_in_frame = roi["width_deg"] <= NATIVE_FOV_DEG and roi["height_deg"] <= NATIVE_FOV_DEG

    print(f"[INFO] {len(rows)} exposures (science, calib_level=2) found.")
    print(f"[INFO] Reference size with no cutout/HDU trimming (full frames): {total_gb:.2f} GB")
    if fits_in_frame:
        print(
            f"[INFO] The ROI ({roi['width_deg']:.2f}x{roi['height_deg']:.2f} deg) is smaller than the "
            f"native FOV (~{NATIVE_FOV_DEG} deg): /ibe/cutout WILL crop each file, actual download "
            "much smaller."
        )
    else:
        print(
            f"[INFO] The ROI ({roi['width_deg']:.2f}x{roi['height_deg']:.2f} deg) is LARGER than the "
            f"native FOV (~{NATIVE_FOV_DEG} deg): most exposures will download close to full size "
            "(dropping unused HDUs in spherex-download still gives ~50-60% savings)."
        )

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    save_roi(roi_out, roi)

    print(f"[DONE] Manifest: {args.out}")
    print(f"[DONE] ROI:      {roi_out}")
    print("[INFO] Next step:")
    print(f"  spherex-download --manifest {args.out} --roi {roi_out} --outdir cutouts_roi")


if __name__ == "__main__":
    main()
