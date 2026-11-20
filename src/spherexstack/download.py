"""Download server-side cutouts for a query manifest, resumable and parallel.

For each row of a manifest produced by :mod:`spherexstack.query`, downloads a
cutout (``/ibe/cutout``, not the full 2040x2040 px / ~71.6 MB frame) centered
on the same ROI used in the query, instead of the full frame. The cutout keeps
native resolution (~6.2"/px) and only the HDUs the pipeline needs (``IMAGE``,
``FLAGS``, ``WCS-WAVE``, plus ``VARIANCE``/``ZODI``/``PSF`` on request).

Resumable: a file that already exists with a size > 0 is skipped.
"""

from __future__ import annotations

import argparse
import csv
import io
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from astropy.io import fits

from spherexstack.progress import ProgressBar
from spherexstack.roi import add_roi_cli_args, roi_from_args

CUTOUT_URL = "https://irsa.ipac.caltech.edu/ibe/cutout"
DATA_PREFIX = "ibe/data/"

# The pipeline only reads IMAGE, FLAGS and WCS-WAVE (plus VARIANCE for ivar
# weighting). The other HDUs the cutout brings (ZODI, and especially PSF,
# which alone is ~50% of the file) are dropped by default to save ~60% of
# disk space, with no loss of resolution or quality for this technique. Use
# --keep-hdus to keep them all (e.g. to add VARIANCE for ivar weighting).
KEEP_HDUS_DEFAULT = ("PRIMARY", "IMAGE", "FLAGS", "WCS-WAVE")


def trim_hdus(data: bytes, keep: tuple[str, ...]) -> bytes:
    """Return ``data`` (a FITS file's raw bytes) with only the HDUs named in ``keep``."""
    with fits.open(io.BytesIO(data), memmap=False) as hdul:
        trimmed = fits.HDUList([hdu for hdu in hdul if hdu.name in keep])
        buf = io.BytesIO()
        trimmed.writeto(buf)
        return buf.getvalue()


def cutout_path_from_uri(uri: str) -> str:
    """``/ibe/cutout`` expects the path WITHOUT the ``ibe/data/`` prefix."""
    if uri.startswith(DATA_PREFIX):
        return uri[len(DATA_PREFIX) :]
    return uri


def download_one(
    row: dict,
    ra_center: float,
    dec_center: float,
    size_deg: float,
    outdir: Path,
    keep_hdus: tuple[str, ...],
    retries: int = 3,
) -> tuple[Path, str, int]:
    """Download one manifest row's cutout. Returns ``(path, status, n_bytes)``.

    ``status`` is ``"ok"``, ``"skip"`` (already on disk), or ``"FAIL: <error>"``.
    """
    uri = row["uri"]
    fname = uri.rsplit("/", 1)[-1]
    out_path = outdir / fname

    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path, "skip", out_path.stat().st_size

    path = cutout_path_from_uri(uri)
    params = urllib.parse.urlencode(
        {"path": path, "ra": f"{ra_center:.6f}", "dec": f"{dec_center:.6f}", "size": f"{size_deg:.6f}"}
    )
    url = f"{CUTOUT_URL}?{params}"

    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=180) as resp:  # noqa: S310 - fixed, trusted IRSA host
                data = resp.read()
            if len(data) < 2880:
                raise ValueError(f"response too small ({len(data)} bytes), doesn't look like FITS")
            if keep_hdus:
                data = trim_hdus(data, keep_hdus)
            tmp_path = out_path.with_suffix(out_path.suffix + ".part")
            tmp_path.write_bytes(data)
            tmp_path.rename(out_path)
            return out_path, "ok", len(data)
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    return out_path, f"FAIL: {last_err}", 0


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    add_roi_cli_args(ap)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--list_out", default="list.txt")
    ap.add_argument(
        "--keep-hdus",
        default=",".join(KEEP_HDUS_DEFAULT),
        help="Comma-separated HDUs to keep. Default: only what the pipeline uses "
        "(IMAGE,FLAGS,WCS-WAVE + PRIMARY), ~60%% less space. Use --keep-hdus=ALL to keep everything "
        "(needed for VARIANCE, e.g. for --weighting ivar).",
    )
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    roi = roi_from_args(args)
    # /ibe/cutout only accepts a single (square) "size". For a rectangular ROI we request
    # the longer side -- if the ROI is smaller than the native FOV (~3.47 deg) this still
    # crops each file; if it's larger, each file is close to full size either way, so there
    # is no loss from using the longer side.
    cutout_size_deg = max(roi["width_deg"], roi["height_deg"])

    keep_hdus = (
        ()
        if args.keep_hdus.strip().upper() == "ALL"
        else tuple(h.strip().upper() for h in args.keep_hdus.split(",") if h.strip())
    )

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    with open(args.manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    print(
        f"[INFO] {len(rows)} files in the manifest. Downloading cutouts to {outdir} "
        f"(ROI center {roi['ra_center']:.4f},{roi['dec_center']:.4f}, "
        f"cutout size={cutout_size_deg:.2f} deg) with {args.workers} workers. "
        f"HDUs kept: {keep_hdus or 'ALL'}"
    )

    ok, skipped, failed = [], [], []
    total_bytes = 0

    bar = ProgressBar(total=len(rows), prefix="Downloading", unit="files")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(download_one, row, roi["ra_center"], roi["dec_center"], cutout_size_deg, outdir, keep_hdus): row
            for row in rows
        }
        for fut in as_completed(futures):
            out_path, status, nbytes = fut.result()
            if status == "ok":
                ok.append(out_path)
                total_bytes += nbytes
            elif status == "skip":
                skipped.append(out_path)
                total_bytes += nbytes
            else:
                failed.append((out_path, status))
                bar.interrupt(f"[WARN] {out_path.name}: {status}")
            bar.update()
    bar.close()

    print(f"[DONE] ok={len(ok)} skip={len(skipped)} fail={len(failed)}  total on disk={total_bytes / 1e9:.2f} GB")
    if failed:
        print("[WARN] Failed (check connectivity or the manifest's 'uri'):")
        for p, err in failed:
            print(f"   {p.name}: {err}")

    all_files = sorted(str(p) for p in outdir.glob("*.fits"))
    with open(args.list_out, "w") as f:
        f.write("\n".join(all_files) + "\n")
    print(f"[DONE] {args.list_out} with {len(all_files)} files.")


if __name__ == "__main__":
    main()
