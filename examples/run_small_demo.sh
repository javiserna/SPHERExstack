#!/usr/bin/env bash
# Small end-to-end demo: query -> download -> cube -> animate, for a region small
# enough to finish in a few minutes on a laptop.
#
# NOTE on the region: the task that produced this package asked for a demo centered
# on RA 4h41m00s, Dec +25d52m00s (70.25, +25.8667 deg) -- the point used in the
# original scripts' own usage examples. Querying IRSA live, that exact spot (and a
# fairly wide area around it) currently has ZERO downloadable SPHEREx Level 2
# exposures -- SPHEREx's sky coverage is still being built up and is patchy. The
# nearest spot with a reliable concentration of exposures, found by querying IRSA
# directly, is used instead (RA 71.75, Dec +27.0, about 1.5 deg away). If you have
# a region you know has coverage, pass different numbers on the command line.
#
# Usage: examples/run_small_demo.sh [outdir] [ra_center] [dec_center] [size_deg]
set -euo pipefail

OUTDIR="${1:-demo_taurus}"
RA_CENTER="${2:-71.75}"
DEC_CENTER="${3:-27.0}"
SIZE_DEG="${4:-0.5}"
N_BINS=20

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$OUTDIR" "$REPO_ROOT/docs/img"
cd "$OUTDIR"

echo "== 1/5 Querying IRSA for exposures in the ROI =="
spherex-query --ra_center "$RA_CENTER" --dec_center "$DEC_CENTER" --size_deg "$SIZE_DEG" --out manifest.csv

echo "== 2/5 Downloading server-side cutouts (resumable) =="
spherex-download --manifest manifest.csv --roi manifest.roi.json --outdir cutouts --workers 6

echo "== 3/5 Building the (RA, Dec, lambda) cube =="
spherex-cube --list list.txt --roi manifest.roi.json --out cube.fits --n_bins "$N_BINS" --dlambda 0.1

echo "== 4/5 Animating the cube (skipping empty-coverage bins) =="
spherex-animate --from_cube cube.fits --out lambda_slices_full_res.gif --sec_per_frame 0.4 --min_coverage 0.05

python3 - "$PWD/lambda_slices_full_res.gif" "$REPO_ROOT/docs/img/demo_lambda_slices.gif" << 'EOF'
# docs/img assets are capped at 600px wide / 5MB for the README; the package's own
# animate.render_lambda_slice always renders at its native ~832px, so this is a
# display-only downscale of the GIF this script already produced, not a change to
# any library code. It reuses animate's own shared-palette quantization (instead of
# a plain PIL re-save) so the downscaled copy doesn't reintroduce the per-frame
# independent-palette color drift that function exists to avoid -- Pillow would
# otherwise recompute a fresh 256-color palette for each frame on a second save.
# PIL.ImageSequence.Iterator is deliberately NOT used here: it yields the SAME
# Image object reseeked in place on every iteration rather than one per frame, so
# collecting frames via list(ImageSequence.Iterator(...)) silently keeps only the
# last frame's pixels under every index -- seek()+copy() per frame avoids that.
import sys
from PIL import Image

from spherexstack.animate import _quantize_frames_to_shared_palette

src, dst = sys.argv[1], sys.argv[2]
im = Image.open(src)
durations = []
raw_frames = []
for i in range(im.n_frames):
    im.seek(i)
    f = im.convert("RGB").copy()
    w, h = f.size
    if w > 600:
        f = f.resize((600, int(h * 600 / w)), Image.LANCZOS)
    raw_frames.append(f)
    durations.append(im.info.get("duration", 400))
frames = _quantize_frames_to_shared_palette(raw_frames)
frames[0].save(dst, save_all=True, append_images=frames[1:], duration=durations, loop=0)
print(f"[DONE] {dst}")
EOF

echo "== 5/5 Building a single-wavelength mosaic at 3.4 um (PAH) =="
# With only a handful of exposures in a 0.5x0.5 deg demo box, a single detector's
# visit may not land any pixel in a narrow lambda +/- 0.05 um band; --dlambda 0.3
# widens the synthetic band enough to get real (partial) coverage for the demo.
# A larger ROI or more exposures gives narrower-band coverage too.
spherex-mosaic --list list.txt --roi manifest.roi.json --lambda_um 3.4 --dlambda 0.3 \
    --out "$REPO_ROOT/docs/img/demo_3p4um.png"

echo "Done. Cube: $OUTDIR/cube.fits"
echo "GIF:  docs/img/demo_lambda_slices.gif"
echo "PNG:  docs/img/demo_3p4um.png"
