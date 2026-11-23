# SPHERExice

**Point this at a field and get back real per-pixel, per-epoch maps of specific SPHEREx
features.** Water, CO2, and CO ice absorption. The 3.3 µm PAH emission feature. The Pa-delta,
Pa-beta, Pa-alpha, Br-beta, and Br-alpha hydrogen recombination lines. Contour masks and
multi-feature RGB composites come free with every map, no extra steps required.

## Which command do I need?

| I want to... | Run this |
|---|---|
| A water-ice, CO2-ice, or CO-ice optical depth map | `spherex-ice-get --feature h2o` (or `co2`, `co`) |
| A PAH emission map, or one of the hydrogen recombination lines | `spherex-ice-get --feature pah` (or `padelta`/`pabeta`/`paalpha`/`brbeta`/`bralpha`) |
| Several features from one download, plus a composite | `spherex-ice-get --feature h2o co2 pah --auto-rgb` |
| To re-run just the mapping step on data you already downloaded | `spherex-ice-map`, see [Step by step](#step-by-step) |
| The real spectrum behind one specific pixel | `spherex-ice-inspect` |
| An RGB or blended composite from maps you already built | `spherex-ice-rgb` or `spherex-ice-blend` |
| To reprocess a field after the raw cutouts are gone | `spherex-ice-from-cube` |

## What this does

Point it at a field, pick one spectral line at a time, and it builds:

1. A map of that line's strength, pixel by pixel, for each separate observation epoch (see
   [Why epochs?](#why-epochs) below), using optical depth for an ice absorption line and flux
   excess (MJy/sr) for an emission line.
2. A contour mask of where that strength crosses a threshold (a sensible default, or one you
   choose), as both a picture and a reusable FITS file with real sky coordinates.
3. Optionally, an RGB composite stitching up to three such maps together, any mix of lines, any
   mix of epochs, into one false-color picture.

It's built on top of [`spherexstack`](../..), this repo's core package one level up, which
handles querying IRSA, downloading cutouts, and reprojecting/coadding exposures onto a common
pixel grid. What this package adds on top is specific to feature mapping, grouping exposures into
observation epochs, the continuum-interpolation math, contour masks, and multi-feature
composites. The [repo-level README](../../README.md) covers the shared stuff, the two-package
layout, how a flaky download gets retried, and what to expect time- and disk-wise on a big field.

## Science cases

- **Where's the water ice around a young star?** Run `--feature h2o` on a field around a known or
  candidate embedded protostar, and compare the source itself against the surrounding cloud.
- **Ice absorption vs. PAH emission.** These two trace very different physical things (shielded,
  dense gas vs. UV-lit surfaces), and are expected to avoid each other spatially. An RGB composite
  of `--feature h2o` (or `co2`) against `--feature pah` shows that pretty directly; see
  [RGB composites](#rgb-composites-up-to-three-maps-in-one-picture).
- **Is that ice signature real, or a one-off glitch?** Build the same field across multiple
  epochs and look for something that repeats independently, rather than trusting whatever one
  epoch's map happens to show; see [Caveats](#caveats-read-before-trusting-a-value).
- **Tracing ionized gas or outflows** with the hydrogen recombination lines. `--feature bralpha`
  (or any of the other four) around an HII region or a protostellar jet. Br-alpha and Pa-beta in
  particular get used a lot as relatively extinction-robust accretion/ionization tracers.
- **A reusable boundary for "where is this feature actually significant"**, the contour FITS
  mask from any map works fine as a spatial selection for your own analysis outside this package,
  photometry, stacking, whatever.
- **Anything else SPHEREx covers.** `--feature custom` with your own line center and continuum
  bands. None of the named features above are special-cased in the pipeline; they're just
  convenient presets.

## Two kinds of quantity, in plain terms

Every map here works the same basic way. Measure the flux right at the line, measure it on either
side of the line too (the "continuum"), and compare the two. What differs is *how* they're
compared, and that comes down to whether the line sits below the continuum or above it.

For ice absorption (H2O, CO2, CO), the line sits below the continuum, blocking some of the light
passing through, so the natural quantity is optical depth, tau, defined as `tau = ln(F_continuum /
F_line)`. Zero means no detectable absorption, 1 means the line's down to about 37% of the
continuum (1/e), and it climbs from there with more ice along the sightline. It's a ratio, so it
has no physical units and doesn't care how bright the background source happens to be.

PAH and the recombination lines work the other way. They sit *above* the continuum, adding light
rather than removing it, so a ratio doesn't make sense (and "optical depth" wouldn't mean
anything for something that's emitting, not absorbing). The map there is just `F_line -
F_continuum`, in the same MJy/sr surface-brightness units as the SPHEREx images themselves, how
much extra light that pixel is putting out at that wavelength, above what the continuum alone
would predict.

`--feature` picks which of the two applies, along with which bands to use; see
[Named features](#named-features---feature).

## Why epochs?

SPHEREx builds a full 0.75–5 µm spectrum of a field over many separate exposures as its linear
variable filter scans across the detector. In practice that takes something like 15–20 days,
after which the field doesn't get revisited again until much later in the survey. To see whether
a feature along a sightline is actually changing over time, or just to check that what you're
looking at is real and not a one-off, each of those visit windows gets treated as its own
"epoch," with its own independent map, instead of averaging everything together from the start.

## Install

This package lives in the same repo as `spherexstack` (see the
[repo-level README](../../README.md)), so from a clone of it:

```bash
# from the repo root:
pip install -e ".[animate]"              # spherexstack core, this package reuses its query/download/mosaic code
pip install -e "packages/spherex-ice[dev]"
```

## Quick start: one command, one field, one line

```bash
spherex-ice-get --ra_center 86.72955 --dec_center 0.392931 --size_deg 0.15 --feature h2o --out my_field
```

This resolves the field, queries IRSA (using a query box at least one native SPHEREx detector FOV
wide, since IRSA's spatial filter matches on each exposure's *pointing center*, not its actual
footprint, so a small query box can otherwise miss exposures that still cover your field),
downloads cutouts for only the field you actually asked for, groups them into epochs, and for
each epoch writes:

- `tau_maps/tau_epochNN.fits` / `.png`, the feature map itself
- `tau_maps/tau_epochNN_contour.fits` / `_contour.png`, the contour mask
- `tau_maps/tau_median.fits` / `.png` (plus its own contour), the per-pixel median across every
  epoch, if there was more than one (see [Median across epochs](#median-across-epochs))

And that's genuinely all `ls my_field` shows you. The downloaded cutouts and the query manifest
stick around too, along with `roi.json`/`epochs.json`, but they're tucked away in a hidden
`my_field/.spherex_ice/` so they don't clutter up the actual products. That's also where to point
`spherex-ice-map` later if you want to re-run just the mapping step (see
[Step by step](#step-by-step)) without downloading anything again.

Raise `--size_deg` for a bigger field, `1.0` for a full square degree, say. Same
command, just more data and a longer runtime, since both the download and the reprojection scale
with area. The repo-level README's
[Performance, time, and disk space](../../README.md#performance-time-and-disk-space-read-before-a-large-field)
section has rough numbers for 1° and 5° fields, plus how a flaky download (a timeout, a dropped
connection) gets retried automatically these days instead of just failing on you.

There's no hard cap on `--size_deg`. IRSA's API doesn't enforce one, unlike the 5°x5° limit on
its web form, but `spherex-ice-get` will warn you if you go past one native SPHEREx detector FOV
(about 3.47°). Past that point, `/ibe/cutout` stops cropping files down at all, so every exposure
comes back close to its full ~72 MB, and a bigger area also just means more exposures from the
survey to begin with, so both the download and the background-matching step (which scales
roughly with the square of the exposure count) can slow down a lot. The warning prints the
reference grid size before any network traffic even starts, so you can judge the cost before
committing to it.

## Named features (`--feature`)

| `--feature` | line | kind | continuum bands | default contour | source of the ranges |
|---|---|---|---|---|---|
| `h2o` (default) | 3.05 um | absorption | 2.50-2.75 / 3.75-4.00 um | 0.5 tau | this project's `ice_mantle_fitter_multiepoch.py` |
| `co2` | 4.27 um | absorption | 3.90-4.10 / 4.45-4.60 um | 0.2 tau | same, `--co2_cont_*` |
| `co` | 4.67 um | absorption | 4.40-4.55 / 4.85-5.00 um | 0.2 tau | same, `--co_cont_*` |
| `pah` | 3.29 um | emission | 2.67-2.87 / 3.75-3.95 um | auto (30% of p95) | `spherexstack`'s own PAH mosaic (2.77/3.85 um continuum points) |
| `padelta` | 1.0052 um | emission | 0.94-0.98 / 1.04-1.08 um | auto (30% of p95) | standard vacuum wavelength; continuum bands are new, see below |
| `pabeta` | 1.2818 um | emission | 1.19-1.23 / 1.33-1.37 um | auto (30% of p95) | same |
| `paalpha` | 1.8751 um | emission | 1.78-1.82 / 1.93-1.97 um | auto (30% of p95) | same |
| `brbeta` | 2.6259 um | emission | 2.50-2.56 / 2.69-2.75 um | auto (30% of p95) | same |
| `bralpha` | 4.0512 um | emission | 3.95-3.99 / 4.10-4.14 um | auto (30% of p95) | same |

`--feature` fills in the line/continuum bands and the absorption-vs-emission kind for you
(`tau.FEATURE_PRESETS` in the code, if you want to see exactly what it's doing). Unlike
h2o/co2/co/pah, the five recombination-line continuum bands haven't been independently validated
against real SPHEREx data in this project; they're a reasonable starting point, picked to be
clear of other known lines, but check the result for your own field and adjust
`--cont_blue_lo/hi`/`--cont_red_lo/hi`/`--dlambda_line` (via `--feature custom`) if something
looks off. One thing worth flagging. Br-beta's line actually sits inside H2O's blue continuum band, so a
strong Br-beta source will add a little excess emission right into an H2O map's continuum there.

```bash
spherex-ice-get --ra_center 86.72955 --dec_center 0.392931 --size_deg 1.0 --feature co2 --out my_field_co2
spherex-ice-get --ra_center 86.72955 --dec_center 0.392931 --size_deg 1.0 --feature bralpha --out my_field_bralpha
```

### Several features in one command

`--feature` takes more than one name at once (space-separated, case-insensitive). The field only
gets downloaded once, and each feature gets its own map, contour, and median in its own
subdirectory:

```bash
spherex-ice-get --ra_center 86.72955 --dec_center 0.392931 --size_deg 0.5 --feature h2o pah co2 --out my_field
# -> my_field/tau_maps/h2o/tau_median.fits, my_field/tau_maps/pah/..., my_field/tau_maps/co2/...
```

Add `--auto-rgb` to build the composite right away too (first feature named goes red, second
green, third blue), skipping a separate `spherex-ice-rgb` call. It works off each feature's
median map, so it needs that median actually built (won't play nice with `--no-median`):

```bash
spherex-ice-get --ra_center 86.72955 --dec_center 0.392931 --size_deg 0.5 --feature h2o co2 pah --auto-rgb --out my_field
# -> ... plus my_field/tau_maps/composite.png/.fits (R=h2o, G=co2, B=pah)
```

Any other line works too, through `--feature custom` plus your own
`--lam_line`/`--dlambda_line`/`--cont_blue_lo/hi`/`--cont_red_lo/hi`. The pipeline has nothing
line-specific baked into it; only these presets sit on top.

## Step by step

For re-running just the mapping step (to try a different `--feature` or `--contour_level` on data
you've already downloaded, say), or for full control over the query/download steps yourself:

```bash
spherex-query --ra_center 86.72955 --dec_center 0.392931 --size_deg 1.0 --out manifest.csv
spherex-download --manifest manifest.csv --roi manifest.roi.json --outdir cutouts --workers 4
spherex-ice-epochs --list list.txt --out epochs.json
spherex-ice-map --epochs epochs.json --roi manifest.roi.json --epoch all --feature co2 --out_dir tau_maps_co2
```

(The query step above uses a 1.0° box even if you only want a smaller field downloaded;
`spherex-ice-get` does this for you automatically. Doing it by hand, download the actual cutout
size you want with `--ra_center`/`--dec_center`/`--size_deg` on `spherex-download` separately from
the query box.) `--epoch N` instead of `--epoch all` builds just the one epoch's map.

## Median across epochs

Whenever more than one epoch gets processed (which is the default with `--epoch all`), an extra
product gets built automatically alongside the per-epoch maps (skip it with `--no-median`).
`tau_median.fits`/`.png`, plus its own contour, is the per-pixel median of tau across every epoch.

This isn't just a cleaner-looking average, it's a genuinely more robust product. It shrugs off
the kind of single-epoch glitch this project has actually run into (one outlier exposure, a
band-to-band registration mismatch near a star) as long as fewer than half the combined epochs are
affected at a given pixel. The tradeoff is that it can't show real time variability anymore, so
use both. The separate epoch maps show whether a feature is stable or changing (see
[Caveats](#caveats-read-before-trusting-a-value)), while the median gives the field's overall,
most-trustworthy structure. `tau_median.fits` also keeps an `N_EPOCHS` HDU around, recording how
many of the combined epochs actually had a valid pixel there (just 1 where only one epoch covered
it, same as not combining at all).

```bash
spherex-ice-map --epochs epochs.json --roi manifest.roi.json --epoch all --feature h2o --out_dir tau_maps
# -> tau_maps/tau_epoch00.fits, tau_epoch01.fits, ..., AND tau_maps/tau_median.fits
```

Only want the combined map, without each epoch's own files cluttering up the output directory?
Add `--median-only`. Every epoch's data still goes into computing the median; only the per-epoch
`tau_epochNN.*` files themselves get skipped.

```bash
spherex-ice-map --epochs epochs.json --roi manifest.roi.json --epoch all --feature h2o --median-only --out_dir tau_maps
# -> just tau_maps/tau_median.fits/.png (+ its contour), nothing per-epoch
```

## Contour maps

Built automatically alongside every map (skip with `--no-contour`), a boundary at
`--contour_level` that defaults to each feature's own level, 0.5 tau for H2O, 0.2 tau for CO2/CO,
or for an emission feature (or `--feature custom`) 30% of that epoch's own 95th-percentile signal,
since there's no single physical cutoff that works across lines of wildly different brightness.
It comes out two ways:

- `tau_epochNN_contour.png`, the same map, with the contour line drawn right on top, so you can
  see exactly where the boundary falls.
- `tau_epochNN_contour.fits`, a plain 0/1 mask (1 meaning at or above the level), with the same
  WCS as the map it came from, ready to use as a spatial selection anywhere else. It's a normal
  FITS image with real RA/Dec coordinates, nothing SPHEREx- or Python-specific about it.

```bash
spherex-ice-map --epochs epochs.json --roi manifest.roi.json --epoch 0 --feature h2o --contour_level 0.3 --out_dir tau_maps
```

## Checking a pixel's real spectrum (`spherex-ice-inspect`)

Every map here comes from just a handful of narrow synthetic bands, the line plus its two
continuum windows (see [Two kinds of quantity](#two-kinds-of-quantity-in-plain-terms)), not the
full SPHEREx spectrum. Before you put too much weight on an interesting pixel (a bright spot in a
contour, an edge that looks a little too sharp, a value that just seems off), it's worth looking
at the actual flux at every wavelength SPHEREx has observed there, not just the three narrow bands
the map used. That's what `spherex-ice-inspect` is for, once a field's already downloaded:

```bash
spherex-ice-inspect --tau_dir my_field/tau_maps --feature h2o pah \
    --cutouts my_field/.spherex_ice/cutouts
```

This opens two windows, the median map(s) you asked for (`h2o`/`pah` here, side by side, sharing
pan and zoom), and an empty spectrum plot next to them. Double-click anywhere on either map and
it drops a colored marker there on every map shown, then plots that exact sky position's real,
per-exposure SPHEREx spectrum (flux vs. wavelength, one point per raw exposure that happened to
cover it, pulled straight from the downloaded cutouts, not re-derived from the map) in the
matching color on the second window. Double-click somewhere else to add another point in a new
color, so you can compare an interesting pixel against a quiet patch of sky nearby. A "Guardar"
button saves everything shown so far as CSV plus a PNG of each window, under `--out_dir` (default
`spectra`); "Borrar todo" clears every marker to start fresh.

This is also the most direct way to sanity-check
[the caveat below](#the-one-thing-every-map-here-can-get-fooled-by) for a specific pixel. A real
absorption or emission feature shows up as an actual dip or bump at the expected wavelength among
the raw points. A jump that's just a step between the two continuum bands, with nothing
resembling a line in between, usually isn't one.

A single pixel can be too noisy to judge a feature by, especially for an extended, diffuse
source rather than a point source. Add `--aperture_px` to switch from a single pixel to a
background-subtracted circular-aperture spectrum instead (radius in SPHEREx native pixels, with
the background estimated from a surrounding annulus and subtracted off), the same recipe
[IRSA's own SPHEREx mosaic-cube tutorial](https://caltech-ipac.github.io/irsa-tutorials/spherex-mosaic/)
uses:

```bash
spherex-ice-inspect --tau_dir my_field/tau_maps --feature h2o pah \
    --cutouts my_field/.spherex_ice/cutouts --aperture_px 3
```

The background annulus defaults to 2x/3x `--aperture_px` (override with `--annulus_in_px`/
`--annulus_out_px`), and each exposure gets converted from MJy/sr to mJy first (see the root
README's [Outputs](../../README.md#outputs) section) so the aperture sum is a genuine flux,
reported in mJy rather than the single-pixel mode's MJy/sr.

## RGB composites (up to three maps in one picture)

`spherex-ice-rgb` combines up to three already-built `tau_epochNN.fits` maps, any mix of
features, any mix of epochs, as long as they share the same pixel grid (built from the same
`--roi`/`--pixscale`), into one false-color picture with the red/green/blue channel assigned by
you:

```bash
spherex-ice-rgb \
  --red   tau_maps_h2o/tau_epoch00.fits --red_label   "H2O tau" \
  --green tau_maps_co2/tau_epoch00.fits --green_label "CO2 tau" \
  --blue  tau_maps_pah/tau_epoch00.fits --blue_label  "PAH MJy/sr" \
  --out composite.png --out_fits composite.fits
```

Each channel is clipped and scaled independently (its own 99th percentile by default; override
per channel with `--red_vmax`/`--green_vmax`/`--blue_vmax`). `--out_fits` also saves the
composite as a 3-plane FITS cube (R/G/B HDUs, shared WCS) if you want to reuse it elsewhere. One
or two channels is fine too, the rest just render black.

## Per-feature colormaps, and blending them (`spherex-ice-blend`)

H2O and PAH each get their own perceptual colormap instead of plain grayscale, H2O a
black-to-saturated-blue-to-white "ice" scale, PAH a black-red-orange-yellow-white "heat" scale
(see each preset's `"cmap"` in `tau.FEATURE_PRESETS`; everything else defaults to grayscale).
`spherex-ice-blend` combines 2-3 such maps while keeping each one's own color, by screen-blending
them (`1 - (1-a)*(1-b)`, the usual trick for combining bright-on-black glow-style images). This is
different from `spherex-ice-rgb` above, which stacks maps as literal red/green/blue primaries
instead:

```bash
spherex-ice-blend --map1 tau_maps/h2o/tau_median.fits --map2 tau_maps/pah/tau_median.fits --out blend.png
```

Each map's colormap and clip default to its own `FEATURE` header, looked up against
`FEATURE_PRESETS` (override with `--cmap1`/`--cmap2`/`--cmap3`, `--vmax1`/`--vmax2`/`--vmax3`); a
feature with no preset colormap just falls back to grayscale.

## Reading the images

Every PNG, map, contour, or composite, gets plotted with real RA/Dec axes (astropy's WCSAxes,
decimal degrees), not just pixel indices, so you can read a sky position straight off the plot and
check the target reticle against the axis labels.

By default, everything renders with a white background, black text, and the data itself as a
photographic negative (more signal = darker). Pass `--dark` for the opposite, classic dark theme
instead (black background, white text, more signal = brighter). The colorbar title is the
feature's own symbol ($\tau_{peak}^{H_2O}$ for water ice, $\Delta F_{PAH}$ for the PAH excess; see
each preset's `"symbol"`), and the bottom-left label is just the MJD range and coverage (no
"epoch N," since the MJD range already tells epochs apart, and the median's much wider range
makes it obvious which is which anyway).

A feature with its own colormap (H2O, PAH) always writes out both versions. The main
`tau_epochNN.png` (and `tau_median.png`) in that colormap against black, brighter meaning more
signal regardless of `--dark`, plus a `tau_epochNN_bw.png`/`tau_median_bw.png` companion in the
classic grayscale rendering, which does respect `--dark`. A feature with no preset colormap just
gets the one grayscale file, same as always.

## Compact cubes: post-processing without the raw cutouts (`--cube`)

The expensive part of this whole package is reading and reprojecting the raw per-exposure FITS
cutouts, and there can be hundreds of them, individually small but numerous. Add `--cube` to
`spherex-ice-get`/`spherex-ice-map` and, at no extra cost (it just repackages the blue/line/red
mosaics each `--feature` already built for its own map), you get one compact FITS file per epoch
(`cubes/epochNN_cube.fits`), bundling those bands together with a `BANDS` table saying what each
slice actually is:

```bash
spherex-ice-get --ra_center 86.72955 --dec_center 0.392931 --size_deg 0.5 --feature h2o co2 pah --cube --out my_field
```

`--cube` on its own keeps the raw cutouts too (both sit on disk at once), so it doesn't save space
by itself, just adds a compact summary alongside. To actually reclaim the disk space, add
`--delete-cutouts-after-cube`. Once every epoch's cube is written and verified to load back
correctly, it deletes `.spherex_ice/cutouts/` (the tiny manifest/roi/epochs.json stay, so you can
re-download the raw data later if you ever need to):

```bash
spherex-ice-get --ra_center 86.72955 --dec_center 0.392931 --size_deg 0.5 --feature h2o co2 pah --cube --delete-cutouts-after-cube --out my_field
```

On a real 3-epoch, 2-feature field I tested, this took 381 raw cutouts (59 MB) down to 3 cube
files (1.9 MB), about 31x smaller, 127x fewer files. Re-deriving a map from the cube afterward is
also a lot faster, not just smaller, about 7x faster in that same test (67.8s down to 9.8s), and
the gap only widens for bigger fields, since building from raw cutouts scales with the number of
exposures (background-matching is roughly O(n²) per band), while `spherex-ice-from-cube` scales
only with the pixel grid size, since it never touches an exposure again.

Later, in a completely separate session, after the raw cutouts are gone, wherever you happen to
be working, derive any of those same features' maps (contour, median, `--auto-rgb` composite,
all of it) straight from the cube, with the exact same flags `spherex-ice-map` takes, but instant,
with no raw exposures on hand at all:

```bash
spherex-ice-from-cube --input_cube my_field/tau_maps/cubes --feature h2o pah --auto-rgb --out_dir reprocessed
```

`--input_cube` takes either one `epochNN_cube.fits` or a whole directory of them (it builds the
median the same way, across whichever cubes you point it at). One catch. A cube only has the
bands for whatever `--feature`s were requested when it was built, so asking
`spherex-ice-from-cube` for something that isn't in there gives you a clear error naming what the
cube actually does have, rather than quietly making something up.

## Output reference

| File | What it is |
|---|---|
| `tau_epochNN.fits` | `TAU` (the map itself), `F_BLUE`/`F_LINE`/`F_RED`/`F_CONT` (the flux mosaics it was built from), `W_BLUE`/`W_LINE`/`W_RED` (per-pixel exposure-count weight, one per band) |
| `tau_epochNN.png` | The map rendered: brighter means more signal, saturating at `--vmax`, using that feature's own colormap if it has one (H2O, PAH), plain grayscale otherwise |
| `tau_epochNN_bw.png` | Only for a feature with its own colormap: the same map again, in the classic grayscale negative/`--dark` rendering, so both versions are always on hand |
| `tau_epochNN_contour.fits` | 0/1 mask at `--contour_level`, same WCS as the map |
| `tau_epochNN_contour.png` / `_contour_bw.png` | The map with the contour line drawn on it (colored + grayscale, same rule as above) |
| `tau_median.fits` | `TAU` (per-pixel median across epochs), `N_EPOCHS` (how many epochs covered each pixel), only built when more than one epoch was processed |
| `tau_median.png` / `_bw.png` / `_contour.fits` / `_contour.png` / `_contour_bw.png` | Same rendering/contour products as a single epoch's map, just built from the median instead |
| `composite.png` / `.fits` | A `spherex-ice-rgb` output, or `spherex-ice-get --auto-rgb`'s (see above) |
| `cubes/epochNN_cube.fits` | `FLUX`/`WEIGHT` (one slice per band) plus a `BANDS` table, only built with `--cube`; feeds straight into `spherex-ice-from-cube` |
| `.spherex_ice/` (hidden) | `spherex-ice-get`'s own plumbing: `roi.json`, `epochs.json`, `manifest.csv`, `list.txt`, `cutouts/`, kept for reuse with `spherex-ice-map` (`--roi`/`--epochs` point here), just out of the way of the actual products. `cutouts/` is what `--delete-cutouts-after-cube` removes; the rest stays in case you want to re-download later |

## Defaults and where they come from

- **Epoch grouping** starts a new epoch after a 30+ day gap with no exposures
  (`--max_gap_days`), or if one epoch would otherwise grow past 45 days (`--max_span_days`).
  Tuned against a real field where a single visit had internal gaps up to ~24 days, but true
  revisits were 130+ days apart; check what `spherex-ice-epochs` prints and adjust if it doesn't
  match your field's actual cadence.
- **Pixel scale** is 6.15"/px, SPHEREx's native resolution.

Every one of these is an adjustable CLI flag; see `spherex-ice-get --help`.

## Caveats (read before trusting a value)

### The one thing every map here can get fooled by

Every map comes from comparing flux in three narrow, fixed wavelength windows, the line, plus two
continuum bands (see [Two kinds of quantity](#two-kinds-of-quantity-in-plain-terms)). Nothing in
the pipeline fits or identifies an actual line profile; it just measures what falls inside those
windows. Which means the map genuinely can't tell the difference between "the feature I asked for
is present" and "something else landed in one of these three windows." This is the single most
important thing to know before trusting a value, and it applies to every feature, not just an edge
case.

It's not hypothetical, either. One of the named presets runs right into it. Br-beta's line sits
inside H2O's own blue continuum band (see [Named features](#named-features---feature)), so a
strong Br-beta source raises that continuum and can make a map read *less* H2O absorption than is
actually there. The reverse, an unrelated absorption feature landing inside a continuum band,
would read as spuriously *more*. The same goes for `--feature custom` with hand-picked bands;
nothing in the code checks what else might be living inside them. A map answers "is there more or
less signal here, in these three windows, than elsewhere," which is usually exactly what you want,
but it's not a substitute for an actual spectral fit. For any pixel you're about to rely on, pull
up [`spherex-ice-inspect`](#checking-a-pixels-real-spectrum-spherex-ice-inspect) and look at the
real spectrum before trusting it.

### A handful of smaller things worth knowing

**It's early mission data.** SPHEREx has only partially scanned most fields so far, so some epochs
can come back with zero coverage in a given band simply because too few exposures happened to
sample that wavelength in that window. That's expected, not a bug, and it fills in as the survey
continues.

**You're often working with very few exposures.** A given band in a given epoch is frequently
built from just 1-8 exposures (compare that to ~148 for the wide CygX maps in Hora et al. 2026).
One uncorrected bad or hot pixel can produce an extreme, isolated spike, so get in the habit of
checking the `F_LINE`/`F_BLUE`/`F_RED` HDUs (or just a small median box) around any pixel before
trusting its exact value. A signal that's extended over several pixels and shows up independently
across epochs is a lot more trustworthy than any single pixel ever is.

**One bad exposure can throw off the whole field, not just its own footprint.** With so few
exposures per band, a single one with an off flux level (a calibration glitch, stray light) has
no redundancy to get down-weighted by background-matching; its offset propagates through the
shared correction solve and can flip the sign of tau across most of the grid. This actually
happened on this project's own test field. One bad exposure, out of only 3 contributing to the
H2O line band, shifted an epoch's median tau from +0.34 to -0.11, with 62% of pixels reading
negative, until it got caught and excluded. `spherex-ice-map`/`spherex-ice-get` now reject that
kind of exposure automatically before building each band (anything whose own median in-band flux
is off by more than 4x from its peers; see `tau.OUTLIER_FACTOR`); pass `--no-reject-outliers` if
you want the raw, unfiltered behavior instead.

**Check a pixel's support before trusting it**, meaning its `W_BLUE`/`W_LINE`/`W_RED` values.
Every `tau_epochNN.fits` stores, per band, how many exposures actually landed on each pixel. A
pixel backed by just one exposure in a band is exactly where a leftover band-to-band registration
mismatch (a star's PSF landing on a slightly different pixel in each band) shows up as an isolated
bright/dark pixel pair, since there's no second sample around to average it down. `--min_weight 2`
on `spherex-ice-map`/`spherex-ice-get` masks out any pixel not backed by at least 2 exposures in
all three bands; you lose a little coverage, but far fewer of these one-off dropouts.

**There's some banding.** SPHEREx's narrow synthetic bands sample a line at whatever wavelength
each exposure's dither happened to land on, which is usually a little off from the line center,
not just at its peak. Hora et al. (2026, Section 2.1.2) call this "banding" and describe it
causing visible curved stripes across single-pass mosaics like these; they partially correct for
it with a factor derived from simulating SPHEREx's resolution against real absorption-line
profiles, and for H2O specifically find the effect is mild (1-13%, since the H2O band is broad
relative to SPHEREx's resolution there). This tool doesn't apply that correction yet, so an
absorption map's tau is a slight (per that paper, under 13% for H2O) lower bound, and you can
still see some striping even in their own corrected maps. It's a different, smaller effect than
the per-pixel dropouts `--min_weight` addresses above, and different again from real coverage
gaps.

**Peak tau is a lower bound on ice column density, not the whole story.** As in Hora et al. (2026,
Section 3.1), away from a sightline toward a bright background star, the true peak optical depth
can't be recovered exactly, since the continuum you're measuring against is itself partly
foreground emission.

**The default contour level is a starting point, not a universal standard.** 0.5 tau for H2O, 0.2
tau for CO2/CO are commonly used "clearly present" thresholds, but what actually counts as
significant depends on your own field's noise level, per everything above. The emission-feature
default (30% of that epoch's 95th percentile) is even more field-dependent by construction. Look
at the plain map first, and pick `--contour_level` on purpose rather than taking the default on
faith.
