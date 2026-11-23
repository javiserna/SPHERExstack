"""Group SPHEREx L2 exposures into observation epochs, by ``MJD-OBS``.

SPHEREx completes one full 0.75-5 um spectrum of a given field over many
separate exposures taken as its linear variable filter (LVF) scans across
the detector on successive orbits -- in practice roughly 15-20 days to
finish one pass, after which the field is revisited (and the pass repeats)
only much later in the survey. To build a separate per-pixel spectrum (and
therefore a separate ice optical depth measurement) for each such visit, the
exposures covering a field must first be split into these "epochs": clusters
of exposures taken close together in time, separated by much longer gaps
where the field was not being observed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from astropy.io import fits


@dataclass
class Epoch:
    """One cluster of exposures taken within a single observation window."""

    index: int
    files: list[Path]
    mjd_min: float
    mjd_max: float

    @property
    def span_days(self) -> float:
        return self.mjd_max - self.mjd_min


def read_mjd_obs(fits_path: Path) -> float | None:
    """``MJD-OBS`` of one exposure (from the ``IMAGE`` HDU header), or None if absent."""
    with fits.open(fits_path, memmap=True) as hdul:
        for ext in ("IMAGE", 0):
            if ext in hdul and "MJD-OBS" in hdul[ext].header:
                return float(hdul[ext].header["MJD-OBS"])
    return None


def group_into_epochs(
    files: list[Path],
    max_gap_days: float = 30.0,
    max_span_days: float = 45.0,
) -> list[Epoch]:
    """Sort ``files`` by ``MJD-OBS`` and split them into epochs.

    A new epoch starts whenever either:

    * the gap since the previous (in time) exposure exceeds ``max_gap_days``
      (the field was not observed for a while -- a new visit), or
    * including the next exposure would stretch the current epoch's span
      (from its first exposure) beyond ``max_span_days`` (a safety cap).

    The defaults were tuned against a real test field: one visit ("epoch")
    is not necessarily a single uninterrupted run -- it can have its own
    internal gaps of a couple of weeks (scheduling), so too tight a
    ``max_gap_days`` (an earlier default of 5 days) over-splits a single
    visit into several spurious epochs. A true revisit of the field, by
    contrast, was separated by well over 100 days in that test. 30/45 days
    sits comfortably between those two scales, but every field's actual
    cadence can differ -- pass ``--max_gap_days``/``--max_span_days``
    explicitly and check the resulting epoch list (e.g. via
    ``spherex-ice-epochs``) if the grouping looks wrong for your field.

    Files without a readable ``MJD-OBS`` are skipped (with no error raised
    here; the caller may want to warn).
    """
    tagged: list[tuple[float, Path]] = []
    for fp in files:
        mjd = read_mjd_obs(fp)
        if mjd is not None:
            tagged.append((mjd, fp))
    tagged.sort(key=lambda t: t[0])

    groups: list[list[tuple[float, Path]]] = []
    current: list[tuple[float, Path]] = []
    for mjd, fp in tagged:
        if current:
            gap = mjd - current[-1][0]
            span_if_added = mjd - current[0][0]
            if gap > max_gap_days or span_if_added > max_span_days:
                groups.append(current)
                current = []
        current.append((mjd, fp))
    if current:
        groups.append(current)

    epochs = []
    for i, grp in enumerate(groups):
        mjds = [m for m, _ in grp]
        epochs.append(
            Epoch(index=i, files=[fp for _, fp in grp], mjd_min=min(mjds), mjd_max=max(mjds))
        )
    return epochs
