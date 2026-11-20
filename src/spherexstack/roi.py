"""Region-of-interest (ROI) definitions shared by the query, download and mosaic steps.

A ROI is a box on the sky, given either explicitly (``ra_min``/``ra_max``/``dec_min``/``dec_max``)
or as a center plus a square size (``ra_center``/``dec_center``/``size_deg``). All pipeline steps
accept the same ``--roi roi.json`` file so the numbers only have to be resolved once and cannot
drift between steps.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def resolve_roi(
    ra_center: float | None = None,
    dec_center: float | None = None,
    size_deg: float | None = None,
    ra_min: float | None = None,
    ra_max: float | None = None,
    dec_min: float | None = None,
    dec_max: float | None = None,
) -> dict[str, float]:
    """Resolve a ROI from exactly one of two parameterizations.

    Parameters
    ----------
    ra_center, dec_center, size_deg : float, optional
        Center (degrees) and side length (degrees) of a square box. Give this
        group, or the box group below, but not both.
    ra_min, ra_max, dec_min, dec_max : float, optional
        Explicit box edges in degrees.

    Returns
    -------
    dict
        Keys ``ra_min``, ``ra_max``, ``dec_min``, ``dec_max``, ``ra_center``,
        ``dec_center``, ``width_deg``, ``height_deg``. ``width_deg`` is the
        RA extent corrected by ``cos(dec_center)``; ``height_deg`` is the
        uncorrected Dec extent.

    Raises
    ------
    ValueError
        If neither or both parameterizations are fully given, or if the box
        edges are degenerate/inverted.
    """
    box_given = None not in (ra_min, ra_max, dec_min, dec_max)
    center_given = None not in (ra_center, dec_center, size_deg)

    if box_given == center_given:
        raise ValueError(
            "Give either --ra_min/--ra_max/--dec_min/--dec_max (explicit box), "
            "or --ra_center/--dec_center/--size_deg (square box), not both or neither."
        )

    if box_given:
        if ra_min >= ra_max or dec_min >= dec_max:
            raise ValueError("ra_min must be < ra_max and dec_min < dec_max.")
        dec_center = (dec_min + dec_max) / 2.0
        cosd = max(math.cos(math.radians(dec_center)), 1e-6)
        ra_center = (ra_min + ra_max) / 2.0
        width_deg = (ra_max - ra_min) * cosd
        height_deg = dec_max - dec_min
    else:
        half = size_deg / 2.0
        dec_min = dec_center - half
        dec_max = dec_center + half
        cosd = max(math.cos(math.radians(dec_center)), 1e-6)
        ra_half = half / cosd
        ra_min = ra_center - ra_half
        ra_max = ra_center + ra_half
        width_deg = size_deg
        height_deg = size_deg

    return {
        "ra_min": ra_min,
        "ra_max": ra_max,
        "dec_min": dec_min,
        "dec_max": dec_max,
        "ra_center": ra_center,
        "dec_center": dec_center,
        "width_deg": width_deg,
        "height_deg": height_deg,
    }


def save_roi(path: str | Path, roi: dict[str, Any]) -> None:
    """Write a ROI dict to ``path`` as JSON."""
    Path(path).write_text(json.dumps(roi, indent=2))


def load_roi(path: str | Path) -> dict[str, float]:
    """Read a ROI dict previously written by :func:`save_roi`."""
    return json.loads(Path(path).read_text())


def add_roi_cli_args(ap: Any) -> None:
    """Add the ROI arguments (both parameterizations, plus ``--roi``) to an ``ArgumentParser``."""
    ap.add_argument(
        "--roi",
        default=None,
        help="Path to a roi.json saved by spherex-query. If given, overrides "
        "--ra_center/--ra_min/etc.",
    )
    ap.add_argument("--ra_center", type=float, default=None)
    ap.add_argument("--dec_center", type=float, default=None)
    ap.add_argument("--size_deg", type=float, default=None)
    ap.add_argument("--ra_min", type=float, default=None)
    ap.add_argument("--ra_max", type=float, default=None)
    ap.add_argument("--dec_min", type=float, default=None)
    ap.add_argument("--dec_max", type=float, default=None)


def roi_from_args(args: Any) -> dict[str, float]:
    """Build a ROI dict from parsed CLI args added via :func:`add_roi_cli_args`."""
    if args.roi:
        return load_roi(args.roi)
    return resolve_roi(
        ra_center=args.ra_center,
        dec_center=args.dec_center,
        size_deg=args.size_deg,
        ra_min=args.ra_min,
        ra_max=args.ra_max,
        dec_min=args.dec_min,
        dec_max=args.dec_max,
    )
