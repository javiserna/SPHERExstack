"""Flux-unit conversion for mosaic/cube outputs.

SPHEREx Level 2 data, and therefore every mosaic or cube this package builds from it,
is natively in MJy/sr (surface brightness). IRSA's own SPHEREx mosaic-cube tutorial
(https://caltech-ipac.github.io/irsa-tutorials/spherex-mosaic/) converts this to a
per-pixel flux density in mJy before doing aperture photometry on it. ``--units mjy``
on ``spherex-mosaic``/``spherex-cube`` follows that same conversion, so results line
up with anything built from that tutorial's recipe.
"""

from __future__ import annotations

import numpy as np
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

UNIT_CHOICES = ("mjy_sr", "mjy")

# MJy/sr -> mJy/pixel, given the pixel scale in arcsec/px:
#   mJy = (MJy/sr) * 23.5045 * (pixel scale in arcsec)^2 / 1000
# matching IRSA's SPHEREx mosaic-cube tutorial exactly.
_MJY_SR_ARCSEC2_TO_MJY = 23.5045


def pixscale_arcsec_from_wcs(wcs: WCS) -> float:
    """Pixel scale (arcsec/px) of a 2D celestial WCS, averaged over both axes."""
    scales_deg = proj_plane_pixel_scales(wcs.celestial if wcs.has_celestial else wcs)
    return float(np.mean(scales_deg) * 3600.0)


def convert_units(data: np.ndarray, units: str, wcs: WCS) -> tuple[np.ndarray, str]:
    """Convert ``data`` (native MJy/sr) to ``units``. Returns ``(converted, bunit)``.

    ``units="mjy_sr"`` is a no-op (native surface brightness). ``units="mjy"``
    converts to a per-pixel flux density using the pixel scale read off ``wcs``.
    """
    if units == "mjy_sr":
        return data, "MJy/sr"
    if units == "mjy":
        pixscale_arcsec = pixscale_arcsec_from_wcs(wcs)
        converted = data * _MJY_SR_ARCSEC2_TO_MJY * (pixscale_arcsec**2) / 1000.0
        return converted, "mJy"
    raise ValueError(f"Unknown units {units!r}, expected one of {UNIT_CHOICES}")
