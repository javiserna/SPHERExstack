import numpy as np
import pytest
from PIL import Image

from spherexstack import animate, mosaic


def test_format_fov_picks_degrees_for_large_fields():
    assert animate._format_fov(2.0, 1.5) == "FOV 2.00° x 1.50°"


def test_format_fov_picks_arcmin_for_small_fields():
    assert animate._format_fov(0.5, 0.5) == "FOV 30.0' x 30.0'"


def test_format_fov_picks_arcsec_for_tiny_fields():
    assert animate._format_fov(0.001, 0.001) == 'FOV 4" x 4"'


def test_fov_deg_from_wcs_matches_reference_grid(tmp_path):
    pixscale = 6.15
    nx, ny = 24, 20
    width_deg = nx * pixscale / 3600.0
    height_deg = ny * pixscale / 3600.0
    ref_wcs, ref_shape = mosaic.build_reference_wcs(10.0, 0.0, width_deg, height_deg, pixscale)

    fov_w, fov_h = animate.fov_deg_from_wcs(ref_wcs, ref_shape)
    assert fov_w == pytest.approx(width_deg, rel=1e-6)
    assert fov_h == pytest.approx(height_deg, rel=1e-6)


def test_render_rgb_composite_writes_a_correctly_sized_image(tmp_path):
    rng = np.random.default_rng(0)
    shape = (40, 50)
    mosaics = [rng.uniform(1.0, 10.0, size=shape) for _ in range(3)]
    # Punch a NaN hole in each channel to exercise the no-coverage-renders-black path.
    for m in mosaics:
        m[0:5, 0:5] = np.nan

    out_path = tmp_path / "rgb.png"
    animate.render_rgb_composite(
        mosaics[0], mosaics[1], mosaics[2], out_path, wavelengths_um=(4.5, 3.0, 1.2), fov_deg=(0.5, 0.4)
    )

    assert out_path.exists()
    with Image.open(out_path) as im:
        assert im.size == (832, 832)
        assert im.mode in ("RGB", "RGBA")
