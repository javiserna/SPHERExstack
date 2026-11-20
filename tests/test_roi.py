import math

import pytest

from spherexstack import roi


def test_resolve_roi_center_mode():
    r = roi.resolve_roi(ra_center=70.0, dec_center=25.0, size_deg=2.0)
    assert r["ra_center"] == pytest.approx(70.0)
    assert r["dec_center"] == pytest.approx(25.0)
    assert r["width_deg"] == pytest.approx(2.0)
    assert r["height_deg"] == pytest.approx(2.0)
    assert r["dec_min"] == pytest.approx(24.0)
    assert r["dec_max"] == pytest.approx(26.0)
    # ra_half is declination-cosine corrected
    cosd = math.cos(math.radians(25.0))
    assert r["ra_max"] - r["ra_center"] == pytest.approx(1.0 / cosd)


def test_resolve_roi_box_mode():
    r = roi.resolve_roi(ra_min=60.0, ra_max=62.0, dec_min=24.0, dec_max=26.0)
    assert r["ra_center"] == pytest.approx(61.0)
    assert r["dec_center"] == pytest.approx(25.0)
    assert r["height_deg"] == pytest.approx(2.0)
    cosd = math.cos(math.radians(25.0))
    assert r["width_deg"] == pytest.approx(2.0 * cosd)


def test_resolve_roi_rejects_both_or_neither():
    with pytest.raises(ValueError):
        roi.resolve_roi()
    with pytest.raises(ValueError):
        roi.resolve_roi(ra_center=1.0, dec_center=1.0, size_deg=1.0, ra_min=0.0, ra_max=2.0, dec_min=0.0, dec_max=2.0)


def test_resolve_roi_rejects_inverted_box():
    with pytest.raises(ValueError):
        roi.resolve_roi(ra_min=10.0, ra_max=5.0, dec_min=0.0, dec_max=1.0)


def test_save_load_roundtrip(tmp_path):
    r = roi.resolve_roi(ra_center=70.0, dec_center=25.0, size_deg=1.5)
    path = tmp_path / "roi.json"
    roi.save_roi(path, r)
    r2 = roi.load_roi(path)
    assert r2 == r


def test_roi_from_args_prefers_roi_file(tmp_path):
    class Args:
        roi = None
        ra_center = 10.0
        dec_center = 0.0
        size_deg = 1.0
        ra_min = ra_max = dec_min = dec_max = None

    r = roi.roi_from_args(Args())
    assert r["ra_center"] == pytest.approx(10.0)

    saved_path = tmp_path / "saved.json"
    roi.save_roi(saved_path, r)

    class ArgsWithFile:
        roi = str(saved_path)
        ra_center = ra_min = ra_max = None
        dec_center = dec_min = dec_max = None
        size_deg = None

    r2 = roi.roi_from_args(ArgsWithFile())
    assert r2 == r
