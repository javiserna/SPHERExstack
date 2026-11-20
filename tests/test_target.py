"""Offline tests for SIMBAD target resolution (urlopen is monkeypatched -- no network)."""

from unittest.mock import patch

import pytest

from spherexstack import target


class _FakeResponse:
    def __init__(self, text: str):
        self._data = text.encode("utf-8")

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def test_resolve_target_parses_csv_response():
    csv_text = "main_id,ra,dec\r\nM 31,10.6847,41.269\r\n"
    with patch("urllib.request.urlopen", return_value=_FakeResponse(csv_text)):
        ra, dec, main_id = target.resolve_target("M31")
    assert ra == pytest.approx(10.6847)
    assert dec == pytest.approx(41.269)
    assert main_id == "M 31"


def test_resolve_target_raises_on_no_match():
    csv_text = "main_id,ra,dec\r\n"
    with patch("urllib.request.urlopen", return_value=_FakeResponse(csv_text)):
        with pytest.raises(ValueError, match="no match"):
            target.resolve_target("not_a_real_object_xyz")


def test_resolve_target_raises_clearly_on_html_response():
    html_text = "<!doctype html><html><body>Page unavailable</body></html>"
    with patch("urllib.request.urlopen", return_value=_FakeResponse(html_text)):
        with pytest.raises(RuntimeError, match="unexpected"):
            target.resolve_target("M31")


def test_resolve_target_raises_clearly_on_connection_error():
    import urllib.error

    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("boom")):
        with pytest.raises(RuntimeError, match="Could not reach SIMBAD"):
            target.resolve_target("M31")


def test_resolve_target_escapes_single_quotes_in_name():
    # Build the query the same way resolve_target does, and check the quote was doubled
    # (a bare single quote would break the ADQL string literal).
    captured = {}

    def fake_urlopen(url, timeout=None):
        captured["url"] = url
        return _FakeResponse("main_id,ra,dec\r\nFoo,1.0,2.0\r\n")

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        target.resolve_target("O'Brien's Object")
    assert "''" in captured["url"] or "%27%27" in captured["url"]
