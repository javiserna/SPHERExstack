from pathlib import Path

from spherexstack import download


def _roi():
    return {"ra_center": 10.0, "dec_center": 0.0, "width_deg": 1.0, "height_deg": 1.0}


def test_run_batch_reports_ok_skip_and_failed(tmp_path, monkeypatch):
    rows = [{"uri": "a.fits"}, {"uri": "b.fits"}, {"uri": "c.fits"}]

    def fake_download_one(row, ra_center, dec_center, size_deg, outdir, keep_hdus):
        name = row["uri"]
        if name == "a.fits":
            return outdir / name, "ok", 100
        if name == "b.fits":
            return outdir / name, "skip", 50
        return outdir / name, "FAIL: timed out", 0

    monkeypatch.setattr(download, "download_one", fake_download_one)
    bar = download.ProgressBar(total=len(rows), prefix="t", unit="files")
    ok, skipped, failed, total_bytes = download._run_batch(
        rows, _roi(), 1.0, tmp_path, (), workers=2, bar=bar
    )
    bar.close()

    assert [p.name for p in ok] == ["a.fits"]
    assert [p.name for p in skipped] == ["b.fits"]
    assert len(failed) == 1
    assert failed[0][0]["uri"] == "c.fits"
    assert total_bytes == 150


def test_main_retries_missing_files_before_giving_up(tmp_path, monkeypatch, capsys):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("uri\nflaky.fits\nsteady.fits\n")
    outdir = tmp_path / "cutouts"
    list_out = tmp_path / "list.txt"

    # "flaky.fits" fails the first pass (simulating a timeout/dropped connection), then
    # succeeds on the first automatic retry pass; "steady.fits" succeeds immediately.
    attempts: dict[str, int] = {}

    def fake_download_one(row, ra_center, dec_center, size_deg, outdir, keep_hdus):
        name = row["uri"]
        out_path = Path(outdir) / name
        attempts[name] = attempts.get(name, 0) + 1
        if name == "flaky.fits" and attempts[name] == 1:
            return out_path, "FAIL: timed out", 0
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"\x00" * 10)
        return out_path, "ok", 10

    monkeypatch.setattr(download, "download_one", fake_download_one)
    download.main(
        [
            "--manifest", str(manifest),
            "--ra_center", "10.0", "--dec_center", "0.0", "--size_deg", "1.0",
            "--outdir", str(outdir),
            "--workers", "1",
            "--list_out", str(list_out),
        ]
    )

    out = capsys.readouterr().out
    assert attempts["flaky.fits"] == 2  # failed once, then succeeded on retry pass 1
    assert "retry pass 1/2" in out
    assert "fail=0" in out
    assert sorted(p.name for p in outdir.glob("*.fits")) == ["flaky.fits", "steady.fits"]


def test_main_continues_with_partial_results_when_retries_exhausted(tmp_path, monkeypatch, capsys):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("uri\nalways_fails.fits\nok.fits\n")
    outdir = tmp_path / "cutouts"
    list_out = tmp_path / "list.txt"

    def fake_download_one(row, ra_center, dec_center, size_deg, outdir, keep_hdus):
        name = row["uri"]
        out_path = Path(outdir) / name
        if name == "always_fails.fits":
            return out_path, "FAIL: timed out", 0
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"\x00" * 10)
        return out_path, "ok", 10

    monkeypatch.setattr(download, "download_one", fake_download_one)
    # Should not raise despite one file never succeeding -- it must finish and write
    # list_out with whatever did download.
    download.main(
        [
            "--manifest", str(manifest),
            "--ra_center", "10.0", "--dec_center", "0.0", "--size_deg", "1.0",
            "--outdir", str(outdir),
            "--workers", "1",
            "--retry-passes", "1",
            "--list_out", str(list_out),
        ]
    )

    out = capsys.readouterr().out
    assert "fail=1" in out
    assert "still missing after 2 total pass(es)" in out
    assert list_out.read_text().strip() == str(outdir / "ok.fits")
