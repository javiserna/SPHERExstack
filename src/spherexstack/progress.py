"""A small, dependency-free terminal progress bar with an ETA.

Downloading cutouts is the one step in the pipeline whose wall-clock time is
genuinely hard to predict up front (it depends on the number of exposures,
which grows with the field size, and on IRSA's own response time), so it is
the one place a live progress indicator earns its keep. This intentionally
does not pull in a dependency like ``tqdm`` for one progress bar.
"""

from __future__ import annotations

import sys
import time
from typing import TextIO


def format_duration(seconds: float | None) -> str:
    """``"1:02:03"`` or ``"02:03"`` style duration, or ``"--:--"`` if unknown."""
    if seconds is None or seconds < 0:
        return "--:--"
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


class ProgressBar:
    """A single, in-place-updating line: ``|####----| 12/40 (30%)  2.1 files/s  ETA 00:14``.

    Meant to be driven one completed item at a time (e.g. from
    ``concurrent.futures.as_completed``), not as a general-purpose
    multi-stage progress tool.
    """

    def __init__(
        self,
        total: int,
        prefix: str = "Downloading",
        unit: str = "files",
        bar_width: int = 28,
        stream: TextIO | None = None,
    ):
        self.total = max(total, 0)
        self.prefix = prefix
        self.unit = unit
        self.bar_width = bar_width
        self.stream = stream if stream is not None else sys.stdout
        self.done = 0
        self.start_time = time.monotonic()
        self._last_len = 0
        self._closed = False

    def interrupt(self, message: str) -> None:
        """Print ``message`` on its own line without leaving a corrupted bar behind it."""
        self.stream.write("\r" + " " * self._last_len + "\r")
        self.stream.write(message + "\n")
        self._render()

    def update(self, done: int | None = None) -> None:
        """Advance to ``done`` completed items, or by one if ``done`` is omitted."""
        self.done = done if done is not None else self.done + 1
        self._render()

    def _render(self) -> None:
        if self.total <= 0 or self._closed:
            return
        frac = min(1.0, self.done / self.total)
        filled = int(round(self.bar_width * frac))
        bar = "#" * filled + "-" * (self.bar_width - filled)
        elapsed = time.monotonic() - self.start_time
        rate = self.done / elapsed if elapsed > 0 else 0.0
        eta = (self.total - self.done) / rate if rate > 0 else None
        line = (
            f"[INFO] {self.prefix}: |{bar}| {self.done}/{self.total} ({100 * frac:.0f}%)"
            f"  {rate:.1f} {self.unit}/s  ETA {format_duration(eta)}"
        )
        pad = max(0, self._last_len - len(line))
        self.stream.write("\r" + line + " " * pad)
        self.stream.flush()
        self._last_len = len(line)

    def close(self) -> None:
        """Finish the bar and move to a fresh line. Safe to call at most once."""
        if self._closed:
            return
        self._closed = True
        self.stream.write("\n")
        self.stream.flush()
