"""Resolve a sky target by name via SIMBAD, for users who don't have RA/Dec on hand.

Used by ``spherex-get`` as an alternative to passing ``--ra``/``--dec``
directly. Talks to SIMBAD's own TAP service with a plain ADQL query (the same
pattern :mod:`spherexstack.query` uses for IRSA), so no extra dependency is
needed beyond the standard library.
"""

from __future__ import annotations

import csv
import urllib.error
import urllib.parse
import urllib.request

SIMBAD_TAP_URL = "https://simbad.cds.unistra.fr/simbad/sim-tap/sync"

# ident.id holds every name/alias SIMBAD knows for an object (not just its main
# identifier), so this also resolves common names like "Andromeda Galaxy", not
# only catalog designations like "M31" or "NGC 224".
SIMBAD_QUERY_TEMPLATE = (
    "SELECT basic.main_id AS main_id, basic.ra AS ra, basic.dec AS dec "
    "FROM basic JOIN ident ON basic.oid = ident.oidref "
    "WHERE ident.id = '{name}'"
)


def resolve_target(name: str, timeout: float = 30.0) -> tuple[float, float, str]:
    """Resolve an object name/identifier to ICRS coordinates via SIMBAD.

    Parameters
    ----------
    name : str
        Any identifier SIMBAD recognizes: a common name ("Andromeda Galaxy"),
        a catalog designation ("M31", "NGC 224"), etc.

    Returns
    -------
    ra_deg, dec_deg : float
        ICRS coordinates, in degrees.
    main_id : str
        SIMBAD's canonical identifier for the resolved object, for display/
        confirmation (the name you typed may be an alias of this).

    Raises
    ------
    ValueError
        If SIMBAD has no match for ``name``.
    RuntimeError
        If the SIMBAD service cannot be reached or returns an unexpected
        (non-CSV) response.
    """
    escaped = name.replace("'", "''")
    query = SIMBAD_QUERY_TEMPLATE.format(name=escaped)
    params = urllib.parse.urlencode({"request": "doQuery", "lang": "adql", "format": "csv", "query": query})
    url = f"{SIMBAD_TAP_URL}?{params}"

    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - fixed, trusted SIMBAD host
            text = resp.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(
            f"Could not reach SIMBAD to resolve '{name}': {exc}. "
            "Use --ra/--dec instead if SIMBAD is unreachable."
        ) from exc

    if text.lstrip().startswith("<"):
        # An HTML page (error/maintenance) instead of CSV means the service
        # is down or misbehaving, not that the target has no match.
        raise RuntimeError(
            f"SIMBAD returned an unexpected (non-CSV) response while resolving '{name}' -- "
            "the service may be temporarily down. Use --ra/--dec instead."
        )

    rows = list(csv.DictReader(text.splitlines()))
    if not rows:
        raise ValueError(f"SIMBAD has no match for '{name}'. Check the spelling, or use --ra/--dec directly.")

    row = rows[0]
    return float(row["ra"]), float(row["dec"]), row["main_id"]
