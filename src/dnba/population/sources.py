"""Registry of data sources with checksums."""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Source:
    key: str
    url: str
    filename: str
    sha256: str
    as_of: str
    description: str


SOURCES = {
    s.key: s
    for s in (
        Source(
            key="regon_2026q2",
            url=(
                "https://stat.gov.pl/download/gfx/portalinformacyjny/pl/defaultaktualnosci/"
                "5504/7/16/1/tablice_kwartalne_regon_20260630.xlsx"
            ),
            filename="tablice_kwartalne_regon_20260630.xlsx",
            sha256="858fd5715b1b4d64180d4bd15310a74b8b8d45d26ce3a79182e3dff7f8a6e6e9",
            as_of="2026-06-30",
            description=(
                "GUS, Kwartalna informacja o podmiotach gospodarki narodowej w rejestrze REGON "
                "(quarterly REGON report), as of 30.06.2026; entities declaring business activity, PKD 2007"
            ),
        ),
    )
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(key: str, raw_dir: Path) -> Path:
    """Returns the path to the source file; downloads it if missing. Always verifies the checksum."""
    src = SOURCES[key]
    path = raw_dir / src.filename
    if not path.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(src.url, path)
    digest = sha256(path)
    if digest != src.sha256:
        raise ValueError(f"{path}: sha256 {digest} ≠ expected {src.sha256}")
    return path
