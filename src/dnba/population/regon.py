"""Parser of the GUS quarterly REGON tables into long format.

Tables from the file `tablice_kwartalne_regon_YYYYMMDD.xlsx`:

- Table 4: PKD 2007 × expected number of employees (all of Poland),
- Table 6: voivodeship × PKD 2007 × expected number of employees,
- Table 11 (one sheet per specific legal form): PKD 2007 × expected number of employees.

We read the PKD division rows (two digits). Sections and subclasses are skipped: a section is
the sum of its divisions, and subclasses are too sparse to sample a population from.
"""

from __future__ import annotations

import re
from pathlib import Path

import openpyxl
import pandas as pd

from .dictionaries import SIZE_CLASSES, normalize_voivodeship

NO_PKD = "none"

# Table 6 rows without a voivodeship breakdown (GUS label, lower case) -> control-total key.
_EXTRA_ROWS = {"brak województwa": "no_voivodeship", "brak pkd": "no_pkd"}


def _num(v) -> int:
    """GUS writes zero as '-'."""
    if v is None or (isinstance(v, str) and v.strip() in {"-", ""}):
        return 0
    if isinstance(v, (int, float)):
        return int(v)
    return int(str(v).replace(" ", "").replace("\xa0", ""))


def _sheets(path: Path) -> dict[str, list[tuple]]:
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    out = {ws.title.strip(): list(ws.iter_rows(values_only=True)) for ws in wb.worksheets}
    wb.close()
    return out


def _parse_pkd_size(rows: list[tuple], col0: int) -> tuple[pd.DataFrame, dict[str, int]]:
    """Division rows from a PKD × size class table.

    `col0` is the index of the „Sekcja” column; it is followed by Dział, Podklasa, Ogółem and the four classes.
    Returns (division frame, control totals: total and no PKD).
    """
    recs, totals, section = [], {}, None
    for r in rows[4:]:
        sec, div, sub = r[col0], r[col0 + 1], r[col0 + 2]
        vals = r[col0 + 3 : col0 + 8]
        if sec is not None:
            label = str(sec).strip()
            if "OGÓŁEM" in label.upper():
                totals["total"] = _num(vals[0])
                continue
            if label.lower() == "brak pkd":
                totals["no_pkd"] = _num(vals[0])
                for s, v in zip(SIZE_CLASSES, vals[1:]):
                    recs.append((NO_PKD, NO_PKD, s, _num(v)))
                continue
            section = label
            continue
        if div is not None and sub is None:
            for s, v in zip(SIZE_CLASSES, vals[1:]):
                recs.append((section, str(div).strip().zfill(2), s, _num(v)))
    df = pd.DataFrame(recs, columns=["pkd_section", "pkd_division", "size_class", "n"])
    return df, totals


def parse_forms(path: Path) -> pd.DataFrame:
    """Table 11: form (sfp) × PKD division × size class.

    Columns: legal_form, pkd_section, pkd_division, size_class, n.
    """
    frames = []
    for title, rows in _sheets(path).items():
        m = re.search(r"sfp=(\d{3})", title)
        if not m:
            continue
        df, totals = _parse_pkd_size(rows, col0=0)
        if df["n"].sum() != totals.get("total", -1):
            raise ValueError(f"{title}: sum of divisions {df['n'].sum()} ≠ total {totals.get('total')}")
        df.insert(0, "legal_form", m.group(1))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def parse_pkd_size(path: Path) -> pd.DataFrame:
    """Table 4: PKD division × size class for all of Poland."""
    df, totals = _parse_pkd_size(_sheets(path)["Tabl 4"], col0=0)
    if df["n"].sum() != totals["total"]:
        raise ValueError(f"Table 4: sum {df['n'].sum()} ≠ total {totals['total']}")
    return df


def parse_voivodeships(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Table 6: voivodeship × PKD division × size class.

    Also returns the totals of rows without a PKD breakdown: „Brak województwa” and „Brak PKD” (national).
    """
    rows = _sheets(path)["Tabl 6"]
    recs, extra, voiv, section = [], {}, None, None
    for r in rows[4:]:
        head, sec, div, sub = r[0], r[1], r[2], r[3]
        vals = r[4:9]
        if head is not None:
            label = str(head).strip()
            if "OGÓŁEM" in label.upper():
                extra["total"] = _num(vals[0])
                continue
            norm = normalize_voivodeship(label)
            if norm is None:
                extra[_EXTRA_ROWS.get(label.lower(), label.lower().replace(" ", "_"))] = _num(vals[0])
                voiv = None
                continue
            voiv, section = norm, None
            extra[f"voivodeship:{voiv}"] = _num(vals[0])
            continue
        if voiv is None:
            continue
        if sec is not None:
            section = str(sec).strip()
            continue
        if div is not None and sub is None:
            for s, v in zip(SIZE_CLASSES, vals[1:]):
                recs.append((voiv, section, str(div).strip().zfill(2), s, _num(v)))
    df = pd.DataFrame(recs, columns=["voivodeship", "pkd_section", "pkd_division", "size_class", "n"])
    return df, extra
