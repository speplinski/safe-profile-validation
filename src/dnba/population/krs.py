"""Voivodeship distribution by legal form from the KRS aggregates (scripts/krs_aggregate.py).

Used as the third, partial margin in IPF (fix of assumption A0-3 for KRS forms).
Counts come from REGON; from KRS only the voivodeship share within a form.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .dictionaries import BUSINESS_FORMS, VOIVODESHIP_BY_NAME

INACTIVE = {"liquidation", "bankruptcy", "dissolution", "suspended"}


def form_voivodeship_shares(krs_dir: Path, min_coverage: float = 0.95, min_entities: int = 1000) -> pd.DataFrame:
    """Voivodeship share among the active entities of a form in register P.

    The denominator is all active entities of the form (K01), so cells removed by the
    min_cell threshold do not lower the other shares; missing cells are returned as NaN (unknown).
    Forms with K14 coverage below `min_coverage` or with fewer than `min_entities` entities
    are skipped. Columns: legal_form, voivodeship, share (NaN = unknown), coverage, n_krs.
    """
    k01 = pd.read_csv(krs_dir / "K01_register_form_status.csv", dtype=str)
    k01["n"] = k01["n"].astype(int)
    active = k01[(k01["register"] == "P") & ~k01["status_section6"].isin(INACTIVE)].groupby("legal_form")["n"].sum()

    k14 = pd.read_csv(krs_dir / "K14_form_seat.csv", dtype=str, keep_default_na=False)
    k14["n"] = k14["n"].astype(int)
    k14 = k14[k14["voivodeship"].isin(VOIVODESHIP_BY_NAME)]
    by = k14.groupby(["legal_form", "voivodeship"])["n"].sum()

    rows = []
    for form in sorted(set(BUSINESS_FORMS) & set(active.index)):
        if BUSINESS_FORMS[form].register != "KRS" or form not in by.index.get_level_values(0):
            continue
        cells = by.loc[form]
        coverage = cells.sum() / active[form]
        if coverage < min_coverage or active[form] < min_entities:
            continue
        for voiv in VOIVODESHIP_BY_NAME:
            n = cells.get(voiv)
            rows.append({"legal_form": form, "voivodeship": voiv,
                         "share": n / active[form] if n is not None else float("nan"),
                         "coverage": coverage, "n_krs": int(active[form])})
    return pd.DataFrame(rows)
