"""Layer 1 validation (part of the G0-public gate).

1. Margin recovery: the joint distribution must reproduce the GUS tables it was built from.
2. Sample fidelity: distributions of the weighted sample against the joint distribution (total variation distance).
3. Hold-out test of the assumption: conditional independence of form and voivodeship (A0-3)
   cannot be checked without a form × voivodeship table. So we check the same assumption
   on a pair for which GUS gives the truth: voivodeship ⊥ size class | division.
   We predict Table 6 from Table 4 and the voivodeship × division sums, then compare with Table 6.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .regon import NO_PKD


def tvd(p: pd.Series, q: pd.Series) -> float:
    """Total variation distance of two distributions (input: counts or weights)."""
    idx = p.index.union(q.index)
    p = p.reindex(idx, fill_value=0) / p.sum()
    q = q.reindex(idx, fill_value=0) / q.sum()
    return float(0.5 * np.abs(p - q).sum())


def margin_recovery(joint_all: pd.DataFrame, target: pd.DataFrame, dims: list[str]) -> dict:
    """Maximum absolute and relative error of margin `dims` of the joint distribution against `target`."""
    a = joint_all.groupby(dims)["n"].sum()
    b = target.groupby(dims)["n"].sum().reindex(a.index, fill_value=0)
    diff = (a - b).abs()
    return {"max_abs": float(diff.max()), "sum_abs": float(diff.sum()), "total": float(b.sum())}


def sample_fidelity(firms: pd.DataFrame, joint: pd.DataFrame, dims: list[list[str]]) -> pd.DataFrame:
    rows = []
    for d in dims:
        p = firms.groupby(d)["weight"].sum()
        q = joint.groupby(d)["n"].sum()
        rows.append({"dimension": " × ".join(d), "cells": len(q), "tvd": tvd(p, q)})
    return pd.DataFrame(rows)


def conditional_independence_holdout(pkd_size: pd.DataFrame, voiv: pd.DataFrame) -> dict:
    """Predicts voivodeship × division × class assuming voivodeship ⊥ class | division and compares with Table 6."""
    v = voiv[voiv["pkd_division"] != NO_PKD]
    t4 = pkd_size[pkd_size["pkd_division"] != NO_PKD]
    p_size = t4.groupby(["pkd_division", "size_class"])["n"].sum() / t4.groupby("pkd_division")["n"].sum()
    wd = v.groupby(["voivodeship", "pkd_division"])["n"].sum()
    pred = (wd.rename("wd").reset_index().merge(p_size.rename("p").reset_index(), on="pkd_division"))
    pred["n"] = pred["wd"] * pred["p"]
    actual = v.groupby(["voivodeship", "pkd_division", "size_class"])["n"].sum()
    predicted = pred.set_index(["voivodeship", "pkd_division", "size_class"])["n"]

    full = tvd(predicted, actual)
    # On the voivodeship × class margin, where the assumption error accumulates over divisions.
    a2 = actual.groupby(level=["voivodeship", "size_class"]).sum()
    p2 = predicted.groupby(level=["voivodeship", "size_class"]).sum()
    rel = ((p2 - a2) / a2).rename("rel_error")
    big = rel[a2.reindex(rel.index) > 0].abs().sort_values(ascending=False)
    return {
        "tvd_voivodeship_division_size": full,
        "tvd_voivodeship_size": tvd(p2, a2),
        "max_rel_error_voivodeship_size": float(big.iloc[0]),
        "largest_deviations": [
            {"voivodeship": w, "size_class": k, "predicted": round(float(p2[(w, k)])),
             "actual": int(a2[(w, k)]), "rel_error": round(float(rel[(w, k)]), 3)}
            for (w, k) in big.index[:5]
        ],
    }


def krs_holdout(joint_all: pd.DataFrame, krs_dir, forms: list[str]) -> list[dict]:
    """Hold-out: voivodeship × division distribution within a form, model against K03 from KRS.

    The KRS margin (A0-7) fixes only the voivodeship within a form; the PKD division within a form and its
    relation to the voivodeship come from GUS and the conditional independence assumption. K03 checks this.
    K03 cells removed by the min_cell threshold are skipped in both distributions.
    """
    from pathlib import Path

    k03 = pd.read_csv(Path(krs_dir) / "K03_form_voivodeship_division.csv", dtype=str, keep_default_na=False)
    k03["n"] = k03["n"].astype(int)
    out = []
    for form in forms:
        a = k03[k03["legal_form"] == form].set_index(["voivodeship", "pkd_division"])["n"]
        m = joint_all[joint_all["legal_form"] == form].groupby(["voivodeship", "pkd_division"])["n"].sum()
        common = a.index.intersection(m.index)
        if len(common) == 0:
            continue
        aw, mw = a.groupby(level="voivodeship").sum(), m.groupby(level="voivodeship").sum()
        out.append({
            "legal_form": form,
            "tvd_voivodeship": tvd(mw, aw),
            "tvd_voivodeship_division": tvd(m.reindex(common), a.reindex(common)),
            "k03_share_compared": float(a.reindex(common).sum() / a.sum()),
        })
    return out
