"""Interventions of hypothesis B-K1 (data/reference/b_k1.json) as code: applicability, the change, the
fields that follow from it, and two checks on every variant: the whole profile is consistent and the
premise the intervention names did change. A profile for which either check fails is not eligible.
"""
from __future__ import annotations

import copy

from .operational import LINKS, consistency, passes_on_for

VERSION = "0.2.0"
SIZE_SWAP = {"0-9": "50-249", "10-49": "0-9", "50-249": "0-9", "250+": "0-9"}
VOIVODESHIP_SWAP = {"mazowieckie": "podkarpackie"}  # any other voivodeship goes to mazowieckie


class NotEligible(Exception):
    pass


def _layer2_with(layer2: dict, change: dict) -> dict:
    new = copy.deepcopy(layer2)
    new.update(change)
    d = new["division_of_work"]  # who passes documents on follows the link to the books
    d["passes_on"] = passes_on_for(new["to_accounting"], d["issues"], d["books"])
    return new


def apply(iid: str, firm: dict, layer2: dict) -> tuple[dict, dict, dict]:
    """(firm, layer2, card options) of the intervened profile; raises NotEligible when the profile does
    not meet the condition, the change would leave it inconsistent, or the named premise is unchanged."""
    f, l2, card = dict(firm), layer2, {}
    if iid == "I1":
        _need(layer2["invoicing_tool"] == "invoicing_program" and layer2["ksef_submission"] == "manual" and layer2["invoices_issued_month"] >= 1)
        l2, named = _layer2_with(layer2, {"ksef_submission": "automatic"}), ["ksef_submission"]
    elif iid == "I2":
        _need(layer2["invoicing_tool"] == "invoicing_program" and layer2["payment_reconciliation"] == "manual" and layer2["invoices_issued_month"] >= 1)
        l2, named = _layer2_with(layer2, {"payment_reconciliation": "automatic"}), ["payment_reconciliation"]
    elif iid == "I3":  # at least one invoice: 0 x 10 changes nothing
        _need(layer2["invoicing_tool"] == "none" and 1 <= layer2["invoices_issued_month"] <= 30)
        l2, named = _layer2_with(layer2, {"invoices_issued_month": 10 * layer2["invoices_issued_month"]}), ["invoices_issued_month"]
    elif iid == "I4":
        b2c = 0.9 if layer2["b2c_share_of_sales"] < 0.5 else 0.0
        l2, named = _layer2_with(layer2, {"b2c_share_of_sales": b2c}), ["b2c_share_of_sales"]
    elif iid == "I5":  # composite: no tool, so every link manual; passes_on follows
        _need(layer2["invoicing_tool"] == "erp" and 1 <= layer2["invoices_issued_month"] <= 30)
        # only what dropping the tool forces: automatic links and KSeF by files become manual; a file
        # exchange with the books or an office taking invoices from KSeF stays as it was
        forced = {k: "manual" for k in LINKS if layer2[k] == "automatic" or (k == "ksef_submission" and layer2[k] == "files")}
        l2 = _layer2_with(layer2, {"invoicing_tool": "none", **forced})
        named = ["invoicing_tool"]
    elif iid == "I6":
        f["size_class"], named = SIZE_SWAP[firm["size_class"]], ["size_class"]
    elif iid == "P1":
        card, named = {"name": "Kwadrel"}, ["name"]  # "kw" is in no syllable: no firm of the sample has this name
    elif iid == "P2":
        f["voivodeship"] = VOIVODESHIP_SWAP.get(firm["voivodeship"], "mazowieckie")
        named = ["voivodeship"]
    elif iid == "P3":
        card, named = {"order_seed": 7}, ["order"]
    else:
        raise ValueError(f"unknown intervention {iid}")
    problems = consistency(l2)
    if problems:
        raise NotEligible(f"{iid}: inconsistent after the change: {problems}")
    changed = [k for k in named if k in ("name", "order") or (l2.get(k, f.get(k)) != layer2.get(k, firm.get(k)))]
    if changed != named:
        raise NotEligible(f"{iid}: the named premise did not change: {sorted(set(named) - set(changed))}")
    return f, l2, card


def _need(cond: bool) -> None:
    if not cond:
        raise NotEligible("the profile does not meet the condition")
