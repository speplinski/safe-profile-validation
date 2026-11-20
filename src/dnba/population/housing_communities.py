"""Housing communities segment, generator prototype 0.1 (profile specification 0.5,
data/reference/housing_community_profile_draft.json, with the project lead's decisions of 15.11.2026). No LLM.

A frozen synthetic population per voivodeship: records with a weight = the voivodeship's frame count / its records.
Every random number is drawn once from a named stream (numpy default_rng of (seed, a stable code of the voivodeship,
a stable code of the stream)), so scenarios change thresholds and parameters, never the draws. Every share of the
specification is realised as an exact count within its group and voivodeship by ranking a fixed uniform (the k records
with the smallest uniform), so a larger share selects a superset of a smaller one; categories (age, renovation, board
size) are cut by rank in the same way. The size is drawn by stratified quantiles (the rank of the frozen uniform), so
the sample mean carries no sampling noise, and the lognormal's scale is adjusted so that the discretised sizes (at
least 2) add up to the D dwellings of the voivodeship; a residual of a few dwellings goes, one at a time, to the
largest communities of at least 5 units, so the reconciliation never changes the legal size regime (review 0.1).

D inputs: the frame (186,682 communities, M-01 2024; registration variant 198,665, REGON 30.06.2026), the IBS register
shares by voivodeship (31.03.2025), BDL 148151 dwellings by voivodeship (2024), the natural persons' dwellings
(3,873,193; the other 457,200 belong to municipalities and other entities), the mean floor area (52.0 m2, all owners),
the M-01 2024 cost rates and the arrears (660,089 dwellings, 1,263,854.9 thousand PLN), the legal rules of the Act
(Dz.U. 2026 poz. 232). The ENBLOC age x size crosstab is resident-weighted, carried to communities as Z. Every other
number is Z: named in PARAMS (from the specification) or PROTOTYPE_Z (needed by the prototype, not yet in the
specification, listed for the project lead's decision).
"""
from __future__ import annotations

import json
import math
import zlib
from pathlib import Path
from statistics import NormalDist

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "data/reference/housing_community_profile_draft.json"
IBS = ROOT / "data/reference/sources/2026-11-14-ibs-enbloc/agg_ibs_wspolnoty.json"
ENBLOC = ROOT / "data/reference/sources/2026-11-14-ibs-enbloc/agg_enbloc_pl.json"
VERSION = "0.3"
SPEC_VERSION = "0.5.3"
PROFILE_DATE = "2026-11-14"
ND = NormalDist()

FRAME = {"central": 186682, "registration": 198665}            # D: M-01 2024 frame; REGON form 085, 30.06.2026
DWELLINGS_ALL = 4330393                                          # D: BDL 148151, 31.12.2024
DWELLINGS_NATURAL = 3873193                                      # D: BDL 148150
NON_NATURAL_DWELLINGS = DWELLINGS_ALL - DWELLINGS_NATURAL        # 457,200: municipalities and other entities
ARREARS_DWELLINGS = 660089                                       # D: Table 8, natural persons' dwellings
ARREARS_PLN = 1263854.9e3                                        # D: Table 7
MEAN_AREA_ALL = 52.0                                             # D: 148148 / 148151 (rounded in the source)
OPERATIONAL_PER_M2 = 48.74 / 12                                  # D: P3958 2024, PLN per m2 a month
MANAGEMENT_SHARE = 0.2857                                        # D: 1613160 / 1613159, 2024
SERVICES_PER_M2 = 6.09                                           # D (derived): services incl. heating, PLN per m2 a month
AGE = ("before 1945", "1945-1969", "1970-1989", "1990-2009", "2010+")
SIZE_ROWS = ((3, 5, "2 = 3-5"), (6, 9, "3 = 6-9"), (10, 19, "4 = 10-19"), (20, 49, "5 = 20-49"), (50, 10 ** 9, "6 = 50+"))

PARAMS = {  # Z of the specification 0.5 (central values; scenarios override)
    "frame": "central",
    "size_sigma": 1.1, "size_regional_mean": True,
    "municipal_presence": 0.30, "municipal_up_to_3": False, "municipal_pre1945_weight": 2.0,
    "municipal_share_beta": (0.9311, 2.3478, 0.017, 0.923),
    "entrusted_above_3": 0.30,
    "professional_board_member": 0.15,
    "administrator_private_owners_board": 0.647, "administrator_municipal_owners_board": 0.949,
    "administrator_civil_code": 0.30,
    "firm_municipal_company_given_municipal": 0.53, "firm_natural_person_given_private": 0.15,
    "own_account_above_3": 0.95, "own_account_up_to_3": 0.50,
    "operator_weights": {"board_member": 0.4, "owner_operator": 0.4, "manager_or_administrator": 0.5,
                         "accounting_office": 0.1},
    "accounting_office_without_firm": 0.25,
    "fund_above_3": 0.85, "fund_up_to_3": 0.40,
    "fund_rate_truncnorm": (2.8069, 1.1578, 0.70, 5.40), "fund_rate_fixed": None, "fund_months": (6, 36),
    "renovation_ongoing": 0.08, "renovation_planned": 0.12, "renovation_recent": 0.16, "loan_given_ongoing_or_recent": 0.30,
    "separate_fund_account": 0.40,
    "small_fixed_management_pln": 150.0, "small_area_factor": 1.0, "small_services_through_community": 0.5,
    "cash_buffer_months": (0.5, 2.0),
    "under_set_advances": 0.0, "under_set_share_of_municipal_company": 0.5, "under_set_financing": "fund",
    "buildings_poisson": False,
    "board_size_probs": (0.5, 0.2, 0.3),  # owners-only board of 1 / 2 / 3 members (working Z, decision of 15.11.2026)
}

PROTOTYPE_Z = {  # needed by the prototype, not yet values of the specification: for the project lead's decision
    "units_nonresidential": "0 in every community (the specification's central simplification)",
    "minimum_units": "2 units in total (the specification), dwellings drawn >= 2",
    "dwelling_area": "the community's MEAN dwelling area (not each unit's area) lognormal with mean 52.0 m2 (D, the mean "
                     "of the source) and sigma 0.25 (Z); the spread is not validated by the matching mean",
    "separation": "the municipality's units are not separated (the former owner keeps them), all others separated",
    "owners": "owner PARTIES, not natural persons: one party per natural persons' unit (co-owners of a unit as one party), "
              "one party for all units of other entities in the community, one for the municipality; a technical "
              "simplification, not data for representation or voting",
    "other_entities": "units of other entities (not natural persons, not the municipality): per community floor(non-municipal "
                      "units x r + u), r set so that the weighted sum is 457,200 - the municipal dwellings (the D total of "
                      "non-natural-person dwellings); their placement among communities is Z",
    "owner_settlement_accounts": "one per unit",
    "board_size": "owners-only board: 1 (0.5), 2 (0.2), 3 (0.3) members; with a professional member: 1 (the professional)",
    "operators": "an owner operating the account (not a board member) is a candidate where no board exists (weight 0.4); "
                 "an accounting office is used by 0.25 of the communities without a serving firm (weight 0.1); operating "
                 "the account decides nothing about representation or permissions",
    "municipal_area": "the municipality's units have the community's mean dwelling area, so the realised property share "
                      "is count / units; the Beta draw is auxiliary",
    "fund_unpaid_invoices": "no unpaid repair invoices at the opening (a simplification; the identity carries the term)",
    "cash_without_account": "without an own account the community's money is held outside a community account (by the "
                            "co-owners or the administrator; the legal questions stay open): funds_on_accounts = 0",
    "arrears": "dwellings in arrears per community = floor(natural persons' dwellings x q x g + u), g lognormal (sigma 0.8, "
               "mean 1, Z), q set so that the weighted count is the D 660,089 in its own scope (natural persons' dwellings); "
               "amount = count x A, A lognormal (sigma 0.6, Z), scaled to the D amount; the spreads are a working "
               "allocation of the known totals, not confirmed by them; arrears of the municipality and other entities are "
               "not generated; the fund's part of an owner's arrears in proportion to the assessed fund contributions",
    "costs_above_7": "operational = area x 4.0617, services = area x 6.09 (the M-01 means of that scope, all settled "
                     "through the community)",
    "flows": "a steady month: new arrears = arrears repaid, so payments received = assessed; payments made = costs; "
             "no repair payments in the month; no loan amounts (flag only); no owners' overpayments",
    "liabilities": "unpaid supplier invoices = 0.5 x the month's costs (operational + services) + any deficit left unpaid "
                   "in the under-set scenario",
    "renovation_recent": "a major renovation finished within the last 12 months: 0.16 of communities, Z (a state at the "
                         "profile date and an event within a year have different reference periods: the value is NOT "
                         "derived from the 24.8% of Olsztyn); ongoing / recent / planned are mutually exclusive "
                         "(a simplification); renovations do not depend on having a fund",
    "under_set_fund_fallback": "in the fund-financed scenario a community without a fund, or with less fund cash than the "
                               "deficit, leaves the deficit unpaid (invoices); the check counts these communities",
    "private_units": "units not of the municipality = natural persons' units + other entities' units (split in 0.3)",
    "operating_ratio_floor": "with the buffer >= 0.5 month and liabilities = 0.5 month of costs the operating current ratio is "
                             ">= 1 by construction in the central scenario, so the TNN cases 0.6-0.9 cannot occur there",
    "repair_spending_since_last_major": "0: the accumulated months stand for the net accumulation",
    "age_class": "ENBLOC weighted row shares by size class (resident-weighted) used as community shares, Z, not a measured "
                 "age distribution of communities; 2 dwellings take the 3-5 row; with several buildings it is the age of a "
                 "representative building",
    "costs_above_7": "uniform M-01 rates for every community and all utilities settled through the community: Z on D "
                     "rates; per-community values are not observations of M-01",
    "board_size_note": "conclusions on multi-person authorisation depend directly on the board-size Z: an agent task on it "
                       "needs a variant with another share of one-person boards, the rest fixed",
    "steady_month_note": "a reference scenario for technical checks: it does not reproduce a renovation or growing payment "
                         "problems; a loan flag without an amount is not zero debt (current ratio not computed)",
}

STREAMS = ("u_size", "u_area", "u_age", "u_municipal", "u_mshare", "u_basis", "u_board", "u_board_size", "u_admin",
           "u_firm", "u_account", "u_operator", "u_fund", "u_rate", "u_months", "u_renov", "u_loan", "u_sepacc",
           "u_small_services", "u_buffer", "u_under", "z_arrears", "u_arrears", "z_amount", "u_office", "u_buildings",
           "u_other")


def _code(name: str) -> int:
    return zlib.crc32(name.encode())


def _ppf(u: np.ndarray) -> np.ndarray:
    return np.array([ND.inv_cdf(min(max(x, 1e-12), 1 - 1e-12)) for x in np.ravel(u)]).reshape(np.shape(u))


def _cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1 + np.vectorize(math.erf)(np.asarray(z, dtype=float) / math.sqrt(2)))


def voivodeships() -> list[dict]:
    """D: IBS register records and BDL 148151 dwellings by voivodeship."""
    rows = json.loads(IBS.read_text())["by_voivodeship"]
    return [{"name": r["voivodeship"], "records_ibs": r["records"],
             "dwellings": r["bdl_148151_dwellings_in_community_buildings_2024"]} for r in rows]


def age_rows() -> dict:
    ct = json.loads(ENBLOC.read_text())["groups"]["wspolnoty"]["crosstabs (rows = dwellings)"]["dwellings x age"]
    out = {}
    for _, _, key in SIZE_ROWS:
        cells = ct[key]
        out[key] = [cells[f"{i + 1} = {a}"]["row_w_pct"] / 100 for i, a in enumerate(AGE)]
    return out


def allocation(records: int, frame: str) -> list[dict]:
    """Frame counts by voivodeship (IBS shares applied to the frame, Z) and records per voivodeship: `records` is the
    target total, each voivodeship gets at least 200, so the actual total can be larger (reported by the check)."""
    vs = voivodeships()
    tot = sum(v["records_ibs"] for v in vs)
    F = FRAME[frame]
    raw = [F * v["records_ibs"] / tot for v in vs]
    counts = [int(math.floor(x)) for x in raw]
    for i in sorted(range(len(vs)), key=lambda i: (-(raw[i] - counts[i]), i))[:F - sum(counts)]:
        counts[i] += 1  # largest remainders: the counts add up to the frame
    out = []
    for v, c in zip(vs, counts):
        r = max(200, int(round(records * c / F)))
        out.append({**v, "frame": c, "records": r, "weight": c / r})
    return out


def draws(seed: int, alloc: list[dict]) -> dict:
    out = {"__meta__": {"seed": seed, "records": {v["name"]: v["records"] for v in alloc}}}
    for v in alloc:
        out[v["name"]] = {}
        for st in STREAMS:
            rng = np.random.default_rng([seed, _code(v["name"]), _code(st)])
            out[v["name"]][st] = rng.standard_normal(v["records"]) if st[0] == "z" else rng.random(v["records"])
    return out


def pick(u: np.ndarray, mask: np.ndarray, share: float) -> np.ndarray:
    """Exactly round(share x mask.sum()) records of mask: those with the smallest u."""
    sel = np.zeros(len(u), bool)
    idx = np.where(mask)[0]
    k = int(round(min(max(share, 0.0), 1.0) * len(idx)))
    if k > 0:
        sel[idx[np.argsort(u[idx], kind="stable")[:k]]] = True
    return sel


def pick_ratio(u: np.ndarray, mask: np.ndarray, share: float, group: np.ndarray, ratio: float) -> np.ndarray:
    """Exact counts with P(selected | group) = ratio x P(selected | not group) and the overall share within mask (the
    group's share capped at 1, the rest then from the others)."""
    n1, n0 = int((mask & group).sum()), int((mask & ~group).sum())
    tot = share * (n1 + n0)
    p0 = tot / (n0 + ratio * n1) if n0 + ratio * n1 else 0.0
    p1 = min(1.0, ratio * p0)
    if p1 == 1.0 and n0:
        p0 = (tot - n1) / n0
    return pick(u, mask & group, p1) | pick(u, mask & ~group, p0)


def ranked(u: np.ndarray, mask: np.ndarray, shares: list[float]) -> np.ndarray:
    """Categories 0.. by the rank of u within mask, with exact counts of the shares (normalised); -1 outside mask."""
    out = np.full(len(u), -1)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return out
    order = idx[np.argsort(u[idx], kind="stable")]
    cuts = np.round(np.cumsum(shares) / sum(shares) * len(order)).astype(int)
    start = 0
    for k, c in enumerate(cuts):
        out[order[start:c]] = k
        start = c
    return out


def stratified(u: np.ndarray) -> np.ndarray:
    """(rank + 0.5) / n: stratified quantiles in the order of the frozen uniform."""
    r = np.empty(len(u))
    r[np.argsort(u, kind="stable")] = np.arange(len(u))
    return (r + 0.5) / len(u)


def categorical(u: np.ndarray, probs: list[float]) -> np.ndarray:
    return np.searchsorted(np.cumsum(probs) / sum(probs), u, side="right").clip(0, len(probs) - 1)


def truncnorm_ppf(u: np.ndarray, mu: float, s: float, a: float, b: float) -> np.ndarray:
    fa, fb = ND.cdf((a - mu) / s), ND.cdf((b - mu) / s)
    return mu + s * _ppf(fa + u * (fb - fa))


def beta_restricted_ppf(u: np.ndarray, al: float, be: float, a: float, b: float) -> np.ndarray:
    """Quantiles of Beta(al, be) restricted to [a, b], by a fine numerical CDF (numpy only); 0 < a < b < 1."""
    if not 0 < a < b < 1:
        raise ValueError("the restriction must lie inside (0, 1): the density can be infinite at 0 or 1")
    x = np.linspace(a, b, 20001)
    pdf = x ** (al - 1) * (1 - x) ** (be - 1)
    cdf = np.concatenate([[0], np.cumsum((pdf[1:] + pdf[:-1]) / 2 * np.diff(x))])
    return np.interp(u, cdf / cdf[-1], x)


def municipal_units(drawn_share: np.ndarray, units: np.ndarray) -> np.ndarray:
    """Municipal units from the drawn (auxiliary) property share: round(share x units) bounded to 1..units-1, so the
    municipality never holds every unit; with equal mean areas the realised share is count / units (review 15.11.2026)."""
    return np.clip(np.rint(drawn_share * units), 1, units - 1).astype(int)


def fund_cash(records: np.ndarray, fund_arrears: np.ndarray, unpaid_fund_invoices: np.ndarray) -> np.ndarray:
    """The identity records - cash = owners' unpaid contributions - unpaid repair invoices charged to the fund."""
    return records - fund_arrears + unpaid_fund_invoices


def operator_candidates(board: bool, civil_code: bool, entrusted: bool, administrator: bool, office: bool,
                        ow: dict | None = None) -> dict:
    """The roles that exist in a community with an own account and their weights (PARAMS operator_weights); an owner
    operating the account (not a board member) is a candidate where no board exists, so the set is never empty."""
    ow = ow or PARAMS["operator_weights"]
    roles = {}
    if board:
        roles["board_member"] = ow["board_member"]
    if civil_code:
        roles["owner_operator"] = ow["owner_operator"]
    if entrusted:
        roles["entrusted_manager"] = ow["manager_or_administrator"]
    if administrator:
        roles["administrator"] = ow["manager_or_administrator"]
    if office:
        roles["accounting_office"] = ow["accounting_office"]
    return roles


def current_ratio(assets: np.ndarray, liabilities: np.ndarray, loan: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Current assets / current liabilities with an explicit status: a loan's short-term part belongs to the
    liabilities, and the prototype draws no loan amounts, so the ratio is not computed for a community with a loan;
    zero liabilities give no ratio either."""
    status = np.where(loan, "loan_amount_not_generated", np.where(liabilities > 0, "computed", "no_current_liabilities"))
    ratio = np.where(status == "computed", assets / np.where(liabilities > 0, liabilities, 1.0), np.nan)
    return ratio, status


def sizes(u: np.ndarray, mean: float, sigma: float, target: int, floor: int = 2, safe_from: int = 5) -> tuple[np.ndarray, dict]:
    """Discretised lognormal sizes (>= floor) by stratified quantiles, the scale adjusted (bisection) so that the sum is
    as close to target as rounding allows; the residual goes one dwelling at a time to the largest records with at
    least safe_from units (never across the 3 / 4 threshold of the legal regime)."""
    z = _ppf(stratified(u))
    base = np.exp(math.log(mean) - sigma * sigma / 2 + sigma * z)
    total = lambda k: int(np.maximum(floor, np.rint(k * base)).sum())  # noqa: E731
    lo, hi = 1e-3, 1e3
    if total(lo) > target or total(hi) < target:
        raise ValueError(f"target {target} cannot be reached with {len(u)} records of at least {floor}")
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if total(mid) < target:
            lo = mid
        else:
            hi = mid
    k = hi if abs(total(hi) - target) <= abs(total(lo) - target) else lo
    dw = np.maximum(floor, np.rint(k * base)).astype(int)
    before = dw.copy()
    dw, left = residual(dw, target, safe_from)
    return dw, {"scale": k, "residual": int(target - before.sum()), "left": left,
                "records_changed_by_residual": int((dw != before).sum()),
                "regime_changed_by_residual": int(((dw <= 3) != (before <= 3)).sum())}


def residual(dw: np.ndarray, target: int, safe_from: int = 5) -> tuple[np.ndarray, int]:
    """One dwelling at a time on the largest records with at least safe_from units (each record at most once per pass,
    passes repeated while needed); a decrease never takes a record below safe_from - 1, so records above 3 stay above 3
    and records up to 3 are never touched."""
    dw = dw.copy()
    diff = target - int(dw.sum())
    while diff != 0:
        step = 1 if diff > 0 else -1
        elig = np.where(dw >= safe_from)[0]
        if len(elig) == 0:
            raise ValueError("no record of at least safe_from units to take the residual")
        order = elig[np.argsort(-dw[elig], kind="stable")]
        moved = 0
        for j in order[:abs(diff)]:
            if step < 0 and dw[j] - 1 < safe_from - 1:
                continue
            dw[j] += step
            diff -= step
            moved += 1
        if moved == 0:
            raise ValueError("the residual cannot be placed without crossing the legal size threshold")
    return dw, int(target - dw.sum())


def poisson_ppf(u: np.ndarray, lam: np.ndarray) -> np.ndarray:
    out = np.zeros(len(u), int)
    for i, (uu, la) in enumerate(zip(u, lam)):
        if la <= 0:
            continue
        k, p = 0, math.exp(-la)
        c = p
        while c < uu and k < 1000:
            k += 1
            p *= la / k
            c += p
        out[i] = k
    return out


def generate(seed: int = 20261115, records: int = 20000, params: dict | None = None, frozen: dict | None = None) -> dict:
    """Column arrays over all voivodeships, the weight of each record and the calibration notes."""
    P = {**PARAMS, **(params or {})}
    alloc = allocation(records, P["frame"])
    frozen = frozen or draws(seed, alloc)
    meta = frozen.get("__meta__", {})
    if meta.get("seed", seed) != seed:
        raise ValueError(f"the frozen draws are of seed {meta['seed']}, not {seed}")
    ages = age_rows()
    nat_mean = DWELLINGS_ALL / FRAME[P["frame"]]
    cols: dict[str, list] = {}
    notes = {"reconciliation": {}, "spec_version": SPEC_VERSION, "generator": VERSION}

    def put(name, arr):
        cols.setdefault(name, []).append(np.asarray(arr))

    for v in alloc:
        d, n, w = frozen[v["name"]], v["records"], v["weight"]
        if len(d["u_size"]) != n:
            raise ValueError("the frozen draws have another number of records")
        # size: discretised lognormal with the voivodeship's (or the national) mean, reconciled to the D dwellings
        mean = v["dwellings"] / v["frame"] if P["size_regional_mean"] else nat_mean
        target = int(round(v["dwellings"] / w)) if P["size_regional_mean"] else int(round(v["frame"] * nat_mean / w))
        dw, rec = sizes(d["u_size"], mean, P["size_sigma"], target)
        notes["reconciliation"][v["name"]] = {"records_target": target, **rec}
        nonres = np.zeros(n, int)
        units = dw + nonres
        regime = np.where(units <= 3, "civil_code", "statutory")
        above3 = units > 3
        area = dw * np.exp(math.log(MEAN_AREA_ALL) - 0.25 ** 2 / 2 + 0.25 * _ppf(d["u_area"]))
        row = np.array([next(k for lo, hi, k in SIZE_ROWS if lo <= max(x, 3) <= hi) for x in dw])
        age_k = np.full(n, -1)
        for _, _, key in SIZE_ROWS:
            r = ranked(d["u_age"], row == key, ages[key])
            age_k = np.where(r >= 0, r, age_k)
        age = np.array(AGE)[age_k]
        if P["buildings_poisson"]:
            buildings = 1 + poisson_ppf(d["u_buildings"], np.where(dw > 20, (dw - 20) / 40, 0.0))
        else:
            buildings = np.ones(n, int)
        # municipal presence (exact counts; pre-1945 with P["municipal_pre1945_weight"] times the probability)
        elig = above3 | P["municipal_up_to_3"]
        muni = pick_ratio(d["u_municipal"], elig, P["municipal_presence"], age == "before 1945", P["municipal_pre1945_weight"])
        al, be, a, b = P["municipal_share_beta"]
        drawn = np.where(muni, beta_restricted_ppf(d["u_mshare"], al, be, a, b), 0.0)
        munits = np.where(muni, municipal_units(drawn, np.maximum(units, 2)), 0).astype(int)
        mshare = munits / units  # equal mean areas (PROTOTYPE_Z): the realised share follows the count
        not_separated = munits.copy()
        private_units = units - munits
        owners = private_units + muni.astype(int)
        # management: legal basis, board composition, administrator or entrusted manager
        entrusted = pick(d["u_basis"], above3, P["entrusted_above_3"])
        basis = np.where(~above3, "civil_code", np.where(entrusted, "art18_entrusted", "statutory_board"))
        board = basis == "statutory_board"
        prof = pick(d["u_board"], board, P["professional_board_member"])
        owners_board = board & ~prof
        bsize = np.where(prof, 1, np.where(owners_board, 1 + ranked(d["u_board_size"], owners_board, list(P["board_size_probs"])), 0))
        civil = basis == "civil_code"
        admin = (pick(d["u_admin"], owners_board & muni, P["administrator_municipal_owners_board"])
                 | pick(d["u_admin"], owners_board & ~muni, P["administrator_private_owners_board"])
                 | pick(d["u_admin"], civil, P["administrator_civil_code"]))
        served = admin | entrusted
        firm = np.full(n, "none", dtype=object)
        firm[served & muni] = "private_firm"
        firm[pick(d["u_firm"], served & muni, P["firm_municipal_company_given_municipal"])] = "municipal_company"
        firm[served & ~muni] = "private_firm"
        firm[pick(d["u_firm"], served & ~muni, P["firm_natural_person_given_private"])] = "natural_person"
        firm = firm.astype(str)
        # accounts and the banking operator over the roles that exist
        account = pick(d["u_account"], above3, P["own_account_above_3"]) | pick(d["u_account"], ~above3, P["own_account_up_to_3"])
        office = pick(d["u_office"], ~served, P["accounting_office_without_firm"])
        operator = np.full(n, "none", dtype=object)
        for i in np.where(account)[0]:
            roles = operator_candidates(bool(board[i]), bool(civil[i]), bool(entrusted[i]), bool(admin[i]), bool(office[i]),
                                        P["operator_weights"])
            names, wts = list(roles), np.array(list(roles.values()))
            operator[i] = names[int(np.searchsorted(np.cumsum(wts) / wts.sum(), d["u_operator"][i], side="right").clip(0, len(names) - 1))]
        operator = operator.astype(str)
        representation = np.where(civil, "co_owners_civil_code", np.where(
            entrusted, "entrusted_manager_contract", np.where(bsize >= 2, "two_board_members_jointly", "single_board_member")))
        # repair fund: rate x area, records and cash apart; renovations do not depend on the fund
        fund = pick(d["u_fund"], above3, P["fund_above_3"]) | pick(d["u_fund"], ~above3, P["fund_up_to_3"])
        mu_r, s_r, a_r, b_r = P["fund_rate_truncnorm"]
        rate = np.where(fund, truncnorm_ppf(d["u_rate"], mu_r, s_r, a_r, b_r) if P["fund_rate_fixed"] is None
                        else P["fund_rate_fixed"], 0.0)
        fund_assessed = rate * area
        m0, m1 = P["fund_months"]
        months = m0 + d["u_months"] * (m1 - m0)
        rk = ranked(d["u_renov"], np.ones(n, bool), [P["renovation_ongoing"], P["renovation_recent"], P["renovation_planned"],
                                                     1 - P["renovation_ongoing"] - P["renovation_recent"] - P["renovation_planned"]])
        renov = np.array(["ongoing", "recent", "planned", "none"])[rk]
        loan = pick(d["u_loan"], np.isin(renov, ["ongoing", "recent"]), P["loan_given_ongoing_or_recent"])
        sep_acc = pick(d["u_sepacc"], fund & account, P["separate_fund_account"])
        # costs and charges of the month
        small = dw <= 7
        small_services = pick(d["u_small_services"], small, P["small_services_through_community"])
        operational = np.where(small, P["small_fixed_management_pln"] + area * OPERATIONAL_PER_M2 * (1 - MANAGEMENT_SHARE)
                               * P["small_area_factor"], area * OPERATIONAL_PER_M2)
        services = np.where(small & ~small_services, 0.0, area * SERVICES_PER_M2)
        under = pick(d["u_under"], firm == "municipal_company", P["under_set_share_of_municipal_company"]) \
            if P["under_set_advances"] > 0 else np.zeros(n, bool)
        mgmt_advance = operational * np.where(under, 1 - P["under_set_advances"], 1.0)
        assessed = mgmt_advance + services + fund_assessed
        for name, arr in (("voivodeship", np.full(n, v["name"])), ("weight", np.full(n, w)), ("dwellings", dw),
                          ("units_nonresidential", nonres), ("units_total", units), ("legal_size_regime", regime),
                          ("floor_area_m2", area), ("building_age", age), ("buildings", buildings),
                          ("municipal", muni), ("municipal_share_drawn", drawn), ("municipal_common_property_share", mshare),
                          ("municipal_units_count", munits), ("not_separated", not_separated),
                          ("private_units", private_units), ("owners", owners), ("owner_settlement_accounts", units),
                          ("management_basis", basis), ("board_composition", np.where(prof, "professional_member",
                                                                                      np.where(owners_board, "owners_only", "none"))),
                          ("board_size", bsize), ("external_administrator", admin),
                          ("serving_firm", firm), ("representation", representation), ("own_account", account),
                          ("accounting_office", office), ("banking_operator", operator), ("repair_fund", fund), ("repair_fund_rate", rate),
                          ("fund_area_m2", np.where(fund, area, 0.0)), ("fund_contributions_assessed", fund_assessed),
                          ("fund_months", months), ("major_renovation", renov), ("loan", loan),
                          ("separate_fund_account", sep_acc), ("accounts_count", account.astype(int) + sep_acc.astype(int)),
                          ("operational_costs", operational), ("services_through_community", services),
                          ("management_advances", mgmt_advance), ("charges_assessed", assessed), ("under_set", under)):
            put(name, arr)
        put("_u_buffer", d["u_buffer"]); put("_z_arrears", d["z_arrears"]); put("_u_arrears", d["u_arrears"])
        put("_z_amount", d["z_amount"]); put("_u_other", d["u_other"])
    c = {k: np.concatenate(v) for k, v in cols.items()}
    W = c["weight"]
    # other entities: the rest of the D non-natural-person dwellings after the municipality's (no silent fix if negative)
    uo = c.pop("_u_other")
    muni_dw = float((W * c["municipal_units_count"]).sum())
    other_target = NON_NATURAL_DWELLINGS - muni_dw
    notes["feasible"] = other_target >= 0
    if not notes["feasible"]:
        notes["infeasible_reason"] = f"municipal dwellings {muni_dw:.0f} exceed the D bound {NON_NATURAL_DWELLINGS}"
    other_cnt = lambda r: np.minimum(c["private_units"], np.floor(c["private_units"] * r + uo)).astype(int)  # noqa: E731
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if float((W * other_cnt(mid)).sum()) < max(other_target, 0) else (lo, mid)
    r_oth = hi if abs(float((W * other_cnt(hi)).sum()) - other_target) <= abs(float((W * other_cnt(lo)).sum()) - other_target) else lo
    other = other_cnt(r_oth) if other_target > 0 else np.zeros(len(W), int)
    c["other_entity_units"] = other
    c["natural_person_units"] = c["private_units"] - other
    c["owners"] = c["natural_person_units"] + (other > 0).astype(int) + c["municipal"].astype(int)
    notes["owner_types"] = {"municipal_dwellings": muni_dw, "other_entity_dwellings": float((W * other).sum()),
                            "other_target": other_target,
                            "natural_person_dwellings": float((W * c["natural_person_units"]).sum()),
                            "natural_person_dwellings_D": DWELLINGS_NATURAL}
    # arrears: calibrated to the D count and amount in their own scope, the natural persons' dwellings
    g = np.exp(0.8 * c.pop("_z_arrears") - 0.32)
    ua = c.pop("_u_arrears")
    count = lambda q: np.minimum(c["natural_person_units"], np.floor(c["natural_person_units"] * q * g + ua)).astype(int)  # noqa: E731
    wsum = lambda q: float((W * count(q)).sum())  # noqa: E731
    lo, hi = 0.0, 1.0
    for _ in range(100):  # the weighted count is non-decreasing in q: bisection to the closest attainable value
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if wsum(mid) < ARREARS_DWELLINGS else (lo, mid)
    q = hi if abs(wsum(hi) - ARREARS_DWELLINGS) <= abs(wsum(lo) - ARREARS_DWELLINGS) else lo
    cnt, got = count(q), wsum(q)
    if abs(got - ARREARS_DWELLINGS) > 2 * W.max():
        raise RuntimeError(f"the arrears calibration did not reach the D count ({got:.0f} vs {ARREARS_DWELLINGS})")
    A = np.exp(0.6 * c.pop("_z_amount") - 0.18)
    raw_amt = cnt * A
    amt = raw_amt * ARREARS_PLN / float((W * raw_amt).sum())
    c["arrears_dwellings"], c["arrears_pln"] = cnt, amt
    notes["arrears"] = {"q": q, "weighted_dwellings": float((W * cnt).sum()), "weighted_pln": float((W * amt).sum())}
    # balances over the profile month (identities of the specification's fields)
    fund_rec_open = c["fund_contributions_assessed"] * c["fund_months"]
    # the fund's part of the owners' arrears, in proportion to the assessed contributions, at most the fund's records
    fund_arrears = np.minimum(fund_rec_open, np.where(
        c["charges_assessed"] > 0, amt * c["fund_contributions_assessed"] / np.maximum(c["charges_assessed"], 1e-9), 0.0))
    fund_inv_open = np.zeros_like(fund_rec_open)      # PROTOTYPE_Z fund_unpaid_invoices
    fund_cash_open = fund_cash(fund_rec_open, fund_arrears, fund_inv_open)
    paid = c["charges_assessed"]                      # steady month (PROTOTYPE_Z flows)
    made = c["operational_costs"] + c["services_through_community"]
    b0, b1 = P["cash_buffer_months"]
    op_open = (b0 + c.pop("_u_buffer") * (b1 - b0)) * made
    deficit = c["operational_costs"] - c["management_advances"]
    route = P["under_set_financing"]
    if route not in ("fund", "invoices", "buffer"):
        raise ValueError(f"under_set_financing {route!r}: fund, invoices or buffer")
    fin_fund = (deficit > 0) & (route == "fund") & c["repair_fund"] & (fund_cash_open >= deficit)
    fin_buffer = (deficit > 0) & (route == "buffer") & (op_open >= deficit)  # the deficit lowers operating cash
    transfer = np.where(fin_fund, deficit, 0.0)
    c["under_set_route"] = np.where(deficit <= 0, "none", np.where(fin_fund, "fund", np.where(fin_buffer, "buffer", np.where(
        route == "invoices", "invoices", "invoices_fallback"))))
    unpaid_growth = np.where((deficit > 0) & ~fin_fund & ~fin_buffer, deficit, 0.0)
    made = made - unpaid_growth                       # an invoice left unpaid is not a payment made
    op_close = op_open + (paid - c["fund_contributions_assessed"]) - made + transfer
    fund_rec_close = fund_rec_open + c["fund_contributions_assessed"] - transfer
    fund_cash_close = fund_cash_open + c["fund_contributions_assessed"] - transfer
    unpaid_invoices = 0.5 * (made + unpaid_growth) + unpaid_growth
    c.update({"fund_records_opening": fund_rec_open, "fund_records_closing": fund_rec_close,
              "fund_cash_opening": fund_cash_open, "fund_cash_closing": fund_cash_close, "fund_arrears": fund_arrears,
              "fund_unpaid_invoices_opening": fund_inv_open, "fund_unpaid_invoices_closing": fund_inv_open,
              "payments_received": paid, "payments_made": made, "operating_cash_opening": op_open,
              "operating_cash_closing": op_close, "transfer_from_fund": transfer, "unpaid_supplier_invoices": unpaid_invoices})
    liab = unpaid_invoices + fund_inv_open            # no owners' overpayments, no refundable utility advances (PROTOTYPE_Z)
    c["current_ratio"], c["current_ratio_status"] = current_ratio(op_close + fund_cash_close + amt, liab, c["loan"])
    # the operating part alone (the repair fund and its arrears left out), the closer scope to the TNN cases
    c["current_ratio_operating"], _ = current_ratio(op_close + amt - fund_arrears, unpaid_invoices, c["loan"])
    held = op_close + fund_cash_close
    c["funds_on_accounts"] = np.where(c["own_account"], held, 0.0)
    c["cash_outside_accounts"] = np.where(c["own_account"], 0.0, held)
    notes["allocation"] = [{k: v[k] for k in ("name", "frame", "records", "weight", "dwellings")} for v in alloc]
    notes["records_actual"] = int(len(W))
    notes["params"] = P
    return {"columns": c, "notes": notes}
