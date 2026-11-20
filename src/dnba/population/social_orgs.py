"""Social organisations segment, prototype 0.2 (profile specification 0.4, data/reference/social_org_profile_draft.json;
code review of 0.1 applied).

A frozen synthetic population per subgroup (records with weights = frame count / records). Every random number is
drawn once, from named streams (numpy default_rng of (seed, a stable code of the subgroup, a stable code of the
stream)), so that scenarios change only thresholds and parameters, never the draws, and adding a stream changes no
other one. The copula loading changes the latent scores Z_j = l F + sqrt(1 - l^2) e_j built from the same F and e_j;
every threshold of a field whose GUS share is defined among organisations with revenue is taken WITHIN those records
(an empirical quantile), so the target margins stay the same across loadings. D figures: the GUS tables by type
(agg_gus_ngo_2024_by_type.json), the revenue fit (scripts/social_org_revenue_fit.py), the frame (segment_frame_draft.json)
and the confirmed accounts (MF 1.5% 2026, ARiMR 2025 aid). Everything else is Z and named in PARAMS or the constants.
No LLM.

Z choices of the structure (documented): fees, other revenue, financial income, procurement and the public/private kind
are outside the copula (independent uniforms); the activity categories are ordered free < paid statutory < economic <
both on the latent scale; a record without revenue has free activity only and no paid staff (GUS shares of all
organisations are kept by drawing the paid and staffed categories among the records with revenue with shares / p).
"""
from __future__ import annotations

import json
import math
import zlib
from pathlib import Path
from statistics import NormalDist

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
GUS = ROOT / "data/reference/sources/2026-11-14-ngo-wspolnoty/agg_gus_ngo_2024_by_type.json"
VERSION = "0.6"
MEASUREMENT_DATE = "2026-11-14"  # the one moment at which the account fields are stated (the frame of the clubs)
N = NormalDist()

# subgroup -> (GUS row of the type, frame count (Z: main number), confirmed own accounts (D))
SUBGROUPS = {
    "foundations": ("Fundacje", 22400, 2960),
    "associations_typical": ("Typowe stowarzyszenia i organizacje społeczne", 32800, 4991 + 52),
    "associations_sport": ("Stowarzyszenia sportowe", 20200, 687),
    "volunteer_fire_brigades": ("Ochotnicze straże pożarne", 14800, 243),
    "hunting_clubs": ("Koła łowieckie", 2500, 1),
    "associations_ordinary": ("Typowe stowarzyszenia i organizacje społeczne", 12400, 0),  # Z: the typical profile
    "rural_housewives_clubs": ("Koła gospodyń wiejskich", 19596, 18100),  # D: 18,100 clubs paid the 2025 aid (of 18,952
    # registered at the end of 2025); the frame of 14.11.2026 also holds the clubs registered in 2026 (by entry date)
}
GROUP_B = {"associations_ordinary"}
ACTIVITY = ("free_only", "paid_statutory", "economic", "paid_statutory_and_economic")
STAFF = ("none", "civil_law_only", "1-5", "6+")
# D, table 1.26 (value shares of all revenue): the market components and the public components
MARKET_PARTS = {"economic": 11.1, "paid_statutory": 18.8, "financial": 1.7, "procurement": 0.8}
PUBLIC_PARTS = {"grants": 22.7 + 11.8 + 5.8, "pit_court_other": 3.5 + 2.1}  # grants: local, central, EU and foreign
NONMARKET_KIND = (0.4016, 0.2109, 0.3875)  # Z: public only, private only, both = GUS-wide 25.7 / 13.5 / 24.8 over 64.0
# D (text pp. 119-120, among organisations with revenue): public 50.5%, local-government grants 30.8%, central 18.4%;
# Z: local and central independent among public recipients -> P(any grant | public) = 1 - (1 - .308/.505)(1 - .184/.505)
P_GRANT_GIVEN_PUBLIC = 1 - (1 - 0.308 / 0.505) * (1 - 0.184 / 0.505)
P_PROCUREMENT_GIVEN_PAID = 0.009 / 0.355  # D: 0.9% of organisations with revenue, over the 35.5% with market revenue (Z)
FINANCIAL_GIVEN_PAID = 0.080  # D: GUS-wide 8.0% (Z: applied to organisations with paid or economic activity)
VAT_LIMIT_2026 = 240000.0  # D: MF, subjective exemption limit from 1 January 2026

PARAMS = {  # every Z the scenarios vary; the central values
    "loading": 0.6,                        # copula loading (0 / 0.4 / 0.6 / 0.8)
    "no_premise_account_A": 0.80,          # 0.60 / 0.80 / 0.95
    "no_premise_account_B": 0.50,          # 0.30 / 0.50 / 0.70
    "premise_account": 0.98,               # Z, to decide: 0.95 / 0.98 / 1.00
    "paid_statutory_counted_in_vat_limit": 0.5,  # the unrecognised part: 0 / 0.5 / 1
    "voluntary_vat": 0.1,                  # among organisations with economic activity under the limit (Z)
    "project_ongoing_given_grants": 0.5,   # Z: a grant-financed project ongoing at the moment of measurement
    "separate_project_account": 0.5,       # 0 / 0.5 / 1
    "aid_2025_as_own_account": True,       # Z: the clubs' 2025 aid went to an account whose owner is not stated; True: an
                                           # own account kept to the measurement date; False: only a premise
    "representative_account_ordinary": 0.5,  # Z: an ordinary association without an own account uses a representative's
                                           # private account (a separate field, never an own account)
    "calibrate_amounts": True,             # step 2: source multipliers fitted to chart 1.54 by subgroup (revenue per record kept)
    "confirmed_selection_noise": 2.6,      # Z, calibrated: the pooled weighted median revenue of the MF-confirmed,
                                           # mean over seeds 0-19, equals the GUS median of OPP (187.6k; 2.3 / 3.0 variants)
}
KGW_AGG = ROOT / "data/reference/sources/2026-11-14-krkgw/agg_krkgw.json"


def kgw_cohorts() -> dict:
    """Clubs (without the unions) in the register of 14.11.2026 by entry date: registered in 2026, and by the end of 2025.
    D from the aggregate when it counts the clubs' entry years apart from the unions (krkgw_aggregate.py 0.2); else the
    D bounds 710-722 (722 entries of 2026 include up to 12 unions) and the midpoint 716 as Z."""
    agg = json.loads(KGW_AGG.read_text())["whole_register"]
    clubs = agg["clubs"]
    if "clubs_registration_year" in agg:
        n26 = agg["clubs_registration_year"].get("2026", 0)
        status = "D: entry dates of the clubs (without the unions)"
    else:
        n26 = agg["registration_year"]["2026"] - agg["unions_by_name"] / 2
        status = "Z: midpoint of the D bounds 710-722 (entries of 2026 incl. up to 12 unions); rerun krkgw_aggregate.py for D"
    by_2025 = clubs - n26
    # Z: the clubs struck off since the end of 2025 (18,952 then) lost the aid in the same proportion
    aid = 18100 * min(1.0, by_2025 / 18952)
    return {"clubs": clubs, "registered_2026": n26, "registered_by_2025_now": by_2025, "aid_2025_in_frame": aid, "status": status}


STREAMS = ("F", "e_revenue", "e_activity", "e_staff", "e_nonmarket", "e_loan", "u_kind", "u_fees", "u_other",
           "u_financial", "u_procurement", "u_grant", "u_fallback", "u_confirm", "n_confirm", "u_account", "u_vat_voluntary",
           "u_project", "u_project_acc", "u_operator", "u_registered_2026", "u_representative", "u_revenue_values")
OPERATOR = ("no_own_account", "board_or_treasurer", "accounting_office", "employee")


def gus_tables() -> dict:
    return json.loads(GUS.read_text())["tables"]


def revenue_fit(p: float, mean: float, median: float) -> tuple[float, float]:
    mpos = mean / p
    z = N.inv_cdf((0.5 - (1 - p)) / p)
    s = z + math.sqrt(z * z + 2 * math.log(mpos / median))
    return math.log(mpos) - s * s / 2, s


def _ppf(u: np.ndarray) -> np.ndarray:
    return np.vectorize(N.inv_cdf)(np.clip(u, 1e-12, 1 - 1e-12))


def _cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1 + np.vectorize(math.erf)(z / math.sqrt(2)))


def _code(name: str) -> int:
    return zlib.crc32(name.encode())


def draws(seed: int, records: int) -> dict:
    """The frozen random numbers by subgroup and named stream."""
    out = {}
    for sg in SUBGROUPS:
        out[sg] = {}
        for st in STREAMS:
            rng = np.random.default_rng([seed, _code(sg), _code(st)])
            out[sg][st] = rng.standard_normal(records) if st[0] in "Fen" else rng.random(records)
    return out


GROUPS = ("market", "public", "private", "fees", "other")  # the five source groups of chart 1.54
GROUP_OF = {"economic": "market", "paid_statutory": "market", "financial": "market", "procurement": "market",
            "grants": "public", "public_other": "public", "private": "private", "fees": "fees", "other": "other"}


def group_shares(amounts: dict) -> dict:
    tot = sum(float(v.sum()) for v in amounts.values()) or 1.0
    return {g: 100 * sum(float(amounts[k].sum()) for k in amounts if GROUP_OF[k] == g) / tot for g in GROUPS}


def feasibility(revenue: np.ndarray, present: np.ndarray, targets: np.ndarray) -> dict:
    """With every record's revenue fixed and absent sources at zero, the value share of a set A of groups is at least the
    revenue share of the records whose present groups all lie in A: the largest violation over all non-empty proper
    subsets of the five groups (in percentage points), with its subset."""
    t = 100 * targets / targets.sum()
    tot = revenue.sum() or 1.0
    worst, arg = 0.0, None
    for mask in range(1, 2 ** len(GROUPS) - 1):
        inA = np.array([(mask >> j) & 1 for j in range(len(GROUPS))], bool)
        inside = present.any(axis=1) & ~(present & ~inA).any(axis=1)
        need = 100 * revenue[inside].sum() / tot
        v = need - t[inA].sum()
        if v > worst:
            worst, arg = v, [g for g, b in zip(GROUPS, inA) if b]
    return {"max_violation_pp": worst, "subset": arg}


def _shares_for(revenue: np.ndarray, w: dict, m: dict) -> tuple[dict, np.ndarray]:
    has = revenue > 0
    ww = {k: v * m[GROUP_OF[k]] for k, v in w.items()}
    tot = sum(ww.values())
    amt = {k: np.where(has & (tot > 0), revenue * v / np.where(tot > 0, tot, 1), 0.0) for k, v in ww.items()}
    return amt, np.array([group_shares(amt)[g] for g in GROUPS]) / 100


def calibrate(revenue: np.ndarray, w: dict, targets: np.ndarray, iters: int = 2000, tol: float = 1e-4,
              max_iters: int = 50000) -> dict:
    """Step 2: multipliers of the five source groups so that the value shares of the subgroup approach chart 1.54 while
    every record keeps its revenue and its absent sources stay at zero (iterative proportional scaling on the group
    totals). After `iters` the scaling continues while the error still falls by at least 1% per 500 iterations, up to
    `max_iters` (a feasible system can converge slowly). The returned error is the one of the RETURNED multipliers;
    converged = that error < tol (as a share; 1e-4 = 0.01 p.p.)."""
    t = targets / targets.sum()
    m = dict.fromkeys(GROUPS, 1.0)
    it, last_check = 0, None
    while it < max_iters:
        _, sh = _shares_for(revenue, w, m)
        err = float(np.abs(sh - t).max())
        if err < tol:
            break
        if it >= iters and it % 500 == 0:
            if last_check is not None and err > 0.99 * last_check:
                break  # no material progress: stop and report the error as it is
            last_check = err
        for j, g in enumerate(GROUPS):
            if t[j] > 0 and sh[j] > 0:
                m[g] = min(1e9, max(1e-9, m[g] * t[j] / sh[j]))
        top = max(m.values())
        m = {g: v / top for g, v in m.items()}  # scale-free: keeps the multipliers bounded
        it += 1
    _, sh = _shares_for(revenue, w, m)
    err = float(np.abs(sh - t).max())
    return {"multipliers": m, "targets_percent": dict(zip(GROUPS, (100 * t).tolist())), "iterations": it,
            "max_abs_error_pp": 100 * err, "converged": err < tol}


def _within(u: np.ndarray, mask: np.ndarray, shares: list[float]) -> np.ndarray:
    """Categories by the empirical quantiles of u WITHIN mask (shares of the categories, low u first); -1 outside."""
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


def generate(seed: int = 20261115, records: int = 3000, params: dict | None = None, frozen: dict | None = None) -> dict:
    """Profiles by subgroup as column arrays (and the weight of a record)."""
    P = {**PARAMS, **(params or {})}
    T = gus_tables()
    frozen = frozen or draws(seed, records)
    lam = P["loading"]
    res = {}
    for sg, (row, count, confirmed) in SUBGROUPS.items():
        d = frozen[sg]
        if len(d["F"]) != records:
            raise ValueError("the frozen draws have another number of records")
        u = {k: _cdf(lam * d["F"] + math.sqrt(1 - lam * lam) * d[f"e_{k}"]) for k in ("revenue", "activity", "staff", "nonmarket", "loan")}
        _, share, _, mean_th, med_th = T["1.25"]["rows"][row]
        p = share / 100
        mu, s = revenue_fit(p, mean_th * 1000, med_th * 1000)
        # a FIXED set of revenue values (iid draws of the mixture from their own stream, independent of the loading),
        # assigned by the rank of the latent revenue score: the same values and the same number of zeros at every loading
        w = d["u_revenue_values"]
        vals = np.where(w >= 1 - p, np.exp(mu + s * _ppf(np.where(w >= 1 - p, (w - (1 - p)) / p, 0.5))), 0.0)
        revenue = np.empty(records)
        revenue[np.argsort(u["revenue"], kind="stable")] = np.sort(vals)
        has_rev = revenue > 0
        # activity (table 1.6, shares of ALL organisations): paid / economic categories only among records with revenue
        a = np.array(T["1.6"]["rows"][row], float) / 100
        a_rev = np.concatenate([[1 - a[1:].sum() / p], a[1:] / p])
        act = np.where(has_rev, _within(u["activity"], has_rev, list(a_rev)), 0)
        # staff (chart 1.33, all organisations): staffed classes only among records with revenue
        st = np.array(T["chart_1.33"]["rows"][row], float) / 100
        st_rev = np.concatenate([[1 - st[1:].sum() / p], st[1:] / p])
        staff = np.where(has_rev, _within(u["staff"], has_rev, list(st_rev)), 0)
        paid_stat = np.isin(act, (1, 3))
        economic = np.isin(act, (2, 3))
        c127 = T["1.27"]["rows"][row]  # [count, market %, non-market %, fees %, other %] among organisations with revenue
        fs, os_ = (c127[3] or 0) / 100, c127[4] / 100  # exact counts within the records with revenue (independent uniforms)
        fees = _within(d["u_fees"], has_rev, [1 - fs, fs]) == 1
        other = _within(d["u_other"], has_rev, [1 - os_, os_]) == 1
        # financial income: 8% (GUS-wide) with paid or economic activity; without it the share that makes the market
        # presence among organisations with revenue equal to table 1.27 (Z)
        paid_any = paid_stat | economic
        a_share = paid_any[has_rev].mean() if has_rev.any() else 0.0
        f_rest = min(1.0, max(0.0, (c127[1] / 100 - a_share) / max(1e-9, 1 - a_share)))
        financial = (_within(d["u_financial"], paid_any, [1 - FINANCIAL_GIVEN_PAID, FINANCIAL_GIVEN_PAID]) == 1) | \
                    (_within(d["u_financial"], has_rev & ~paid_any, [1 - f_rest, f_rest]) == 1)
        procurement = _within(d["u_procurement"], paid_any, [1 - P_PROCUREMENT_GIVEN_PAID, P_PROCUREMENT_GIVEN_PAID]) == 1
        # non-market presence: exactly the GUS share among the records with revenue (table 1.27); every organisation with
        # revenue needs a source, so the records without any other source get it first; then (step 1 of the amount
        # calibration, Z, review of 51790d8) the rest of the quota, the public kind within non-market funds and the
        # presence of other revenue go by a score beta * rank of revenue + (1 - beta) * rank of their own uniform or latent score, with the
        # smallest beta (0, 0.1, ..., 1) for which chart 1.54 is feasible with fixed revenues (Z, reported); every count
        # stays exact
        v154 = dict(zip(GROUPS, T["chart_1.54"]["rows"][row]))
        msum, psum = sum(MARKET_PARTS.values()), sum(PUBLIC_PARTS.values())
        targets = np.array([v154[g] for g in GROUPS], float)
        rrev = np.argsort(np.argsort(revenue, kind="stable"), kind="stable") / max(1, records - 1)
        rk = lambda x: np.argsort(np.argsort(x, kind="stable"), kind="stable") / max(1, records - 1)  # noqa: E731
        quota = int(round(c127[2] / 100 * has_rev.sum()))

        def assign(beta):
            oth = _within(beta * rrev + (1 - beta) * rk(d["u_other"]), has_rev, [1 - os_, os_]) == 1
            fe = fees.copy()
            must = has_rev & ~(paid_any | financial | fe | oth)
            nm = must.copy()
            k = max(0, quota - int(must.sum()))
            rest = np.where(has_rev & ~nm)[0]
            if k > 0:
                sc = beta * rrev + (1 - beta) * rk(u["nonmarket"])
                nm[rest[np.argsort(-sc[rest], kind="stable")[:k]]] = True
            # only if the quota is smaller than the records without any other source: one of fees and other (Z)
            nn = has_rev & ~(paid_any | financial | fe | oth) & (np.cumsum(must) > quota)
            nm = nm & ~nn
            cand = [(c127[3] or 0) if sg != "foundations" else 0, c127[4]]
            pick = np.searchsorted(np.cumsum(cand) / max(1e-9, sum(cand)), d["u_fallback"], side="right").clip(0, 1)
            fe = fe | (nn & (pick == 0))
            oth = oth | (nn & (pick == 1))
            # kinds by ascending score: private only (the smallest), public only, both (exact counts)
            kd = _within(beta * rrev + (1 - beta) * rk(d["u_kind"]), nm, [NONMARKET_KIND[1], NONMARKET_KIND[0], NONMARKET_KIND[2]])
            pub = nm & (kd != 0)
            pri = nm & (kd != 1)
            gr = _within(d["u_grant"], pub, [1 - P_GRANT_GIVEN_PUBLIC, P_GRANT_GIVEN_PUBLIC]) == 1
            flags = {"economic": economic, "paid_statutory": paid_stat, "financial": financial, "procurement": procurement,
                     "grants": gr, "public_other": pub, "private": pri, "fees": fe, "other": oth}
            base = {**{k2: v154["market"] * MARKET_PARTS[k2] / msum for k2 in MARKET_PARTS},
                    "grants": v154["public"] * PUBLIC_PARTS["grants"] / psum, "private": v154["private"],
                    "fees": v154["fees"], "other": v154["other"]}
            ww = {k2: flags[k2] * max(base[k2], 1e-3) for k2 in base}  # a present source with a zero value share: small (Z)
            ww["public_other"] = pub * max(v154["public"], 1e-3) * np.where(gr, PUBLIC_PARTS["pit_court_other"] / psum, 1.0)
            present = np.column_stack([sum(ww[k2] for k2 in ww if GROUP_OF[k2] == g) > 0 for g in GROUPS])
            return nm, pub, pri, gr, fe, oth, nn, ww, feasibility(revenue[has_rev], present[has_rev], targets)

        beta_used = 0.0
        for beta in [b / 10 for b in range(11)] if P["calibrate_amounts"] else [0.0]:
            nonmarket, public, private, grants_present, fees, other, none, w, feas = assign(beta)
            beta_used = beta
            if feas["max_violation_pp"] <= 0.01:
                break
        calib = calibrate(revenue, w, targets) if P["calibrate_amounts"] else {"multipliers": dict.fromkeys(GROUPS, 1.0)}
        calib.update(beta=beta_used, feasibility=feas)
        # the status of the subgroup's calibration: only 'fitted' meets the targets; the others stay as diagnostics
        calib["status"] = ("not_calibrated" if not P["calibrate_amounts"] else
                           "infeasible" if feas["max_violation_pp"] > 0.01 else
                           "not_converged" if not calib["converged"] else "fitted")
        w = {k: v * calib["multipliers"][GROUP_OF[k]] for k, v in w.items()}
        tot_w = sum(w.values())
        amounts = {k: np.where(has_rev & (tot_w > 0), revenue * v / np.where(tot_w > 0, tot_w, 1), 0.0) for k, v in w.items()}
        calib["achieved_percent"] = group_shares(amounts)
        removed = np.zeros(records)
        if sg in GROUP_B:  # no economic nor paid public benefit activity (art. 42 ust. 1 pkt 3-4): those sources (and
            # procurement, a paid service) are removed and NOT moved to other sources, so the revenue shrinks
            removed = amounts["economic"] + amounts["paid_statutory"] + amounts["procurement"]
            for k in ("economic", "paid_statutory", "procurement"):
                amounts[k] = np.zeros(records)
            revenue = sum(amounts.values())  # the sum of the kept amounts (no numeric residue), then the flags
            act = np.zeros_like(act)
            paid_stat = economic = procurement = np.zeros(records, bool)
            has_rev = revenue > 0
            staff = np.where(has_rev, staff, 0)  # no paid staff without revenue (Z, as for the other subgroups)
        # evidence of accounts, with its date (D counts): MF 1.5% transfers in 2026 to the OPP's own account (ranked by the
        # latent revenue score + noise, Z, among organisations with revenue); the clubs' 2025 aid paid to an account
        # (owner not stated), only among clubs registered by the end of 2025 (which ones: Z)
        n_conf = round(records * confirmed / count)
        mf_2026 = np.zeros(records, bool)
        aid_2025 = np.zeros(records, bool)
        registered_2026 = np.zeros(records, bool)
        if sg == "rural_housewives_clubs":
            kc = kgw_cohorts()
            n26 = round(records * kc["registered_2026"] / kc["clubs"])
            registered_2026[np.argsort(d["u_registered_2026"])[:n26]] = True
            elig = np.where(~registered_2026)[0]
            n_conf = round(records * kc["aid_2025_in_frame"] / kc["clubs"])
            aid_2025[elig[np.argsort(d["u_confirm"][elig])[:n_conf]]] = True
        elif n_conf:
            score = np.where(has_rev, _ppf(u["revenue"]) + P["confirmed_selection_noise"] * d["n_confirm"], -np.inf)
            mf_2026[np.argsort(-score)[:n_conf]] = True
        opp = mf_2026.copy()  # MF recipients are OPP; ordinary associations may be OPP (art. 40b ust. 1 pkt 5): no data, none here (Z)
        # VAT (the 2026 limit applied to the scale of the 2024 revenue: Z): counted sales = economic + procurement + the
        # variant of paid statutory; financial income is not a sale
        counted = amounts["economic"] + amounts["procurement"] + P["paid_statutory_counted_in_vat_limit"] * amounts["paid_statutory"]
        vat = (economic | paid_stat) & ((counted > VAT_LIMIT_2026) | (economic & (d["u_vat_voluntary"] < P["voluntary_vat"])))
        # the three VAT fields: kinds of activity (a prototype approximation, Z: economic sales and procurement taxable,
        # paid statutory sales unrecognised (the variant), free activity, financial income and non-market revenue out of
        # scope; exemptions by object not modelled), the sales counted in the limit, the status
        kinds_flags = (("taxable", economic | procurement), ("unrecognised_paid_statutory", paid_stat),
                       ("out_of_scope", financial | nonmarket | fees | other | ~has_rev))
        vat_kinds = np.array([",".join(k for k, f in kinds_flags if f[i]) for i in range(records)], dtype=object)
        # own account at the measurement date: the MF 2026 transfer (D, an own account at the transfer; kept to the
        # measurement date: Z) and, if aid_2025_as_own_account, the clubs' 2025 aid (Z) come first; otherwise one
        # probability for an organisation with any premise, whatever their number (Z), and the A / B thresholds only
        # for organisations without any premise
        conf = mf_2026 | (aid_2025 & P["aid_2025_as_own_account"])
        premise = grants_present | (staff > 0) | economic | vat | (aid_2025 & (not P["aid_2025_as_own_account"]))
        thr = np.where(premise, P["premise_account"], P["no_premise_account_B"] if sg in GROUP_B else P["no_premise_account_A"])
        own_account = conf | (d["u_account"] < thr)
        project = grants_present & (d["u_project"] < P["project_ongoing_given_grants"])
        extra = own_account & project & (d["u_project_acc"] < P["separate_project_account"])
        accounts = np.where(own_account, 1 + extra.astype(int), 0)  # own accounts: 0 without an own account
        # who operates the organisation's OWN banking (Z, precedence top-down); "employee" only with employed staff
        uo = d["u_operator"]
        op = np.where(staff == 3, np.where(uo < 0.7, 3, 2),
             np.where(staff == 2, np.where(uo < 0.5, 2, np.where(uo < 0.8, 3, 1)),
             np.where(staff == 1, np.where(uo < 0.6, 2, 1),
             np.where(revenue > 100000, np.where(uo < 0.6, 2, 1), 1))))
        operator = np.where(own_account, op, 0)
        representative = (~own_account & (d["u_representative"] < P["representative_account_ordinary"])) if sg in GROUP_B \
            else np.zeros(records, bool)
        lp = T["1.32"]["rows"][row][1] / 100  # all organisations of the type; an exact count, rising with the latent scale
        loan = _within(u["loan"], np.ones(records, bool), [1 - lp, lp]) == 1
        res[sg] = {"weight": count / records, "revenue": revenue, "has_revenue": has_rev, "activity": act, "staff": staff,
                   "nonmarket": nonmarket, "public": public, "private": private, "grants_present": grants_present,
                   "fees": fees, "other": other, "financial": financial, "procurement": procurement, "economic": economic,
                   "paid_statutory": paid_stat, "amounts": amounts, "confirmed": conf, "mf_2026": mf_2026, "aid_2025": aid_2025,
                   "registered_2026": registered_2026, "opp": opp, "vat": vat, "vat_kinds": vat_kinds,
                   "vat_counted_sales_pln": counted, "operator": operator,
                   "uses_representative_private_account": representative,
                   "premise": premise, "own_account": own_account, "accounts": accounts, "project": project, "loan": loan,
                   "fallback_source": none, "removed_revenue": removed, "calibration": calib, "revenue_before_removal": revenue + removed,
                   "params": P}
    return res
