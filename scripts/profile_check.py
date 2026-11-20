#!/usr/bin/env python3
"""Checks of layer-2 profiles before agents are run, and a comparison of the dependence variants.

Part 1, profile checks, four kinds kept apart:
- contradiction (error, the profile fails): facts that cannot hold together, e.g. one B2B client and a largest
  client below 100% of B2B sales, overdue receivables without B2B sales; criterion: zero cases;
- inadequate field (error): a field that does not apply carries a value (a sales term without B2B clients),
  or a field that applies is missing;
- period and scope (only where the definitions allow a comparison): yearly B2B clients against the B2B part
  of the active customers of a month (a lower bound: customers of a month include consumers with an
  invoice); yearly clients against 12 x the B2B invoices of a typical month is an approximation (a scenario,
  not an observed yearly volume), reported as a flag, not an error; revenue against invoiced sales is not
  checked: the revenue includes unvoiced sales and the invoice values are not in the profile;
- atypical combination (flag for review, never rejected): a very high mean value of a B2B invoice (B2B sales
  = revenue x (1 - B2C share), all invoiced under A0-28, divided by the B2B invoices of a year) and a very high
  B2B sale per client (B2B sales over the lower bound of the client band). No mean invoice value is taken
  from the whole revenue, which includes sales without an invoice.
Reported: counts, weighted shares (active firms inside the GUS DPN scope, weight_active) and a few synthetic examples (layer-2 fields,
size and section only).

Part 2, the dependence variants (independent, mrit_rank, stress_coupled) on the same firms, weights and
generator version: weighted shares of every category of each field in each variant, the largest difference
from independent, overall, by size, by broad section and by size x PKD section, with the number and the effective number
(Kish) of firms; the dependences obtained, as rank correlations of the banded values (the overdue shares
also on the grid of the published MRiT table). Since 0.6.15 the variants are drawn as for comparing agents
(layer2.draw_population): the values of the coupled fields are re-assigned by rank inside cells of size x section x
weight class, so every share over whole cells (overall, by size, by broad section, by size x section) equals
'independent' and the differences are zero; the tolerances (TOL_*, fixed before any agent is compared) stay as a
check. Before 0.6.15 the per-firm copula kept only each field's theoretical distribution, and the tolerances were
exceeded (1.21 / 3.63 p.p.). mrit_rank is meant to reproduce the MRiT dependence in one segment named beforehand:
firms of 250+ persons, rank correlation on the MRiT grid within TOL_RANK of the MRiT value; not in every small
cell.

    python3 scripts/profile_check.py
"""
import argparse
import collections
import csv
import gzip
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from dnba.population import layer2 as l2  # noqa: E402

POP = "data/population/firms_regon_2026q2_krs_strata-size_class-min500_n10000_seed42.csv"
AUTH = "data/population/authority_regon_2026q2_krs_strata-size_class-min500_n10000_seed42_krs20261107.jsonl.gz"
SIZES = ("0-9", "10-49", "50-249", "250+")
BROAD = {**{s: "industry" for s in "BCDE"}, "F": "construction", "G": "trade", "H": "transport", "I": "accommodation",
         "J": "information", "M": "professional", "N": "administrative"}
FIELDS = ("b2b_clients_year", "largest_b2b_client_share", "payment_terms_sales_days", "payment_terms_purchases_days",
          "overdue_receivables_share", "overdue_payables_share", "liquidity_declared")
ORDERED = {"b2b_clients_year": l2.CLIENT_BANDS, "largest_b2b_client_share": l2.CONCENTRATION,
           "payment_terms_sales_days": l2.TERMS, "payment_terms_purchases_days": l2.TERMS,
           "overdue_receivables_share": l2.OVERDUE, "overdue_payables_share": l2.OVERDUE, "liquidity_declared": l2.LIQUIDITY_CATS}
CLIENT_LOW = {"0": 0, "1": 1, "2": 2, "3-5": 3, "6-20": 6, "21-100": 21, "101+": 101}
CLIENT_HIGH = {"0": 0, "1": 1, "2": 2, "3-5": 5, "6-20": 20, "21-100": 100, "101+": 10**9}
# flags for review (atypical, not errors): a mean B2B invoice over FLAG_INVOICE_X times the size median of A0-28,
# and a B2B sale per client over FLAG_PER_CLIENT_PLN a year
FLAG_INVOICE_X = 20
FLAG_PER_CLIENT_PLN = {"0-9": 5e6, "10-49": 50e6, "50-249": 200e6, "250+": 1e9}
# tolerances fixed before any comparison of agents (part 2)
TOL_OVERALL = 0.01   # largest difference of a weighted share from independent, all active firms
TOL_SIZE = 0.03      # ... within a size class
TOL_RANK = 0.05      # mrit_rank, 250+: rank correlation on the MRiT grid vs the MRiT value
MIN_GROUP_N = 10     # groups with fewer firms in the sample are reported, flagged small_n and kept out of the summaries
# 0.6.15 (rules set before any agent is run, after the achieved couplings of the population were seen): the agent
# comparison uses the assignments replicate = REPLICATES (never replicate 0) and averages over them; a coupling counts
# as achieved, and contrasts that rest on it as interpretable, where (a) the lower bound of the 95% interval of the
# mean over REPLICATES of (variant - independent), in weighted rank correlation of the banded values, is at least
# MIN_COUPLING, and (b) that mean is at least MIN_COPULA_RATIO of what the per-firm copula achieves in the same size
# class. The interval is a paired bootstrap over firms stratified by the rank cells (size x section x weight class).
# The same rule applies to the case set the agents see (achieved_coupling on the case firms).
REPLICATES = (1, 2, 3, 4, 5)
MIN_COUPLING = 0.05
MIN_COPULA_RATIO = 0.5
BOOT_REPS = 2000
COUPLED_PAIRS = {"mrit_rank": ("overdue_rec_pay",),
                 "stress_coupled": ("overdue_rec_pay", "concentration_liquidity", "receivables_liquidity", "concentration_receivables")}
VAT_EXEMPTION_2024 = 200_000  # VAT act art. 113 subjective exemption limit for revenue of 2024 (revenue_year_pln is 2024)
MRIT_GRID = (("0", "(0,5%]"), ("(5,15%]",), ("(15,30%]",), (">30%",))


def check(firm: dict, p: dict) -> dict[str, list[str]]:
    """{'contradiction': [...], 'inadequate': [...], 'scope': [...], 'atypical': [...]} for one profile."""
    out = {"contradiction": [], "inadequate": [], "scope": [], "atypical": []}
    out["contradiction"] += l2.consistency(p, firm)
    n, conc = p.get("b2b_clients_year"), p.get("largest_b2b_client_share")
    sales_fields = ("largest_b2b_client_share", "payment_terms_sales_days", "overdue_receivables_share")
    suspended = p.get("activity_status") == "suspended"
    if suspended:
        out["inadequate"] += [f"{k} given for a suspended firm" for k in ("b2b_clients_year",) + sales_fields if p.get(k) is not None]
    elif n is None:
        out["inadequate"].append("b2b_clients_year missing for an operating firm")
    elif n == "0":
        out["inadequate"] += [f"{k} given without B2B clients" for k in sales_fields if p.get(k) is not None]
        if p.get("b2c_share_of_sales", 1) < 1:
            out["contradiction"].append("sales to firms without B2B clients")
    else:
        out["inadequate"] += [f"{k} missing with B2B clients" for k in sales_fields if p.get(k) is None]
        if conc is not None and n == "1" and conc != ">=75%":
            out["contradiction"].append("one B2B client below 100% of B2B sales")
        if conc is not None and n == "2" and conc in ("<25%", "25-<50%"):
            out["contradiction"].append("two B2B clients, the largest below 50%")
        if conc == "<25%" and CLIENT_HIGH[n] < 5:
            out["contradiction"].append("largest client below 25% with fewer than 5 clients")
    reasons = p.get("null_reasons")
    if reasons is not None:
        nulls = {k for k, v in p.items() if v is None}
        out["inadequate"] += [f"{k} null without a reason" for k in sorted(nulls - set(reasons))]
        out["inadequate"] += [f"{k} has a null reason but a value" for k in sorted(set(reasons) - nulls)]
    if p.get("overdue_receivables_share") not in (None, "0") and n in (None, "0"):
        out["contradiction"].append("overdue receivables without a basis")
    for k in ("payment_terms_purchases_days", "overdue_payables_share"):
        if p.get(k) is None and not suspended:  # 0.6.11: unknown for a suspended firm (its past 12 months)
            out["inadequate"].append(f"{k} missing")
    if n not in (None, "0"):
        b2b_month = p["invoices_issued_month"] - p.get("invoices_b2c_month", 0)
        b2b_customers = p["customers_active_month"] - p.get("invoices_b2c_month", 0)
        if CLIENT_HIGH[n] < b2b_customers:
            out["scope"].append("fewer B2B clients in a year than B2B customers in a month")
        if b2b_month > 0 and CLIENT_LOW[n] > 12 * b2b_month:
            out["scope"].append("flag: more B2B clients in a year than 12 x B2B invoices of a typical month (approximation)")
        b2b_sales = p.get("revenue_annual_pln", p["revenue_year_pln"]) * (1 - p["b2c_share_of_sales"])  # current yearly scale
        size = firm["size_class"]
        if b2b_month > 0 and b2b_sales / (12 * b2b_month) > FLAG_INVOICE_X * l2.INV_VALUE_B2B[size] * l2.INV_SCALE:
            out["atypical"].append("mean B2B invoice over 20x the size median")
        if CLIENT_LOW[n] > 0 and b2b_sales / CLIENT_LOW[n] > FLAG_PER_CLIENT_PLN[size]:
            out["atypical"].append("B2B sale per client above the flag level")
        # the VAT exemption limit binds the revenue of 2024; a firm founded after 2024 is not assessed (0.6.13)
        # art. 113 ust. 9: a firm that started in 2024 has the limit in proportion to its time of activity
        vat_limit = VAT_EXEMPTION_2024 * p.get("months_active_2024", 12) / 12
        if p.get("has_separate_business_account") is False and (p["revenue_year_pln"] or 0) > vat_limit and b2b_month > 0 \
                and b2b_sales / (12 * b2b_month) > 15_000:
            out["atypical"].append("no business account, VAT-taxable revenue and B2B invoices over PLN 15,000 (white-list friction)")
    return out


def eff_n(w):
    w = np.asarray(w, dtype=float)
    return float(w.sum() ** 2 / (w ** 2).sum()) if len(w) and (w ** 2).sum() > 0 else 0.0


PAIR_FIELDS = {"overdue_rec_pay": ("overdue_receivables_share", "overdue_payables_share"),
               "concentration_liquidity": ("largest_b2b_client_share", "liquidity_declared"),
               "receivables_liquidity": ("overdue_receivables_share", "liquidity_declared"),
               "concentration_receivables": ("largest_b2b_client_share", "overdue_receivables_share")}


def _order(field):
    return {"largest_b2b_client_share": l2.CONCENTRATION, "liquidity_declared": l2.LIQUIDITY_CATS}.get(field, l2.OVERDUE)


def rho_tables(counts: np.ndarray, x: np.ndarray, y: np.ndarray, kx: int, ky: int) -> np.ndarray:
    """Weighted rank correlation of two banded variables (mid-ranks of the weighted bands), for every row of
    `counts` (B x n firm weights, e.g. weight x bootstrap count): computed from the B joint tables at once."""
    onehot = np.zeros((len(x), kx * ky))
    onehot[np.arange(len(x)), x * ky + y] = 1.0
    t = (counts @ onehot).reshape(-1, kx, ky)
    tot = t.sum(axis=(1, 2))
    px, py = t.sum(axis=2), t.sum(axis=1)
    rx, ry = np.cumsum(px, axis=1) - px / 2, np.cumsum(py, axis=1) - py / 2
    mx, my = (px * rx).sum(1) / tot, (py * ry).sum(1) / tot
    dx, dy = rx - mx[:, None], ry - my[:, None]
    cov = np.einsum("bij,bi,bj->b", t, dx, dy) / tot
    vx, vy = (px * dx ** 2).sum(1) / tot, (py * dy ** 2).sum(1) / tot
    with np.errstate(invalid="ignore", divide="ignore"):
        return cov / np.sqrt(vx * vy)


def interpretable(lo: float, mean: float, cop: float | None) -> bool:
    """The rule fixed on 14.11.2026, on unrounded values: the lower bound at least MIN_COUPLING and the mean at least
    MIN_COPULA_RATIO of the per-firm copula's coupling."""
    return bool(lo >= MIN_COUPLING and cop is not None and cop > 0 and mean / cop >= MIN_COPULA_RATIO)


def achieved_coupling(firms: list, aw: dict, wclass: dict, base: list, reps: dict, copula: dict, subset=None) -> dict:
    """Achieved coupling per coupled pair x size x variant: mean over REPLICATES of (variant - independent), paired
    bootstrap over firms stratified by rank cell (95% interval), the per-firm copula's (variant - independent) and the
    ratio, and the verdict of the pre-set rule. `subset`: firm ids (e.g. the case set the agents see); None = all."""
    rng = np.random.default_rng(20261114)
    keep = [i for i, f in enumerate(firms) if subset is None or f["firm_id"] in subset]
    out = {"replicates": list(REPLICATES), "bootstrap_reps": BOOT_REPS, "min_coupling": MIN_COUPLING,
           "min_copula_ratio": MIN_COPULA_RATIO, "subset_firms": None if subset is None else len(keep), "pairs": {}}
    # in a case set the shares of the coupled fields differ between the variants (they are kept only over whole cells
    # of the population): the largest weighted difference per field and variant, mean over replicates
    wk = np.array([aw[firms[i]["firm_id"]] for i in keep])
    out["share_differences"] = {}
    for dep, rr in reps.items():
        for field in l2.COUPLED_FIELDS[dep]:
            cats = _order(field)
            def shares(prof):
                v = np.array([cats.index(prof[i][field]) if prof[i][field] is not None else -1 for i in keep])
                tot = wk[v >= 0].sum()
                return np.array([wk[v == k].sum() / tot if tot else 0.0 for k in range(len(cats))])
            s0 = shares(base)
            out["share_differences"].setdefault(dep, {})[field] = round(float(np.mean([np.abs(shares(r) - s0).max() for r in rr])), 4)
    for dep, whats in COUPLED_PAIRS.items():
        for what in whats:
            fa, fb = PAIR_FIELDS[what]
            oa, ob = _order(fa), _order(fb)
            for size in ("0-9", "10-49", "50-249", "250+"):
                idx = [i for i in keep if firms[i]["size_class"] == size and base[i][fa] is not None and base[i][fb] is not None]
                if len(idx) < MIN_GROUP_N:
                    continue
                w = np.array([aw[firms[i]["firm_id"]] for i in idx])
                cells = collections.defaultdict(list)
                for j, i in enumerate(idx):
                    cells[(firms[i]["pkd_section"], wclass[firms[i]["firm_id"]])].append(j)
                boot = np.zeros((BOOT_REPS, len(idx)))
                for js in cells.values():
                    boot[:, js] = rng.multinomial(len(js), [1 / len(js)] * len(js), size=BOOT_REPS)
                counts = np.vstack([w[None, :], boot * w[None, :]])  # row 0: the sample itself

                def rho(prof):
                    x = np.array([oa.index(prof[i][fa]) for i in idx])
                    y = np.array([ob.index(prof[i][fb]) for i in idx])
                    return rho_tables(counts, x, y, len(oa), len(ob))
                r0 = rho(base)
                rv = [rho(rr) for rr in reps[dep]]
                diffs = np.mean([r - r0 for r in rv], axis=0)
                cop = float((rho(copula[dep]) - r0)[0])
                lo, hi = np.nanpercentile(diffs[1:], [2.5, 97.5])
                mean = float(diffs[0])
                ratio = None if not cop or cop <= 0 else mean / cop  # compared unrounded (audit of 0.6.16)
                out["pairs"].setdefault(what, {}).setdefault(dep, {})[size] = {
                    "mean_diff": round(mean, 3), "ci95": [round(float(lo), 3), round(float(hi), 3)],
                    "independent_rank_corr": round(float(r0[0]), 3), "variant_rank_corr": round(float(np.mean([r[0] for r in rv])), 3),
                    "per_replicate": [round(float((r - r0)[0]), 3) for r in rv],
                    "copula_diff": round(cop, 3), "copula_ratio": None if ratio is None else round(ratio, 2), "n": len(idx),
                    "interpretable": interpretable(float(lo), mean, cop)}
    return out


def wrank(rows, digits: int | None = 3):
    """Weighted rank correlation of two banded variables: rows (w, x, y), mid-ranks of the weighted bands."""
    if len(rows) < 3:
        return None
    w = np.array([r[0] for r in rows], dtype=float)

    def mid(col):
        x = np.array([r[col] for r in rows])
        out, cum = np.zeros(len(x)), 0.0
        for v in sorted(set(x)):
            m = x == v
            out[m] = cum + w[m].sum() / 2
            cum += w[m].sum()
        return out
    a, b = mid(1), mid(2)
    ma, mb = np.average(a, weights=w), np.average(b, weights=w)
    va, vb = np.average((a - ma) ** 2, weights=w), np.average((b - mb) ** 2, weights=w)
    if va == 0 or vb == 0:
        return None
    r = float(np.average((a - ma) * (b - mb), weights=w) / np.sqrt(va * vb))
    return r if digits is None else round(r, digits)


def mrit_value():
    m = json.load(open(os.path.join(ROOT, "data/processed/mrit_delays_2025.json"), encoding="utf-8"))["all_firms"]["joint_firms"]
    rows = []
    for key, n in m["cells"].items():
        r, c = key.split("|")
        rows.append((n, m["rows_overdue_in"].index(r), m["cols_overdue_out"].index(c)))
    return wrank(rows, digits=None)  # 0.6.16: unrounded, compared unrounded; rounded only in the report


def grid(cat):
    return next(i for i, g in enumerate(MRIT_GRID) if cat in g)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--firms", default=os.path.join(ROOT, POP))
    ap.add_argument("--authority", default=os.path.join(ROOT, AUTH))
    ap.add_argument("--examples", type=int, default=3)
    a = ap.parse_args()
    firms = list(csv.DictReader(open(a.firms, encoding="utf-8")))
    auth = {x["firm_id"]: x for x in map(json.loads, gzip.open(a.authority))}
    # active firms inside the GUS DPN scope (column scope of calibrate_active.py), as in scripts/layer2_check.py
    aw = {r["firm_id"]: float(r["weight_active"]) for r in csv.DictReader(open(os.path.splitext(a.firms)[0] + ".active_weights.csv"))
          if r.get("scope", "gus_dpn") == "gus_dpn"}
    kinds = ("contradiction", "inadequate", "scope", "atypical")
    viol = {d: {k: collections.Counter() for k in kinds} for d in l2.DEPENDENCE}
    vw = {d: {k: collections.Counter() for k in kinds} for d in l2.DEPENDENCE}
    examples = collections.defaultdict(list)
    shares = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))  # (dep, group) -> field -> cat
    weights = collections.defaultdict(list)
    pairs = collections.defaultdict(list)
    total_w = 0.0
    # 0.6.15: the variants as used for comparing agents, drawn for the population: values of the coupled fields
    # re-assigned by rank inside cells of size x section x weight class, so that the shares stay exactly as in
    # 'independent' (layer2.draw_population)
    active = [f for f in firms if aw.get(f["firm_id"], 0.0) > 0]
    wclass = {f["firm_id"]: round(aw[f["firm_id"]], 6) for f in active}
    indep = l2.draw_population(active, auth, "independent", wclass)
    reps = {dep: [l2.draw_population(active, auth, dep, wclass, replicate=r, base=indep) for r in REPLICATES]
            for dep in l2.DEPENDENCE if dep != "independent"}
    prof = {"independent": indep, **{dep: rr[0] for dep, rr in reps.items()}}  # part 1 and the share tables: replicate 1
    for i, f in enumerate(active):
        w = aw[f["firm_id"]]
        total_w += w
        p0 = prof["independent"][i]
        groups = ("all", f"size|{f['size_class']}", f"section|{BROAD.get(f['pkd_section'], 'other')}",
                  f"size_section|{f['size_class']}|{f['pkd_section']}",
                  # 0.6.15: cuts that are not unions of the rank cells, reported without a tolerance
                  f"other_persons|{f['size_class']}|{p0['persons_working']}", f"other_books|{f['size_class']}|{p0['bookkeeping_form']}",
                  f"other_form|{f.get('form_group', '')}", f"other_clients|{f['size_class']}|{p0['b2b_clients_year']}",
                  f"other_terms|{f['size_class']}|{p0['payment_terms_sales_days']}", f"other_age|{f['size_class']}|{p0['firm_age']}")
        for dep in l2.DEPENDENCE:
            p = prof[dep][i]
            for kind, msgs in check(f, p).items():
                for m in msgs:
                    viol[dep][kind][m] += 1
                    vw[dep][kind][m] += w
                    if len(examples[(dep, kind, m)]) < a.examples:
                        examples[(dep, kind, m)].append({"size_class": f["size_class"], "pkd_section": f["pkd_section"],
                                                         **{k: p.get(k) for k in ("activity_status", "revenue_year_pln", "b2c_share_of_sales",
                                                                                  "invoices_issued_month", "invoices_b2c_month", "customers_active_month") + FIELDS}})
            for g in groups:
                if dep == "independent":
                    weights[g].append(w)
                for k in FIELDS:
                    shares[(dep, g)][k][str(p.get(k))] += w
            if p["overdue_receivables_share"] is not None:
                o = l2.OVERDUE
                pairs[(dep, "overdue_rec_pay", f["size_class"])].append((w, o.index(p["overdue_receivables_share"]), o.index(p["overdue_payables_share"])))
                pairs[(dep, "overdue_rec_pay_mrit_grid", f["size_class"])].append((w, grid(p["overdue_receivables_share"]), grid(p["overdue_payables_share"])))
                pairs[(dep, "concentration_liquidity", f["size_class"])].append((w, l2.CONCENTRATION.index(p["largest_b2b_client_share"]), l2.LIQUIDITY_CATS.index(p["liquidity_declared"])))
                pairs[(dep, "receivables_liquidity", f["size_class"])].append((w, o.index(p["overdue_receivables_share"]), l2.LIQUIDITY_CATS.index(p["liquidity_declared"])))
                pairs[(dep, "concentration_receivables", f["size_class"])].append((w, l2.CONCENTRATION.index(p["largest_b2b_client_share"]), o.index(p["overdue_receivables_share"])))

    # part 1
    part1 = {}
    for dep in l2.DEPENDENCE:
        part1[dep] = {kind: {m: {"n": viol[dep][kind][m], "weighted_share": round(vw[dep][kind][m] / total_w, 6),
                                 "examples": examples[(dep, kind, m)]} for m in viol[dep][kind]} for kind in kinds}
    passed = all(not part1[d]["contradiction"] and not part1[d]["inadequate"] for d in l2.DEPENDENCE)
    # part 2
    part2 = {"tolerances": {"overall": TOL_OVERALL, "size": TOL_SIZE, "rank_250plus": TOL_RANK}, "groups": {}, "dependence": {}}
    worst = {"overall": 0.0, "size": 0.0}
    for g in sorted(weights):
        res = {"n": len(weights[g]), "effective_n": round(eff_n(weights[g]), 1), "fields": {}, "small_n": len(weights[g]) < MIN_GROUP_N}
        for k in FIELDS:
            ref = shares[("independent", g)][k]
            tot = sum(ref.values())
            # 0.6.16 (audit of 0.6.14): differences and tolerances on unrounded shares; rounding only for the report
            raw = {d: {c: v / tot for c, v in shares[(d, g)][k].items()} for d in l2.DEPENDENCE}
            per = {d: {c: round(v, 4) for c, v in x.items()} for d, x in raw.items()}
            cats = sorted(set().union(*[set(v) for v in raw.values()]))  # sorted: reproducible ties
            diff, in_se, no_se, diff_raw = {}, {}, {}, {}
            for d in l2.DEPENDENCE:
                if d == "independent":
                    continue
                gaps = {c: abs(raw[d].get(c, 0) - raw["independent"].get(c, 0)) for c in cats}
                diff_raw[d] = max(gaps.values())
                diff[d] = round(diff_raw[d], 4)
                ratios = []
                for c, gap in gaps.items():  # the largest ratio over categories, not the ratio of the largest gap
                    q = raw["independent"].get(c, 0)
                    n_eff = max(res["effective_n"], 1)
                    if q <= 0 and gap > 0:  # zero base share: counted apart, and kept in the measure with a pseudo-count
                        no_se[d] = no_se.get(d, 0) + 1
                    q = min(max(q, 0.5 / n_eff), 1 - 0.5 / n_eff)  # pseudo-count of half a firm (0.6.14)
                    se = (2 * q * (1 - q) / n_eff) ** 0.5  # two draws of the same firms, as if independent
                    if se > 0:
                        ratios.append(gap / se)
                in_se[d] = round(max(ratios), 2) if ratios else None
            res["fields"][k] = {"shares": per, "max_diff_from_independent": diff, "max_diff_in_se": in_se,
                                "categories_without_se": no_se}
            if res["small_n"]:
                pass  # groups under MIN_GROUP_N firms are reported but kept out of the summaries: noise
            elif g == "all":
                worst["overall"] = max(worst["overall"], *diff_raw.values())
            elif g.startswith("size|"):
                worst["size"] = max(worst["size"], *diff_raw.values())
        part2["groups"][g] = res
    for (dep, what, size), rows in sorted(pairs.items()):
        part2["dependence"].setdefault(what, {}).setdefault(dep, {})[size] = {"rank_corr": wrank(rows), "n": len(rows), "effective_n": round(eff_n([r[0] for r in rows]), 1)}
    m = mrit_value()
    copula = {dep: [l2.draw(f, auth.get(f["firm_id"]), dependence=dep) for f in active] for dep in reps}
    part2["achieved_coupling"] = achieved_coupling(active, aw, wclass, indep, reps, copula)
    part2["changed_profiles"] = {"note": f"weighted share of firms whose coupled fields differ from independent, mean over replicates {list(REPLICATES)}",
                                 **{dep: round(float(np.mean([sum(aw[f["firm_id"]] * any(q.get(k) != p.get(k) for k in l2.COUPLED_FIELDS[dep])
                                                                  for f, p, q in zip(active, indep, rr)) for rr in reps[dep]])) / total_w, 4)
                                    for dep in reps}}
    # MRiT segment: mean over the replicates used for agents, rank correlation on the MRiT grid, 250+
    seg = [i for i, f in enumerate(active) if f["size_class"] == "250+" and indep[i]["overdue_receivables_share"] is not None]
    wseg = np.array([[aw[active[i]["firm_id"]] for i in seg]])
    got = float(np.mean([rho_tables(wseg, np.array([grid(rr[i]["overdue_receivables_share"]) for i in seg]),
                                    np.array([grid(rr[i]["overdue_payables_share"]) for i in seg]), len(MRIT_GRID), len(MRIT_GRID))[0]
                         for rr in reps["mrit_rank"]]))
    part2["mrit_segment"] = {"segment": "250+", "mrit_on_grid": None if m is None else round(m, 3),
                             "mrit_rank_on_grid": round(got, 3),
                             "within_tolerance": None if m is None else abs(got - m) <= TOL_RANK,  # unrounded (0.6.16)
                             "note": "the tolerance applies to mrit_rank only; stress_coupled adds a common factor and is not held to it"}
    part2["marginals"] = {"largest_diff_overall": round(worst["overall"], 4), "within_overall": worst["overall"] <= TOL_OVERALL,
                          "largest_diff_by_size": round(worst["size"], 4), "within_size": worst["size"] <= TOL_SIZE,
                          "largest_diff_in_se": max((x for k, g in part2["groups"].items() if not g["small_n"] and not k.startswith("other_")
                                                     for f in g["fields"].values() for x in f["max_diff_in_se"].values() if x is not None), default=None),
                          "largest_diff_other_cuts": max((x for k, g in part2["groups"].items() if not g["small_n"] and k.startswith("other_")
                                                          for f in g["fields"].values() for x in f["max_diff_from_independent"].values()), default=None),
                          "note": "since 0.6.15 the variants keep the generated shares by construction in every group made of whole cells (size x section x weight class): overall, size, broad section, size x section; other cuts (persons working, bookkeeping, form group, client band) are reported without a tolerance and keep some drift; "
                                  "max_diff_in_se: the difference over sqrt(2 p (1 - p) / effective n), a conservative scale of that noise; by section and by size x section reported, no tolerance"}
    out = {"generator": l2.VERSION, "population": os.path.basename(a.firms), "weights": "weight_active, scope gus_dpn", "active_firms_in_sample": len(weights["all"]),
           "method": __doc__.split("\n\n")[1].strip(), "part1_profiles": part1, "part1_passed": passed, "part2_variants": part2}
    path = os.path.join(ROOT, "data", "processed", f"profile_check_v{l2.VERSION}.json")
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for dep in l2.DEPENDENCE:
        print(f"== {dep}")
        for kind in kinds:
            for msg, v in part1[dep][kind].items():
                print(f"  {kind:13s} {v['n']:6d}  {100 * v['weighted_share']:7.3f}%  {msg}")
    print("part 1 passed (no contradiction, no inadequate field):", passed)
    print("marginals:", part2["marginals"])
    print("MRiT segment:", part2["mrit_segment"])
    print("changed profiles (weighted share):", part2["changed_profiles"])
    for what, by in part2["achieved_coupling"]["pairs"].items():
        print("achieved", what, {d: {s: (v["mean_diff"], v["ci95"], v["copula_ratio"], v["interpretable"]) for s, v in x.items()} for d, x in by.items()})
    for what in ("overdue_rec_pay_mrit_grid", "concentration_liquidity", "receivables_liquidity"):
        print(what, {d: {s: v["rank_corr"] for s, v in by.items()} for d, by in part2["dependence"].get(what, {}).items()})
    print("written:", path)


if __name__ == "__main__":
    main()
