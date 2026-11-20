#!/usr/bin/env python3
"""B-K2 analysis plan (data/reference/b_k2_outcome_pairs.json, version 0.2): two claims kept apart.

- pair claim (main, one Holm family of four): controlled trials, four versions of the same base profile (LL, HL, LH,
  HH of the pair's two fields, everything else fixed, each version valid); interaction = HH - HL - LH + LL on the
  observed 1-5 ratings: the claim is non-additivity of the ratings on this scale, not of a latent judgement
  (rounding of an additive judgement can make it); mean over base profiles with size classes weighted by their
  share of active firms; sign-flip permutation test; the final decision is pair_family() over exactly the plan's
  four tests: Holm on the main analysis and on the analysis without ratings at 1 or 5, the same sign, and no size
  class lost (a class lost to 'brak podstaw' gives a result restricted to the kept classes, not a success);
- population claim (separate Holm family of four): variant minus independent, attributed to the variant as a whole;
  only in the size classes where the function's pair is interpretable in the case set.

    python3 scripts/b_k2_plan.py [--report data/processed/profile_check_v<version>.json]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
PLAN = os.path.join(ROOT, "data", "reference", "b_k2_outcome_pairs.json")
SIZES = ("0-9", "10-49", "50-249", "250+")
VERSIONS = ("LL", "HL", "LH", "HH")


def load_plan(path: str = PLAN) -> dict:
    """The plan, checked: every function's pair is a coupled pair; every population test's pair is coupled in its
    variant; the levels are categories of their fields; the pair family has one test per function."""
    import profile_check as pc
    from dnba.population import layer2 as l2
    plan = json.load(open(path, encoding="utf-8"))
    funcs = {f["id"]: f for f in plan["functions"]}
    if len(funcs) != len(plan["functions"]) or len(plan["families"]["pair"]["tests"]) != len(funcs):
        raise ValueError("one pair test per function")
    for f in funcs.values():
        if f["pair"] not in pc.PAIR_FIELDS:
            raise ValueError(f"{f['id']}: unknown pair {f['pair']}")
        if re.search(r"najbardziej przydatn|\bpar[ayę]\b|\bwariant", f["text"].lower()):
            raise ValueError(f"{f['id']}: the agent's text must be neutral")
    for t in plan["families"]["population"]["tests"] + plan["families"]["exploratory"]:
        if funcs[t["function"]]["pair"] not in pc.COUPLED_PAIRS.get(t["variant"], ()):
            raise ValueError(f"{t['function']}: its pair is not coupled in {t['variant']}")
    cats = {"overdue_receivables_share": l2.OVERDUE, "overdue_payables_share": l2.OVERDUE,
            "liquidity_declared": l2.LIQUIDITY_CATS, "largest_b2b_client_share": l2.CONCENTRATION}
    for field, lv in plan["levels"].items():
        if field == "note":
            continue
        if lv["L"] not in cats[field] or lv["H"] not in cats[field] or lv["L"] == lv["H"]:
            raise ValueError(f"levels of {field}")
    if not {x for f in funcs.values() for x in pc.PAIR_FIELDS[f["pair"]]} <= set(plan["controls"]):
        raise ValueError("a field of a pair is not among the controls")
    return plan


CONCENTRATION_BANDS = ("6-20", "21-100", "101+")  # the plan: 6 or more B2B clients, so that both levels are free


def pair_versions(profile: dict, firm: dict, pair: str, plan: dict) -> dict | None:
    """The four versions of a base profile for a pair, or None if the profile does not qualify (the plan's
    base_profiles: active; both fields apply; invoice financing: a sales term other than 0 days; concentration
    pairs: 6 or more B2B clients) or any version is not valid (layer2.consistency)."""
    import profile_check as pc
    from dnba.population import layer2 as l2
    fa, fb = pc.PAIR_FIELDS[pair]
    if profile.get(fa) is None or profile.get(fb) is None or profile.get("activity_status") != "active":
        return None
    if pair == "receivables_liquidity" and profile.get("payment_terms_sales_days") == "0":
        return None  # invoice financing: with a 0-day term no invoice is before its due date
    if "largest_b2b_client_share" in (fa, fb) and profile.get("b2b_clients_year") not in CONCENTRATION_BANDS:
        return None  # the consistency rules allow 3-5 clients; the plan asks for 6 or more
    out = {}
    for v in VERSIONS:
        q = {**profile, fa: plan["levels"][fa][v[0]], fb: plan["levels"][fb][v[1]]}
        if l2.consistency(q, firm):
            return None
        out[v] = q
    return out


def interaction(r: dict) -> float:
    """HH - HL - LH + LL on the observed 1-5 ratings: non-additivity of the RATINGS on this scale. Rounding to
    whole points can make it from an additive latent judgement (e.g. 2.1 + 0.3 A + 0.3 B rounds to 2, 2, 2, 3), so it
    is never read as evidence that the agent combines the cues beyond an additive judgement (plan, claims)."""
    return float(r["HH"] - r["HL"] - r["LH"] + r["LL"])


def sign_flip_p(x: np.ndarray, w: np.ndarray, reps: int = 10000, seed: int = 20261114) -> float:
    """Two-sided sign-flip permutation p-value of the weighted mean of x (paired, per base profile)."""
    if len(x) == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    obs = abs(np.average(x, weights=w))
    flips = rng.choice([-1.0, 1.0], size=(reps, len(x)))
    null = np.abs((flips * x) @ w / w.sum())
    return float((1 + (null >= obs - 1e-12).sum()) / (reps + 1))


INVALID = "invalid"  # no valid answer from the model (failed call, malformed answer): never 'brak podstaw'


def pair_test(ratings: list[dict], sizes: list[str], class_share: dict, weights: list[float] | None = None,
              flips: int = 10000, flip_seed: int | None = None) -> dict:
    """RAW result of one pair test (never a final decision: pair_family decides for the family).
    ratings: per base profile {'LL','HL','LH','HH'} -> 1-5, None ('brak podstaw') or INVALID; sizes: its size class;
    class_share: share of the target population per size class; weights: the firm's weight inside its class (its
    weight_active; equal if None), so that a class's mean is the mean over its active firms. Main analysis: profiles
    with all four ratings; robustness: also without profiles with any rating at 1 or 5. Reported: counts and weight
    coverage by class from the initial sample, classes lost to 'brak podstaw' and to invalid answers (apart), and both
    kinds of missing by version with their denominator. flips: sign flips per p-value (the analysis uses the default).
    flip_seed: the seed of the random sign flips; None keeps the analysis's fixed seed (sign_flip_p's default). The
    test is exact over the randomness of the flip set (random transformations with the identity added); for one fixed
    set it need not be, so a simulation of the procedure's size draws a new set per replication (b_k2_power
    --flip-seed per-replication)."""
    weights = list(weights) if weights is not None else [1.0] * len(ratings)

    def run(keep):
        x = np.array([interaction(r) for r, k in zip(ratings, keep) if k])
        sz = [s for s, k in zip(sizes, keep) if k]
        wu = [w for w, k in zip(weights, keep) if k]
        tot_by = {s: sum(w for w, t in zip(wu, sz) if t == s) for s in sorted(set(sz))}
        n_by = {s: sz.count(s) for s in sorted(set(sz))}
        w = np.array([class_share[s] * wi / tot_by[s] for s, wi in zip(sz, wu)]) if sz else np.array([])
        est = float(np.average(x, weights=w)) if len(x) else None
        p = sign_flip_p(x, w, reps=flips) if flip_seed is None else sign_flip_p(x, w, reps=flips, seed=flip_seed)
        return {"interaction": est, "p": p, "n": len(x), "n_by_class": n_by}
    valid = [all(isinstance(r.get(v), int) for v in VERSIONS) for r in ratings]
    no_answer = [any(r.get(v) == INVALID for v in VERSIONS) for r in ratings]
    inner = [c and all(1 < r[v] < 5 for v in VERSIONS) for r, c in zip(ratings, valid)]
    main, rob = run(valid), run(inner)
    initial = {s: sizes.count(s) for s in sorted(set(sizes))}
    answered = {s for s, na in zip(sizes, no_answer) if not na}  # classes with at least one profile fully answered
    target = sum(class_share[s] for s in initial)
    main["weight_coverage"] = sum(class_share[s] for s in main["n_by_class"]) / target if target else 0.0
    lost = sorted(set(initial) - set(main["n_by_class"]))
    main["classes_lost_to_invalid"] = [s for s in lost if s not in answered]
    main["classes_lost"] = [s for s in lost if s in answered]  # lost to 'brak podstaw'
    present = set(main["n_by_class"])
    rob["weight_coverage"] = (sum(class_share[s] for s in rob["n_by_class"]) / sum(class_share[s] for s in present)
                              if present else 0.0)
    rob["classes_lost"] = sorted(present - set(rob["n_by_class"]))
    return {"initial_n_by_class": initial, "main": main, "robustness": rob,
            "no_basis_by_version": {v: {"n": sum(r.get(v) is None for r in ratings), "of": len(ratings)} for v in VERSIONS},
            "invalid_by_version": {v: {"n": sum(r.get(v) == INVALID for r in ratings), "of": len(ratings)} for v in VERSIONS}}


RULES = ("pilot_0.7", "confirmatory_draft_0.8")


def pair_family(results: dict, plan: dict | None = None, alpha: float = 0.05, rule: str = "pilot_0.7") -> dict:
    """The final decision for the pair family. `results` must hold exactly the plan's four tests (keyed by function
    id): a missing, extra or renamed test is refused (ValueError), so that Holm always corrects over the four.
    A test succeeds for the target population only if (1) Holm over the four rejects it on the main analysis, (2)
    Holm over the four rejects it on the robustness analysis, (3) both have the same sign, (4) the main analysis
    keeps every size class of the initial sample (none lost to 'brak podstaw' or to invalid answers) and (5) the robustness analysis keeps
    every class of the main one. If (1)-(3) and (5) hold but classes were lost to 'brak podstaw' or to invalid answers
    (reported apart), the result is reported as restricted to the kept classes ('success_restricted_to'), never as a
    success for the population.
    No direction is fixed in advance. The claim is non-additivity of the ratings on the 1-5 scale.
    rule 'confirmatory_draft_0.8' (a proposal for the confirmatory study, data/reference/b_k2_confirmatory_draft.json; not
    the pilot's rule): a test succeeds if (1) and (4) hold; the robustness analysis is a sensitivity result
    ('sensitivity_agrees': Holm rejects it there too, with the same sign, and it keeps every class of the main
    analysis), not a condition. The default is the pilot's
    rule, with the same output as before the option existed."""
    if rule not in RULES:
        raise ValueError(f"rule {rule!r}, expected one of {RULES}")
    plan = plan or load_plan()
    names = {f["id"] for f in plan["functions"]}
    if set(results) != names:
        raise ValueError(f"the pair family must be exactly the plan's tests {sorted(names)}, got {sorted(results)}")
    main = holm({k: r["main"]["p"] for k, r in results.items()}, alpha)
    rob = holm({k: r["robustness"]["p"] for k, r in results.items()}, alpha)
    out = {}
    for k, r in results.items():
        a, b = r["main"]["interaction"], r["robustness"]["interaction"]
        same_sign = a is not None and b is not None and np.sign(a) == np.sign(b) != 0
        lost_main = r["main"]["classes_lost"] + r["main"].get("classes_lost_to_invalid", [])
        kept_main, kept_rob = not lost_main, not r["robustness"]["classes_lost"]
        core = bool(main[k] and rob[k] and same_sign and kept_rob) if rule == "pilot_0.7" else bool(main[k] and a is not None)
        out[k] = {"holm_main": main[k], "holm_robustness": rob[k], "same_sign": bool(same_sign),
                  "classes_lost_to_no_basis": r["main"]["classes_lost"],
                  "classes_lost_to_invalid": r["main"].get("classes_lost_to_invalid", []),
                  "weight_coverage": r["main"]["weight_coverage"],
                  "classes_lost_in_robustness": r["robustness"]["classes_lost"],
                  "success": core and kept_main,
                  "success_restricted_to": sorted(r["main"]["n_by_class"]) if core and not kept_main else None}
        if rule != "pilot_0.7":
            out[k].update(rule=rule, sensitivity_agrees=bool(rob[k] and same_sign and kept_rob))
    return out


def holm(ps: dict, alpha: float = 0.05) -> dict:
    """Holm step-down: name -> rejected."""
    order = sorted(ps, key=ps.get)
    out, stop = {}, False
    for i, k in enumerate(order):
        stop = stop or ps[k] > alpha / (len(order) - i)
        out[k] = not stop
    return out


def population_contrasts(plan: dict, achieved: dict) -> list[dict]:
    """Per population test (and exploratory one): the size classes kept and dropped, from achieved_coupling's
    'pairs' block (population or case set)."""
    funcs = {f["id"]: f for f in plan["functions"]}
    out = []
    fam = plan["families"]
    for role, tests in (("population", fam["population"]["tests"]), ("exploratory", fam["exploratory"])):
        for t in tests:
            pair = funcs[t["function"]]["pair"]
            cells = achieved["pairs"].get(pair, {}).get(t["variant"], {})  # a pair or class without data: not interpretable
            keep = [s for s in SIZES if s in cells and cells[s].get("interpretable")]
            out.append({"function": t["function"], "pair": pair, "variant": t["variant"], "role": role,
                        "keep": keep, "drop": [s for s in SIZES if s not in keep], "dropped_whole": not keep})
    return out


def main() -> None:
    import profile_check as pc
    from dnba.population import layer2 as l2
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", default=os.path.join(ROOT, "data", "processed", f"profile_check_v{l2.VERSION}.json"))
    a = ap.parse_args()
    rep = json.load(open(a.report, encoding="utf-8"))["part2_variants"]["achieved_coupling"]
    plan = load_plan()
    print("pair family (controlled trials, Holm over four):")
    for f in plan["functions"]:
        print(f"  {f['id']:24s} {f['pair']}")
    print(f"population family (rule: lower bound >= {pc.MIN_COUPLING}, mean >= {pc.MIN_COPULA_RATIO} of the copula's; "
          f"{os.path.relpath(a.report, ROOT)}):")
    for c in population_contrasts(plan, rep):
        print(f"  {c['role']:11s} {c['function']:24s} {c['pair']:26s} {c['variant']:15s} keep {c['keep']} drop {c['drop']}")


if __name__ == "__main__":
    main()
