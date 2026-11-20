#!/usr/bin/env python3
"""SAFE validation: an assumption-based (Z) field of the synthetic population against its nearest SAFE question (D).

Now one pair: overdue_receivables_share (A0-37, status "assumption") against SAFE qa21 (problems due to late payments
from private entities). The two are different measures with different periods: a share of the value of B2B sales
invoices of the last 12 months paid late (A0-37) versus a reported problem caused by late payments in the previous
and current quarter (qa21), on different bases (firms with B2B sales vs all eligible SAFE firms). A firm with any late-paid invoice need not report a problem,
so the synthetic share above 0 is expected to be at least the SAFE "yes"; the band "over 30%" of value is not a
counterpart of "yes, regularly" (regular small delays or one large delay). A contextual table, not a test, and no
basis for re-tuning A0-37: tuning to the already seen 2026Q3 shares would be calibration, not validation. The comparison is by size class; the synthetic side is weighted by
weight_active among firms with a known value (B2B sales in the window), the SAFE side is unweighted within the size
class (all firms of the class, so firms without B2B sales count as answers too), "don't know" left out. Wilson 95%
intervals on the SAFE shares (sampling only). Aggregates only.

    python3 scripts/safe_assumption_check.py --audit data/processed/safe_validation_audit_v0.1.json \
        --profiles data/processed/b_runs/b_k2_pilot_qwen3.8-27b/profiles.jsonl.gz \
        --frame data/population/firms_regon_2026q2_krs_strata-size_class-min500_n10000_seed42.csv \
        --weights data/population/firms_regon_2026q2_krs_strata-size_class-min500_n10000_seed42.active_weights.csv \
        --out data/processed/safe_assumption_check_v0.1.json
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

VERSION = "0.1"
SIZE_SAFE = {"1-9": "0-9", "10-49": "10-49", "50-249": "50-249", "250+": "250+"}


def wilson(k: int, n: int, z: float = 1.96) -> list:
    if not n:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def sha(p: str) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("audit", "profiles", "frame", "weights", "out"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    audit = json.loads(Path(a.audit).read_text())
    w = {r["firm_id"]: float(r["weight_active"]) for r in csv.DictReader(open(a.weights, encoding="utf-8"))}
    size = {r["firm_id"]: r["size_class"] for r in csv.DictReader(open(a.frame, encoding="utf-8"))}
    syn = defaultdict(lambda: defaultdict(float))
    version = set()
    with gzip.open(a.profiles, "rt", encoding="utf-8") as f:
        for ln in f:
            x = json.loads(ln)
            if x["dependence"] != "independent" or w.get(x["firm_id"], 0) <= 0:
                continue
            p = x["profile"]
            version.add(p.get("layer2_version"))
            g = syn[size[x["firm_id"]]]
            v, wt = p["overdue_receivables_share"], w[x["firm_id"]]
            g["firms"] += 1
            if v is None:
                g["unknown_or_na_w"] += wt
                continue
            g["known_w"] += wt
            g["above_0_w"] += wt * (v != "0")
            g["over_30_w"] += wt * (v == ">30%")
    out = {"script": f"safe_assumption_check {VERSION}", "audit_sha256": sha(a.audit), "profiles_sha256": sha(a.profiles),
           "layer2_version": sorted(version), "pair": {"synthetic": "overdue_receivables_share (A0-37, assumption)",
                                                       "safe": "qa21 (late payments from private entities)"},
           "synthetic_by_size": {}, "safe_by_wave_and_size": {}}
    for s, g in sorted(syn.items()):
        out["synthetic_by_size"][s] = {"firms": int(g["firms"]), "share_above_0": round(g["above_0_w"] / g["known_w"], 4),
                                       "share_over_30": round(g["over_30_w"] / g["known_w"], 4),
                                       "share_not_applicable_or_unknown": round(g["unknown_or_na_w"] / (g["known_w"] + g["unknown_or_na_w"]), 4)}
    for wave, d in audit["waves"].items():
        for s, c in d["answers_by_size"].items():
            n = sum(v for k, v in c.items() if k != "dont_know")
            yes = c.get("yes_regularly", 0) + c.get("yes_occasionally", 0)
            reg = c.get("yes_regularly", 0)
            out["safe_by_wave_and_size"].setdefault(wave, {})[SIZE_SAFE[s]] = {
                "n": n, "share_yes": round(yes / n, 4), "share_yes_ci95": wilson(yes, n),
                "share_regularly": round(reg / n, 4), "share_regularly_ci95": wilson(reg, n)}
    out["note"] = ("different measures (value share paid late vs a reported problem); SAFE 1-9 covers only firms with "
                   "employees, the synthetic 0-9 all firms; SAFE unweighted within the size class")
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(out["synthetic_by_size"], ensure_ascii=False))


if __name__ == "__main__":
    main()
