#!/usr/bin/env python3
"""SAFE validation: print every number of the paper's tables from the aggregate reports (no API, no SAFE microdata).

Reads data/processed/ and data/reference/ only. It also checks that the confirmatory report was made under the frozen
settings that are in this archive: the frozen file's, the plan's and the analysis script's sha256 recorded in the
report equal the files here.

    python3 scripts/safe_paper_numbers.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(name: str) -> dict:
    return json.loads((P / name).read_text())


def main() -> None:
    t = load("safe_test_v0.1.json")
    checks = {"frozen settings": t.get("frozen_sha256") == sha(ROOT / "data/reference/safe_frozen.json"),
              "plan": t.get("plan_sha256") == sha(ROOT / "data/reference/safe_validation_plan.json"),
              "analysis script": t.get("script_sha256") == sha(ROOT / "scripts/safe_analyze.py")}
    print("confirmatory report made under the archived files:", checks)
    if not all(checks.values()):
        sys.exit("a file of this archive differs from the one the confirmatory report was made with")
    print("\nTable 1 (development pilots, mean d):")
    for label, f, key in (("1", "safe_dev_pilot_v0.1.json", None), ("2", "safe_dev_pilot2_v0.1.json", None),
                          ("3", "safe_dev_pilot3_v0.1.json", None), ("3, K", "safe_dev_pilot3_calibration_v0.1.json", None)):
        r = load(f)
        print(f"  pilot {label}: sample {r['firms']}, mean d", {n: m["primary"]["mean_d"] for n, m in r["models"].items()},
              "complete cases", {n: m["primary"]["firms"] for n, m in r["models"].items()})
    c = load("safe_elicitation_compare_v0.1.json")["selection"]
    print("  elicitation comparison S:", c["S_errors_as_marginal"], "eligible", c["eligible"], "chosen", c["chosen"])
    print("\nTable 2 (confirmatory, 2026Q3):")
    for n, m in t["models"].items():
        pr = m["primary"]
        print(f"  {n}: K {t['draws_by_model'][n]}, firms {pr['firms']} (coverage {m['coverage']['share']}), mean d {pr['mean_d']}, "
              f"CI {pr['ci95']}, directional p {t['primary_family']['p_directional_H1'][n]}, Holm {t['primary_family']['holm_rejected'][n]}")
        s = pr["sensitivity"]
        print(f"     sign-flip p {pr['p_sign_flip']:.4f}, t p {pr['p_t']:.4f}, errors {m['technical_errors']}, "
              f"worst {s['errors_worst_symmetric']['mean_d']}, against the claim {s['errors_against_the_claim']['mean_d']}, "
              f"weighted {pr['mean_d_weighted_wgtcommon']}")
    print("\nTable 3 (references, band log score minus reference):")
    for n, m in t["models"].items():
        rf = m["references"]
        print(f"  {n}: full-marginal {rf['full']['minus_marginal']} {rf['full']['minus_marginal_ci95']} (n {rf['full']['firms']}), "
              f"short-marginal {rf['segment']['minus_marginal']} {rf['segment']['minus_marginal_ci95']} (n {rf['segment']['firms']}), "
              f"full-shuffled {m['full_minus_shuffled']['mean']} {m['full_minus_shuffled']['ci95']} (n {m['full_minus_shuffled']['firms']}), "
              f"full-logit {rf['full']['minus_logit']}")
    print(f"  statistical band log-loss (all {t['firms']} firms):", t["statistical"]["band_logloss"])
    print("  H3 (exploratory):", {n: (m["h3_full_vs_logit_segment"]["mean"], m["h3_full_vs_logit_segment"]["ci95"],
                                      m["h3_full_vs_logit_segment"]["firms"]) for n, m in t["models"].items()})
    print("\nTable 4 (explanatory measures, descriptive; models on the complete cases of the primary comparison):")
    for n, m in t["models"].items():
        print(f"  {n}: ten-value log-loss {m['ten_logloss']}, RPS {m['rps']}, firms {m['primary']['firms']}")
    print("  statistical ten-value log-loss:", t["statistical"]["ten_logloss"])
    print("  statistical RPS:", t["statistical"]["rps"])
    q = load("safe_h1_family_qualification_v0.1.json")
    print("\nCalibration: H1 bounds within levels", q["h1_bounds_within_levels"], "of", q["h1_bounds"],
          "; H3 over:", q["h3"]["bounds_over_their_level"])


if __name__ == "__main__":
    main()
