#!/usr/bin/env python3
"""SAFE validation: the qualification of the primary family (the three H1 under Holm) from an existing calibration
report (the project lead's decision of 19.11.2026). No model call, no change of the calibration report.

The calibration report (safe_analyze --level-check --k-from, development data) is read as it is: its
ready_for_freezing stays as written. From its stored counts every simultaneous one-sided Clopper-Pearson bound is
recomputed with the correction for ALL the report's bounds (36: not relaxed to the 18 H1 bounds) and checked against the
report's own bounds; the H1 family qualifies when the calibration is complete and formally valid (status
"calibration", no reason against it, K taken from a first report with --k-from, 4,000 x 10,000, all three models, both
sample sizes, the three levels) and every one of the 18 H1 bounds is within its level. H3 is not part of the family.

    python3 scripts/safe_h1_qualification.py --calibration data/processed/safe_dev_pilot3_calibration_v0.1.json \\
        --out data/processed/safe_h1_family_qualification_v0.1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).resolve().parent))
import safe_analyze as A  # noqa: E402

VERSION = "0.2"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    raw = Path(a.calibration).read_bytes()
    r = json.loads(raw)
    lc = r.get("level_check")
    if not lc or r.get("build", {}).get("split") != "dev":
        sys.exit("not a calibration report of development data")
    reasons = list(lc.get("not_ready_because") or [])
    if lc.get("status") != "calibration" or reasons:
        sys.exit(f"the calibration is not formally valid: {lc.get('status')} {reasons}")
    if not r.get("k_source"):
        sys.exit("K not taken from a first report (--k-from)")
    models = sorted(r["models"])
    if len(models) != A.FAMILY_SIZE:
        sys.exit(f"the family has {A.FAMILY_SIZE} models")
    # since 0.2 (the project lead's counterexamples, 19.11.2026): exactly the declared grid, 95%, the 36-bound
    # correction, the counts of the summary equal to the per-model details, levels compared in full precision
    if lc.get("simultaneous_confidence") != 0.95:
        sys.exit(f"simultaneous confidence {lc.get('simultaneous_confidence')}, the plan says 0.95")
    scen = ("expected_complete_cases", "all_eligible")
    grid = {(n, h, sc, lv) for n in models for h in ("H1", "H3") for sc in scen for lv in A.LOCAL_LEVELS}
    m = len(grid)  # 3 models x 2 hypotheses x 2 sample sizes x 3 levels = 36
    checks = lc["checks"]
    if len(checks) != m or lc.get("number_of_bounds") != m or lc.get("expected_bounds") != m:
        sys.exit(f"the report has {len(checks)} bounds, the declared grid {m}")
    q = 1 - (1 - 0.95) / m
    if abs(lc.get("bound_quantile_each", 0) - q) > 1e-6:
        sys.exit(f"the report's correction is not over {m} bounds")
    seen, keyed = set(), []
    for c in checks:  # first pass: the grid alone (every combination exactly once)
        n_, h_, sc_ = c["check"].split("/")
        lv = next((x for x in A.LOCAL_LEVELS if abs(x - c["level"]) < 1e-5), None)
        key_ = (n_, h_, sc_, lv)
        if lv is None or key_ not in grid or key_ in seen:
            sys.exit(f"a bound outside the declared grid or twice: {c['check']} at {c['level']}")
        seen.add(key_)
        keyed.append((c, n_, h_, sc_, lv))
    if seen != grid:
        sys.exit("the declared grid is not complete")
    for c, n_, h_, sc_, lv in keyed:  # second pass: the counts and the bounds
        det = r["models"][n_]["level_check"][f"{h_}/{sc_}"]  # the per-model details of the same replications
        k, nrep = c["claim_tail_rejections"], c["reps"]
        if det["claim_tail"][f"{lv:.4f}"] != k or det["reps"] != nrep:
            sys.exit(f"{c['check']} at {lv}: the summary's counts differ from the details")
        up = 1.0 if k >= nrep else float(beta.ppf(q, k + 1, nrep - k))
        if abs(up - c["simultaneous_upper"]) > 6e-6 or (up <= lv) != c["within_level"]:
            sys.exit(f"a bound does not recompute: {c['check']} at {lv}")
        c["within_level_full_precision"] = up <= lv
    h1 = [c for c in checks if c["check"].split("/")[1] == "H1"]
    for n in models:
        for sc_ in scen:
            x = r["models"][n]["level_check"][f"H1/{sc_}"]
            if x["reps"] < A.LEVEL_MIN_REPS or x["inner"] < A.LEVEL_MIN_INNER or x["replications_without_p"] > 0.01 * x["reps"]:
                sys.exit(f"{n}/H1/{sc_}: fewer replications than the plan's minimum or too many without a p value")
    qualified = all(c["within_level_full_precision"] for c in h1)
    out = {"script": f"safe_h1_qualification {VERSION}", "script_sha256": A.sha(Path(__file__).read_bytes()),
           "calibration_report_sha256": A.sha(raw), "calibration_build_items_sha256": r["build"]["items_sha256"],
           "K_by_model": lc.get("K_by_model"), "k_source": r["k_source"],
           "calibration_ready_for_freezing_as_written": lc.get("ready_for_freezing"),
           "h1_bounds": len(h1), "h1_bounds_within_levels": sum(c["within_level_full_precision"] for c in h1),
           "correction": f"simultaneous one-sided {lc['simultaneous_confidence']} Clopper-Pearson, Bonferroni over all {m} bounds "
                         "of the report (not relaxed)",
           "h1_family_qualified": qualified,
           "h3": {"bounds_over_their_level": [f"{c['check']} at {c['level']}" for c in checks
                                              if c["check"].split("/")[1] == "H3" and not c["within_level"]],
                  "role": "exploratory since 19.11.2026: estimates and intervals, no confirmatory claim, no level passed to H1"},
           "decision": "the project lead (19.11.2026): primary family = the three H1 under Holm at alpha 0.05 with the directional "
                       "rule; the choice between the graph and Holm was open in the plan and is made before the confirmatory "
                       "analysis, after the development results, because the calibration of H3 was not confirmed for two models "
                       "and in the graph a rejected H3 passes its level to the other H1; no model's results were selected",
           "checks_h1": h1}
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1, allow_nan=False) + "\n")
    print(json.dumps({k: out[k] for k in ("h1_bounds", "h1_bounds_within_levels", "h1_family_qualified",
                                           "calibration_ready_for_freezing_as_written")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
