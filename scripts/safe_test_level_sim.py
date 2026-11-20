#!/usr/bin/env python3
"""SAFE validation: the level of candidate tests of mean(d) = 0 under skewed differences (no data, no model call).

Four shapes of d with mean 0 (normal; strongly skewed lognormal; 3% of extreme losses at -5.8, the log score of a
worst-scored error, the rest shifted to mean 0; 45% zeros and a skewed rest), n = 120 (pilot) and 692 (test wave),
REPS replications; tests: the sign-flip permutation test, the t test, the studentised bootstrap (bootstrap-t), the
percentile bootstrap interval; rejection rates at 0.05 and 0.05/3 (the smallest Holm threshold of three). Monte Carlo
error about 0.006 at 0.05 and 0.003 at 0.0167 with 1500 replications.

    python3 scripts/safe_test_level_sim.py --out data/processed/safe_test_level_sim_v0.2.json
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import stats

REPS, RESAMPLES, SEED = 1500, 1000, 20261118  # v0.2 adds the split by tail (same seed, same draws as v0.1)


def shapes(rng):
    def heavy(n):
        f = rng.random(n) < 0.03
        x = rng.normal(0, 0.5, n)
        x[f] = -5.8
        return x + 0.03 * 5.8

    def zeros(n):
        z = rng.random(n) < 0.45
        x = (rng.gamma(2, 0.25, n) - 0.5) * 1.5
        x[z] = 0
        return x
    return {"normal": lambda n: rng.normal(0, 0.6, n),
            "skewed lognormal": lambda n: (rng.lognormal(0, 1, n) - np.exp(0.5)) * 0.3,
            "3% at -5.8": heavy, "45% zeros + skewed": zeros}


def tests(rng, d):
    n, m = len(d), d.mean()
    se = d.std(ddof=1) / np.sqrt(n)
    s = rng.choice([-1.0, 1.0], size=(RESAMPLES, n))
    p_flip = (np.abs((s * d).mean(1)) >= abs(m) - 1e-12).mean()
    p_t = 2 * stats.t.sf(abs(m / se), n - 1)
    b = rng.integers(0, n, (RESAMPLES, n))
    bm, bs = d[b].mean(1), d[b].std(1, ddof=1) / np.sqrt(n)
    p_bt = (np.abs((bm - m) / bs) >= abs(m / se)).mean()
    lo, hi = np.quantile(bm, [0.025, 0.975])
    lo3, hi3 = np.quantile(bm, [0.025 / 3, 1 - 0.025 / 3])
    return p_flip, p_t, p_bt, not (lo <= 0 <= hi), not (lo3 <= 0 <= hi3), float(m > 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    out = {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "reps": REPS, "resamples": RESAMPLES,
           "seed": SEED, "results": {}}
    for name, g in shapes(rng).items():
        for n in (120, 692):
            R = np.array([tests(rng, g(n)) for _ in range(REPS)], dtype=float)
            out["results"][f"{name} | n={n}"] = {
                "sign_flip": [round((R[:, 0] <= 0.05).mean(), 4), round((R[:, 0] <= 0.05 / 3).mean(), 4)],
                "t": [round((R[:, 1] <= 0.05).mean(), 4), round((R[:, 1] <= 0.05 / 3).mean(), 4)],
                "bootstrap_t": [round((R[:, 2] <= 0.05).mean(), 4), round((R[:, 2] <= 0.05 / 3).mean(), 4)],
                "percentile_interval": [round(R[:, 3].mean(), 4), round(R[:, 4].mean(), 4)],
                "claim_tail": {t: [round(((R[:, c] <= 0.05) & (R[:, 5] == 1)).mean(), 4),
                                   round(((R[:, c] <= 0.05 / 3) & (R[:, 5] == 1)).mean(), 4)]
                               for t, c in (("sign_flip", 0), ("t", 1), ("bootstrap_t", 2))},
                "other_tail": {t: [round(((R[:, c] <= 0.05) & (R[:, 5] == 0)).mean(), 4),
                                   round(((R[:, c] <= 0.05 / 3) & (R[:, 5] == 0)).mean(), 4)]
                               for t, c in (("sign_flip", 0), ("t", 1), ("bootstrap_t", 2))}}
            print(name, n, out["results"][f"{name} | n={n}"])
    out["note"] = ("pairs are the rejection rates at alpha 0.05 and 0.05/3 under a true null; claim_tail / other_tail split "
                   "them by the sign of the mean (the claim needs d > 0, so its error is the claim tail; nominal half of "
                   "the two-sided rate: 0.025 and 0.0083)")
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
