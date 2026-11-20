#!/usr/bin/env python3
"""SAFE validation, step 0: an audit of the held-out question and of the split, before any model call.

Reads the local extract of the Polish SAFE public use file records (data/local/safe/, not in git) and writes only
aggregates (counts and association measures, no record-level data) to a JSON file:
- the waves of the question (column qa21_<period>; the wave code is checked against the column's coverage);
- per wave: observations, distinct firms (permid), size classes, the reference-period group, the answer codes of
  the question overall and by size class;
- the panel: firms of the test wave seen in earlier waves (by permid), and the development sample after removing
  every earlier observation of a test firm;
- candidate card fields: coverage in every wave and, on the development sample, Cramer's V with the question's
  answer (codes 1-3; "don't know" and missing left out), with a flag at V >= 0.3 for review; fields that are
  conditional on the question itself (its follow-ups) are excluded by construction and listed with the reason.
- the available signal of candidate held-out questions: on the development sample, 5-fold cross-validation grouped
  by firm, log-loss of the marginal distribution, of a size x sector cell rule (add-one smoothing toward the
  marginal) and of an L2 multinomial logistic regression on the structural, product-use and application fields
  (the target left out); scales of 1-10 are banded 1-3 / 4-7 / 8-10 for this check only; "don't know" left out.
  This is a SCREENING estimate: the penalty is chosen from a small grid on the same folds that are reported (so
  the best value may be optimistic, it is not a bound on predictability) and gain_se is by observation although
  some firms are in both development waves. The chosen questions are checked with nested cross-validation and
  firm-clustered errors in scripts/safe_signal_check.py.
No weights: the firm-level prediction is unweighted; population-level weights are a separate step.

    python3 scripts/safe_validation_audit.py --safe data/local/safe/safe_poland_all_waves.csv.gz \
        --out data/processed/safe_validation_audit_v0.1.json
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

VERSION = "0.1"
QUESTION = "qa21"
WAVES = {"32": "2024q3", "36": "2025q3", "40": "2026q3"}
TEST = "40"
ANSWERS = {"1": "yes_regularly", "2": "yes_occasionally", "3": "no", "99": "dont_know"}
SIZE = {"1": "1-9", "2": "10-49", "3": "50-249", "4": "250+"}
# candidate card fields by group; a group is a design choice for the plan, not a result of this audit
GROUPS = {
    "structural": ["d1_rec", "d3_rec", "d4", "d5_rec", "d6", "d7_rec"],
    "products_used": ["q4_a_rec", "q4_b_rec", "q4_c_rec", "q4_d_rec", "q4_e_rec", "q4_f_rec", "q4_g_rec",
                      "q4_h_rec", "q4_j_rec", "q4_m_rec", "q4_p_rec", "q4_r_rec"],
    "applications_and_unmet_need": ["q7a_a_rec", "q7a_b_rec", "q7a_c_rec", "q7a_d_rec", "q7b_a_rec", "q7b_b_rec",
                                    "q7b_c_rec", "q7b_d_rec"],
    "perceptions_same_interview": ["q0b_1", "q0b_2", "q0b_3", "q0b_4", "q0b_5", "q0b_6", "q2_a", "q2_e", "q2_g",
                                   "q2_i", "q11_a", "q11_c", "q11_d", "q16_a", "q16_b", "q17", "q19_a"],
}
SIGNAL_TARGETS = {"qa21": "question", "q0b_1": "1-10", "q0b_2": "1-10", "q0b_3": "1-10", "q0b_4": "1-10",
                  "q0b_5": "1-10", "q0b_6": "1-10", "q2_a": "1-3", "q2_e": "1-3", "q2_g": "1-3", "q11_c": "1-3",
                  "q11_d": "1-3", "q5_a_rec": "1-3", "q5_f_rec": "1-3", "q5_g_rec": "1-3"}
SIGNAL_FEATURES = GROUPS["structural"] + GROUPS["products_used"] + GROUPS["applications_and_unmet_need"][:4]
LAMBDAS = (0.03, 0.1, 0.3)
FOLLOW_UPS = r"^qa(3|4)_"  # consequences and severity of late payments: asked only after a "yes" to qa21/qa22
MISSING = {"", ".", " ", "nan"}


def norm(v: str) -> str:
    v = (v or "").strip()
    if v in MISSING:
        return ""
    return v[:-2] if v.endswith(".0") else v


def cramers_v(pairs: list) -> float | None:
    """Cramer's V of a contingency table from (x, y) pairs; None below 2 levels on either side."""
    xs, ys = sorted({x for x, _ in pairs}), sorted({y for _, y in pairs})
    n = len(pairs)
    if len(xs) < 2 or len(ys) < 2 or not n:
        return None
    cnt = Counter(pairs)
    rx, ry = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    chi = sum((cnt[(x, y)] - rx[x] * ry[y] / n) ** 2 / (rx[x] * ry[y] / n) for x in xs for y in ys)
    return math.sqrt(chi / n / (min(len(xs), len(ys)) - 1))


def _design(rs: list, feats: list, vocab: dict | None = None):
    if vocab is None:
        vocab = {}
        for r in rs:
            for f in feats:
                vocab.setdefault((f, norm(r[f]) or "NA"), len(vocab))
    x = np.zeros((len(rs), len(vocab)))
    for i, r in enumerate(rs):
        for f in feats:
            j = vocab.get((f, norm(r[f]) or "NA"))
            if j is not None:
                x[i, j] = 1
    return x, vocab


def _fit(x, y, k: int, lam: float):
    m = x.shape[1]
    t = np.eye(k)[y]

    def f(w):
        b, c = w[:m * k].reshape(m, k), w[m * k:]
        z = x @ b + c
        lz = z - logsumexp(z, 1, keepdims=True)
        p = np.exp(lz)
        g = np.concatenate([(x.T @ (p - t) / len(y) + lam * b).ravel(), (p - t).mean(0)])
        return -np.sum(lz * t) / len(y) + lam / 2 * np.sum(b * b), g

    w = minimize(f, np.zeros(m * k + k), jac=True, method="L-BFGS-B", options={"maxiter": 2000}).x
    return w[:m * k].reshape(m, k), w[m * k:]


def signal(dev: list, label, feats: list, seed: int = 1) -> dict:
    """Cross-validated log-loss (grouped by firm) of the marginal, the size x sector rule and the best L2 logit."""
    rows = [(r, label(r)) for r in dev]
    rows = [(r, v) for r, v in rows if v is not None]
    cats = sorted({v for _, v in rows})
    k = len(cats)
    y = np.array([cats.index(v) for _, v in rows])
    rs = [r for r, _ in rows]
    firms = sorted({r["permid"] for r in rs})
    import random as _r
    _r.Random(seed).shuffle(firms)
    fold = {p: i % 5 for i, p in enumerate(firms)}
    fo = np.array([fold[r["permid"]] for r in rs])

    def cv(keys=None, lam=None):
        p = np.zeros((len(rs), k))
        for q in range(5):
            tr, te = np.where(fo != q)[0], np.where(fo == q)[0]
            if lam is None:
                prior = np.bincount(y[tr], minlength=k) / len(tr)
                cnt = defaultdict(lambda: np.zeros(k))
                for i in tr:
                    cnt[tuple(norm(rs[i][c]) for c in keys)][y[i]] += 1
                for i in te:
                    cc = cnt[tuple(norm(rs[i][c]) for c in keys)]
                    p[i] = (cc + k * prior) / (cc.sum() + k)
            else:
                xtr, v = _design([rs[i] for i in tr], feats)
                xte, _ = _design([rs[i] for i in te], feats, v)
                b, c = _fit(xtr, y[tr], k, lam)
                z = xte @ b + c
                p[te] = np.exp(z - logsumexp(z, 1, keepdims=True))
        return np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))

    marg, rule = cv([]), cv(["d1_rec", "d3_rec"])
    fits = {lam: cv(lam=lam) for lam in LAMBDAS}
    lam = max(fits, key=lambda l: fits[l].mean())
    gain = fits[lam] - marg
    return {"n": len(y), "categories": cats, "logloss_marginal": round(-marg.mean(), 4),
            "logloss_size_sector_rule": round(-rule.mean(), 4), "logloss_best_logit": round(-fits[lam].mean(), 4),
            "best_lambda": lam, "gain_best_over_marginal": round(gain.mean(), 4),
            "gain_se": round(gain.std() / math.sqrt(len(gain)), 4)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--safe", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    raw = Path(a.safe).read_bytes()
    rows = list(csv.DictReader(gzip.decompress(raw).decode("utf-8").splitlines()))
    cols = set(rows[0])
    by_wave = defaultdict(list)
    for r in rows:
        if r["wave"] in WAVES:
            by_wave[r["wave"]].append(r)
    # the column of every wave is answered only in that wave
    for w, period in WAVES.items():
        c = f"{QUESTION}_{period}"
        for w2, rs in by_wave.items():
            n = sum(norm(r[c]) != "" for r in rs)
            if (w2 == w) != (n > 0):
                raise SystemExit(f"{c}: answered in wave {w2} ({n} rows), expected only in wave {w}")

    def ans(r: dict) -> str:
        return norm(r[f"{QUESTION}_{WAVES[r['wave']]}"])

    out = {"script": f"safe_validation_audit {VERSION}", "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "source_sha256": hashlib.sha256(raw).hexdigest(), "source_rows": len(rows), "question": QUESTION,
           "answer_codes": ANSWERS, "waves": {}, "panel": {}, "development": {}, "fields": {}}
    for w, rs in sorted(by_wave.items()):
        out["waves"][WAVES[w]] = {
            "wave_code": w, "observations": len(rs), "firms": len({r["permid"] for r in rs}),
            "size": {SIZE.get(norm(r["d1_rec"]), norm(r["d1_rec"])): 0 for r in rs} | dict(Counter(SIZE.get(norm(r["d1_rec"]), norm(r["d1_rec"])) for r in rs)),
            "reference_period_group_experiment_1": dict(Counter(norm(r["experiment_1"]) for r in rs)),
            "answers": dict(Counter(ANSWERS.get(ans(r), ans(r) or "missing") for r in rs)),
            "answers_by_size": {s: dict(Counter(ANSWERS.get(ans(r), ans(r) or "missing") for r in rs
                                                if SIZE.get(norm(r["d1_rec"])) == s)) for s in SIZE.values()},
        }
    test_firms = {r["permid"] for r in by_wave[TEST]}
    earlier = [w for w in WAVES if w != TEST]
    all_earlier = [r for r in rows if r["wave"] not in (TEST,) and int(r["wave"]) < int(TEST)]
    out["panel"] = {
        "test_wave": WAVES[TEST],
        "test_firms_seen_in": {WAVES[w]: len(test_firms & {r["permid"] for r in by_wave[w]}) for w in earlier},
        "test_firms_seen_in_any_earlier_wave": len(test_firms & {r["permid"] for r in all_earlier}),
        "earlier_observations_of_test_firms_all_waves": sum(r["permid"] in test_firms for r in all_earlier),
        "firms_in_both_development_waves": len({r["permid"] for r in by_wave[earlier[0]]} & {r["permid"] for r in by_wave[earlier[1]]}),
    }
    dev = [r for w in earlier for r in by_wave[w] if r["permid"] not in test_firms]
    out["development"] = {
        "rule": "question waves before the test wave, without any observation of a firm that is in the test wave",
        "observations": len(dev), "firms": len({r["permid"] for r in dev}),
        "observations_removed": sum(len(by_wave[w]) for w in earlier) - len(dev),
        "answers": dict(Counter(ANSWERS.get(ans(r), ans(r) or "missing") for r in dev)),
    }
    for g, fs in GROUPS.items():
        for f in fs:
            if f not in cols:
                out["fields"][f] = {"group": g, "absent": True}
                continue
            cov = {WAVES[w]: round(sum(norm(r[f]) != "" for r in rs) / len(rs), 4) for w, rs in sorted(by_wave.items())}
            pairs = [(norm(r[f]), ans(r)) for r in dev if norm(r[f]) != "" and ans(r) in ("1", "2", "3")]
            v = cramers_v(pairs)
            out["fields"][f] = {"group": g, "coverage": cov, "dev_pairs": len(pairs),
                                "cramers_v_dev": None if v is None else round(v, 4),
                                "review": v is not None and v >= 0.3}
    def band(v: str) -> str:
        x = int(v)
        return "1-3" if x <= 3 else ("4-7" if x <= 7 else "8-10")

    out["signal"] = {"features": SIGNAL_FEATURES, "lambdas": list(LAMBDAS), "targets": {}}
    for t, kind in SIGNAL_TARGETS.items():
        if t == QUESTION:
            lab = lambda r: ans(r) if ans(r) in ("1", "2", "3") else None  # noqa: E731
        elif kind == "1-10":
            lab = lambda r, t=t: band(norm(r[t])) if norm(r[t]) in {str(i) for i in range(1, 11)} else None  # noqa: E731
        else:
            lab = lambda r, t=t: norm(r[t]) if norm(r[t]) in ("1", "2", "3") else None  # noqa: E731
        out["signal"]["targets"][t] = signal(dev, lab, [f for f in SIGNAL_FEATURES if f != t])
    out["excluded_by_construction"] = {
        "columns": sorted(c for c in cols if re.match(FOLLOW_UPS, c)),
        "reason": "follow-ups of qa21/qa22 (consequences, severity): asked only after a yes, so they reveal the answer",
    }
    out["note"] = ("aggregates only; the wave codes are checked against the coverage of the question's column; "
                   "Cramer's V is a screen for fields that may restate the answer, not a test")
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("panel", "development")}, ensure_ascii=False))
    print("review:", [f for f, d in out["fields"].items() if d.get("review")])


if __name__ == "__main__":
    main()
