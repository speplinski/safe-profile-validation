#!/usr/bin/env python3
"""SAFE validation: how much the card fields predict the chosen questions on the development sample (no model call).

Replaces the screening estimate of safe_validation_audit.py 0.1 (penalty chosen on the reported folds, standard
errors by observation) for the two questions chosen after that screening: q0b_3 (main; importance of access to
finance, 1-10) and qa21 (additional; late payments from private entities, 3 answers). Development sample as in the
audit: waves 2024Q3 and 2025Q3 without any observation of a firm that is in the test wave 2026Q3.

- Nested cross-validation grouped by firm: 5 outer folds; inside each training part, 4 inner folds choose the L2
  penalty from LAMBDAS by the MAIN measure (since 0.2: q0b_3 the band log-loss of the ten-value predictions; qa21
  its log-loss); the outer fold is scored with the model refit on the whole training part.
- Convergence (since 0.2): every fit must report success and finite coefficients and loss; a penalty with a failed
  inner fit is not eligible (recorded), a failed outer refit stops the script; every fit is in the diagnostics
  (status, message, iterations, largest absolute gradient).
- Methods: the marginal distribution; the size x sector cell rule (smoothing toward the marginal with weight K);
  a multinomial logit on the card fields (main effects); the same with interactions of size and of sector with every
  other card field.
- Card fields: exactly the columns of the card key (data/reference/safe_card_key.json): structural (d1_rec size,
  d3_rec sector, d4 turnover, d5_rec age, d6 ownership, d7_rec export), the 11 sources of financing (q4_*_rec),
  applications (q7a_*_rec) and their outcomes (q7b_*_rec): 25 columns; no perceptions of the same interview. A field
  not asked is its own category.
- q0b_3: the models are fitted on the ten values; the main score is the log-loss after summing the ten
  probabilities into the frozen bands 1-3 / 4-7 / 8-10; also the ten-value log-loss and the ranked probability
  score (RPS, ordinal). "Don't know"/NA (99) left out and counted. qa21: three answers, "don't know" left out.
- Uncertainty: the gain of a method over another is the mean per-observation difference of log scores, with a
  standard error clustered by firm (firms seen in both development waves). It is an approximate error of the
  obtained difference of validation scores: it does not cover the variability of refitting, of the penalty choice
  and of the folds (models scored in different folds share training data), and it is not the uncertainty of a
  later comparison of two LLM variants on the test wave.
Aggregates only.

    python3 scripts/safe_signal_check.py --safe data/local/safe/safe_poland_all_waves.csv.gz \
        --out data/processed/safe_signal_check_v0.3.json
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

VERSION = "0.3"
WAVES = {"32": "2024q3", "36": "2025q3", "40": "2026q3"}
TEST = "40"
KEY_PATH = Path(__file__).resolve().parents[1] / "data/reference/safe_card_key.json"


def card_fields() -> list:
    """The SAFE columns on the card, from the card key (since 0.3; 0.2 also had q4_g_rec, which is not on the card:
    a derived column equal to leasing or factoring)."""
    k = json.loads(KEY_PATH.read_text(encoding="utf-8"))
    out = [f["safe"] for f in k["fields"]] + [i["safe"] for i in k["sources_of_financing"]["items"]]
    out += [i["safe"] for i in k["applications"]["items"]] + [i["outcome"] for i in k["applications"]["items"]]
    if len(out) != len(set(out)):
        raise SystemExit("the card key names a column twice")
    return out


CARD = card_fields()
LAMBDAS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
BANDS = {"1-3": range(1, 4), "4-7": range(4, 8), "8-10": range(8, 11)}


def norm(v: str) -> str:
    v = (v or "").strip()
    if v in ("", ".", "nan"):
        return "NA"
    return v[:-2] if v.endswith(".0") else v


def design(rows: list, feats: list, inter: bool, vocab: dict | None = None):
    def keys(r):
        ks = [(f, norm(r[f])) for f in feats]
        if inter:
            for a in ("d1_rec", "d3_rec"):
                ks += [(f"{a}*{f}", norm(r[a]) + "|" + norm(r[f])) for f in feats if f != a]
        return ks
    if vocab is None:
        vocab = {}
        for r in rows:
            for k in keys(r):
                vocab.setdefault(k, len(vocab))
    x = np.zeros((len(rows), len(vocab)))
    for i, r in enumerate(rows):
        for k in keys(r):
            j = vocab.get(k)
            if j is not None:
                x[i, j] = 1
    return x, vocab


DIAG: list = []


class FitFailure(Exception):
    pass


def fit(x, y, k: int, lam: float, tag: str = ""):
    m = x.shape[1]
    t = np.eye(k)[y]

    def f(w):
        b, c = w[:m * k].reshape(m, k), w[m * k:]
        z = x @ b + c
        lz = z - logsumexp(z, 1, keepdims=True)
        p = np.exp(lz)
        g = np.concatenate([(x.T @ (p - t) / len(y) + lam * b).ravel(), (p - t).mean(0)])
        return -np.sum(lz * t) / len(y) + lam / 2 * np.sum(b * b), g

    res = minimize(f, np.zeros(m * k + k), jac=True, method="L-BFGS-B", options={"maxiter": 3000})
    w = res.x
    finite = bool(np.all(np.isfinite(w)) and np.isfinite(res.fun))
    d = {"tag": tag, "lambda": lam, "success": bool(res.success), "status": int(res.status),
         "message": str(res.message), "nit": int(res.nit), "finite": finite,
         "max_abs_grad": float(np.max(np.abs(res.jac))) if res.jac is not None else None}
    DIAG.append(d)
    if not (res.success and finite):
        raise FitFailure(f"{tag} lambda={lam}: {d['message']} (finite={finite})")
    return w[:m * k].reshape(m, k), w[m * k:]


def predict(rows_tr, y_tr, rows_te, k, method, lam=None, tag=""):
    if method in ("marginal", "rule"):
        prior = (np.bincount(y_tr, minlength=k) + 0.5) / (len(y_tr) + 0.5 * k)
        if method == "marginal":
            return np.tile(prior, (len(rows_te), 1))
        cnt = defaultdict(lambda: np.zeros(k))
        for r, v in zip(rows_tr, y_tr):
            cnt[(norm(r["d1_rec"]), norm(r["d3_rec"]))][v] += 1
        out = np.zeros((len(rows_te), k))
        for i, r in enumerate(rows_te):
            c = cnt[(norm(r["d1_rec"]), norm(r["d3_rec"]))]
            out[i] = (c + k * prior) / (c.sum() + k)
        return out
    inter = method == "logit_interactions"
    xtr, voc = design(rows_tr, CARD, inter)
    xte, _ = design(rows_te, CARD, inter, voc)
    b, c = fit(xtr, y_tr, k, lam, tag)
    z = xte @ b + c
    return np.exp(z - logsumexp(z, 1, keepdims=True))


def folds(rows: list, n: int, seed: int) -> np.ndarray:
    firms = sorted({r["permid"] for r in rows})
    random.Random(seed).shuffle(firms)
    f = {p: i % n for i, p in enumerate(firms)}
    return np.array([f[r["permid"]] for r in rows])


def nested(rows: list, y: np.ndarray, k: int, method: str, score_fn, tag: str = "",
           seed: int = 1) -> tuple[np.ndarray, list, list]:
    """score_fn(p, y) -> per-observation log scores of the MAIN measure, used to choose the penalty."""
    outer = folds(rows, 5, seed)
    excluded = []
    p = np.zeros((len(rows), k))
    chosen = []
    for o in range(5):
        tr, te = np.where(outer != o)[0], np.where(outer == o)[0]
        rtr = [rows[i] for i in tr]
        lam = None
        if method.startswith("logit"):
            inner = folds(rtr, 4, seed + 1 + o)
            score = {}
            for lm in LAMBDAS:
                s = 0.0
                try:
                    for q in range(4):
                        a, b = np.where(inner != q)[0], np.where(inner == q)[0]
                        pp = predict([rtr[i] for i in a], y[tr][a], [rtr[i] for i in b], k, method, lm,
                                     f"{tag}/{method}/outer{o}/inner{q}")
                        s += score_fn(pp, y[tr][b]).sum()
                except FitFailure as e:
                    excluded.append({"outer": o, "lambda": lm, "reason": str(e)})
                    continue
                score[lm] = s
            if not score:
                raise SystemExit(f"{tag}/{method}/outer{o}: no penalty with converged inner fits")
            lam = max(score, key=score.get)
            chosen.append(lam)
        try:
            p[te] = predict(rtr, y[tr], [rows[i] for i in te], k, method, lam, f"{tag}/{method}/outer{o}/refit")
        except FitFailure as e:
            raise SystemExit(f"outer refit did not converge: {e}")
    return p, chosen, excluded


def plain_scores(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))


def band_scores(cats: list):
    """Per-observation log score of the bands, from ten-value predictions (columns in the order of cats)."""
    cols = [[cats.index(str(v)) for v in rng] for rng in BANDS.values()]
    band_of = np.array([next(i for i, rng in enumerate(BANDS.values()) if int(c) in rng) for c in cats])

    def f(p: np.ndarray, y: np.ndarray) -> np.ndarray:
        pb = np.stack([p[:, c].sum(1) for c in cols], 1)
        yb = band_of[y]
        return np.log(np.clip(pb[np.arange(len(yb)), yb], 1e-12, 1))
    return f


def cluster_se(g: np.ndarray, firms: list) -> float:
    n, mean = len(g), g.mean()
    by = defaultdict(float)
    for v, f in zip(g, firms):
        by[f] += v - mean
    m = len(by)
    return math.sqrt(m / (m - 1) * sum(s * s for s in by.values())) / n


def rps(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    k = p.shape[1]
    cp = np.cumsum(p, 1)[:, :-1]
    co = (np.arange(k - 1)[None, :] >= y[:, None]).astype(float)
    return ((cp - co) ** 2).sum(1) / (k - 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--safe", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    raw = Path(a.safe).read_bytes()
    rows = [r for r in csv.DictReader(gzip.decompress(raw).decode("utf-8").splitlines()) if r["wave"] in WAVES]
    test_firms = {r["permid"] for r in rows if r["wave"] == TEST}
    dev = [r for r in rows if r["wave"] != TEST and r["permid"] not in test_firms]
    test = [r for r in rows if r["wave"] == TEST]
    methods = ["marginal", "rule", "logit_main", "logit_interactions"]
    out = {"script": f"safe_signal_check {VERSION}", "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "source_sha256": hashlib.sha256(raw).hexdigest(), "card_fields": CARD, "lambdas": list(LAMBDAS),
           "development_rule": "waves 2024q3 and 2025q3 without any observation of a firm in the test wave 2026q3",
           "questions": {}}
    specs = {
        "q0b_3": (lambda r: norm(r["q0b_3"]), [str(i) for i in range(1, 11)], "main"),
        "qa21": (lambda r: norm(r["qa21_" + WAVES[r["wave"]]]), ["1", "2", "3"], "additional"),
    }
    for q, (get, cats, role) in specs.items():
        keep = [r for r in dev if get(r) in cats]
        k = len(cats)
        y = np.array([cats.index(get(r)) for r in keep])
        firms = [r["permid"] for r in keep]
        res = {"role": role, "categories": cats, "n_dev": len(keep), "firms_dev": len(set(firms)),
               "excluded_dev": dict(Counter(get(r) for r in dev if get(r) not in cats)),
               "excluded_test": dict(Counter(get(r) for r in test if get(r) not in cats)),
               "n_test_answering": sum(get(r) in cats for r in test), "methods": {}, "gains": {}}
        logs = {}
        if q == "q0b_3":
            score_fn = band_scores(cats)
            res["penalty_choice"] = "band log-loss of the ten-value predictions (the main measure)"
        else:
            score_fn = plain_scores
            res["penalty_choice"] = "log-loss of the three answers"
        for m in methods:
            n0 = len(DIAG)
            p, lam, excl = nested(keep, y, k, m, score_fn, q)
            fits = DIAG[n0:]
            diag = {"fits": len(fits), "failed": sum(not (d["success"] and d["finite"]) for d in fits),
                    "messages": dict(Counter(d["message"] for d in fits)),
                    "max_iterations": max((d["nit"] for d in fits), default=None),
                    "largest_abs_gradient": max((d["max_abs_grad"] for d in fits if d["max_abs_grad"] is not None), default=None),
                    "penalties_excluded": excl}
            if q == "q0b_3":
                lb = score_fn(p, y)
                l10 = np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))
                res["methods"][m] = {"logloss_bands": round(-lb.mean(), 4), "logloss_10": round(-l10.mean(), 4),
                                     "rps_10": round(float(rps(p, y).mean()), 4), "chosen_lambdas": lam,
                                     "fit_diagnostics": diag}
                logs[m] = lb
            else:
                l3 = np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))
                res["methods"][m] = {"logloss": round(-l3.mean(), 4), "chosen_lambdas": lam, "fit_diagnostics": diag}
                logs[m] = l3
        for a_, b_ in (("marginal", "rule"), ("marginal", "logit_main"), ("marginal", "logit_interactions"),
                       ("rule", "logit_main"), ("rule", "logit_interactions")):
            g = logs[b_] - logs[a_]
            res["gains"][f"{b_} over {a_}"] = {"mean": round(float(g.mean()), 4), "se_clustered_by_firm": round(cluster_se(g, firms), 4)}
        out["questions"][q] = res
    out["fits_total"] = len(DIAG)
    out["fits_failed"] = sum(not (d["success"] and d["finite"]) for d in DIAG)
    out["note"] = ("the main score of q0b_3 is the band log-loss of ten-value predictions; the gains are the "
                   "uncertainty of these comparisons on the development sample only")
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    for q, r in out["questions"].items():
        print(q, r["n_dev"], r["firms_dev"], json.dumps(r["methods"]), json.dumps(r["gains"]))


if __name__ == "__main__":
    main()
