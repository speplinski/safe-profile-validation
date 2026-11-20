#!/usr/bin/env python3
"""SAFE validation: score the LLM runs and the statistical methods on one build, by data/reference/safe_validation_plan.json.
No model call. Writes aggregates only (no record-level data).

Inputs: the build of safe_build.py (key.json, design.json, items.jsonl.gz checked against design.json), one run
directory of run_b_k2.py per model (--run NAME=DIR, names unique; each run must be of this build, with the
safe_probabilities parser and max_attempts 2, its last session finished (finished_at, no stopped_reason), and every
item either valid or with max_attempts model answers), and the SAFE extract (its sha256 must be the build's) for
the statistical methods, trained on the development sample (for a development build without the build's firms).

Conventions: a LOG SCORE is higher-is-better (the log of the floored probability of the observed band); a
LOG-LOSS is its negative (lower is better); d and every "minus" are differences of log scores, so a positive value
favours the first method.

Scoring (the plan): every method's ten probabilities are floored as (1 - FLOOR) p + FLOOR / 10; main measure the
band log score; the primary family has the FAMILY_SIZE models of the plan: for each, full vs segment on the firms with
valid answers in both (d = log score full - log score segment; since 0.4 a two-sided studentised bootstrap test
(bootstrap-t, FLIPS resamples, fixed seed), because the sign-flip test needs a symmetric d under the null and was too
liberal for skewed d in the simulation; the sign-flip and t p values are reported next to it; the skewness of d and,
with --level-check, the test's level on the model's own centred differences); bootstrap interval of the mean d; a
model absent from this report, or without a common firm, enters Holm with p = 1 (flagged); the claim is supported only
by a Holm rejection WITH mean d > 0. Sensitivities: technical errors scored with the marginal; scored worst in both
arms (symmetric); scored against the claim (a failed full card worst, a failed segment card with the marginal); the
floor 0.001 and 0.05. A model with more than ERROR_FLAG technical errors in a variant is flagged (main result and
sensitivities on equal footing), with the difference of error shares between full and segment and its bootstrap
interval. Full vs shuffled is diagnostic only (interval, no test; since 0.4, as a test after selecting the models
would need a hierarchical alpha transfer). References for every variant: minus marginal, minus rule, minus logit (card
fields), minus logit on size and sector only.
Explanatory (no p values): ten-value log-loss, RPS, calibration by band and by size class, by size class, technical
errors by outcome, the share of firms where the floor acts, stability of the repeats of both cards and, on firms with
repeats of both, the model-noise variance of d and its share of var(d) (for the local model at temperature 0 with a
fixed seed the repeats are near-deterministic, not exactly), SD of d by size and pooled to the test wave's size shares
with the minimum detectable effect, the ablation of the logit without the applications (q7a, q7b). The runs may use
the parser safe_probabilities or safe_counts, the build's.

Since 0.7 (project lead's review, 18.11.2026): a run with more than one served model is refused; --k NAME=K fixes the
draws averaged per card (the build must have them for every firm); on the test split --frozen is required (run
settings, weights digest or served model, and K per model) and --k / --average-draws are refused; --average-draws is a
pilot diagnostic; the main result is named the mean d among complete cases, with the coverage next to it; the
technical-error diagnostics stay without common firms; the references carry a bootstrap-t interval against the
marginal; the precision uses the total SD of d over the test wave's size mix and the run's coverage; the logit penalty
is chosen with the floored band score; the level check also covers H3, with a Wilson interval of the claim tail.

    python3 scripts/safe_analyze.py --build <build dir> --run qwen=<dir> --run haiku=<dir> --run luna=<dir> \
        --safe data/local/safe/safe_poland_all_waves.csv.gz --out <report.json>
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm as normal

sys.path.insert(0, str(Path(__file__).resolve().parent))
import safe_signal_check as SC  # noqa: E402

VERSION = "0.8"
FLOOR = 0.01
FLIPS = 10000
BOOT = 2000
SEED = 20261118
SCALE = [str(i) for i in range(1, 11)]
BAND_OF = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
BANDS = ("1-3", "4-7", "8-10")
ALPHA = 0.05
FAMILY_SIZE = 3
ERROR_FLAG = 0.05
FLOORS_SENSITIVITY = (0.001, 0.05)
MIN_FIRMS = 30  # fewer common firms: no test (p None, Holm/graph with p = 1)
SUM_RANGE = (50, 200)  # counts format: answers with a sum outside are a sensitivity (plan, F4)
SUM_FLAG = 0.05
K_MIN_FIRMS = 20  # fewer firms with repeats of both cards: the K rule gives the conservative K = 3
VARIANTS = ("full", "segment", "shuffled")
MAX_ATTEMPTS = 2
SEGMENT_FIELDS = ["d1_rec", "d3_rec"]
NO_APPLICATIONS = [f for f in SC.CARD if not f.startswith(("q7a_", "q7b_"))]
ROOT = Path(__file__).resolve().parents[1]


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def floor(p: np.ndarray, f: float = FLOOR) -> np.ndarray:
    return (1 - f) * p + f / 10


def band_probs(p: np.ndarray) -> np.ndarray:
    return np.stack([p[:, BAND_OF == b].sum(1) for b in range(3)], 1)


def scores(p: np.ndarray, y: np.ndarray, f: float = FLOOR) -> dict:
    """Per-firm log scores (higher is better) of floored ten-value predictions: bands, ten values; and RPS."""
    q = floor(p, f)
    pb = band_probs(q)
    return {"band": np.log(pb[np.arange(len(y)), BAND_OF[y]]), "ten": np.log(q[np.arange(len(y)), y]),
            "rps": SC.rps(q, y)}


def worst_band(y: np.ndarray) -> np.ndarray:
    """The log score of a prediction that gives the true band only the floor's probability."""
    return np.log(np.array([FLOOR / 10 * (BAND_OF == BAND_OF[v]).sum() for v in y]))


def flip_test(d: np.ndarray, flips: int = FLIPS, seed: int = SEED):
    """Two-sided sign-flip permutation p value of the mean of d ((k + 1) / (flips + 1)); None without data."""
    if len(d) == 0:
        return None
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    s = rng.choice([-1.0, 1.0], size=(flips, len(d)))
    k = int((np.abs((s * d).mean(1)) >= obs - 1e-12).sum())
    return (k + 1) / (flips + 1)


def boot_ci(d: np.ndarray, reps: int = BOOT, seed: int = SEED):
    if len(d) == 0:
        return None
    rng = np.random.default_rng(seed)
    m = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(reps)])
    return [round(float(np.quantile(m, 0.025)), 4), round(float(np.quantile(m, 0.975)), 4)]


def boot_t_test(d: np.ndarray, reps: int = FLIPS, seed: int = SEED):
    """Two-sided studentised bootstrap (bootstrap-t) p value of mean(d) = 0, (k + 1) / (reps + 1); None without data.
    Since 0.4 the main test: it needs no symmetry of d under the null (the sign-flip test does), and in the simulation
    of docs/walidacja-safe.md it kept the level best under skewed differences."""
    n = len(d)
    if n < 2:
        return None
    m, se = d.mean(), d.std(ddof=1) / math.sqrt(n)
    if se == 0:  # a constant d: the studentised statistic is undefined
        return None
    rng = np.random.default_rng(seed)
    t_obs = abs(m / se)
    k = 0
    for start in range(0, reps, 1000):
        b = rng.integers(0, n, (min(1000, reps - start), n))
        x = d[b]
        sb = x.std(1, ddof=1) / math.sqrt(n)
        with np.errstate(divide="ignore", invalid="ignore"):
            tb = np.where(sb > 0, np.abs((x.mean(1) - m) / sb), np.inf)  # a resample with se* = 0 counts as extreme
        k += int((tb >= t_obs - 1e-12).sum())
    return (k + 1) / (reps + 1)


def boot_t_ci(d: np.ndarray, reps: int = FLIPS, seed: int = SEED, level: float = 0.95):
    """Symmetric studentised bootstrap interval of the mean, m +- q(|t*|) se, the interval that matches the two-sided
    boot_t_test (since 0.6; a resample with se* = 0 counts as extreme, as in the test); None below MIN_FIRMS values, with
    se 0 or when the quantile is infinite."""
    n = len(d)
    if n < MIN_FIRMS:
        return None
    m, se = d.mean(), d.std(ddof=1) / math.sqrt(n)
    if se == 0:
        return None
    rng = np.random.default_rng(seed)
    ts = []
    for start in range(0, reps, 1000):
        b = rng.integers(0, n, (min(1000, reps - start), n))
        x = d[b]
        sb = x.std(1, ddof=1) / math.sqrt(n)
        with np.errstate(divide="ignore", invalid="ignore"):
            ts.append(np.where(sb > 0, np.abs((x.mean(1) - m) / sb), np.inf))
    q = float(np.quantile(np.concatenate(ts), level))
    if not math.isfinite(q):
        return None
    return [round(float(m - q * se), 4), round(float(m + q * se), 4)]


def graph_procedure(p: dict, w: dict, g: dict, alpha: float = ALPHA) -> dict:
    """Bretz et al. (2009) graphical multiple test: p and w by hypothesis, g[(a, b)] the transition weights; a None p
    counts as 1. Returns the rejected hypotheses."""
    p = {k: (1.0 if v is None else v) for k, v in p.items()}
    w, act, rej = dict(w), set(p), {k: False for k in p}
    G = {(a, b): g.get((a, b), 0.0) for a in p for b in p if a != b}
    while True:
        cand = [j for j in sorted(act) if p[j] <= w[j] * alpha]
        if not cand:
            return rej
        j = min(cand, key=lambda k: (p[k], k))
        rej[j] = True
        act.discard(j)
        for l in act:
            w[l] += w[j] * G[(j, l)]
        newG = {}
        for l in act:
            for k in act:
                if l == k:
                    continue
                den = 1 - G[(l, j)] * G[(j, l)]
                newG[(l, k)] = (G[(l, k)] + G[(l, j)] * G[(j, k)]) / den if den > 0 else 0.0
        G.update(newG)
        w[j] = 0.0


def t_test(d: np.ndarray):
    from scipy.stats import t as tdist
    n = len(d)
    if n < 2 or d.std(ddof=1) == 0:
        return None
    return float(2 * tdist.sf(abs(d.mean() / (d.std(ddof=1) / math.sqrt(n))), n - 1))


LOCAL_LEVELS = (ALPHA / FAMILY_SIZE, ALPHA / 2, ALPHA)  # the local levels the graph can reach: 0.0167, 0.025, 0.05
WARN_CLAIM_TAIL = 0.0125  # diagnostic warning for the claim tail at alpha/3 (1.5 x the equal-split reference 0.0083)
LEVEL_MIN_REPS, LEVEL_MIN_INNER = 4000, 10000  # the plan's minimum for a calibration (fewer: a technical check only)
LEVEL_CHECKS = ("H1/expected_complete_cases", "H1/all_eligible", "H3/expected_complete_cases", "H3/all_eligible")


def calibration_check(d: np.ndarray, n: int, reps: int = 4000, inner: int = FLIPS, seed: int = SEED) -> dict:
    """The level of the directional test on this model's own differences (since 0.7, the project lead's criterion of
    18.11.2026): samples of n drawn from d - mean(d) (a true null with the shape of the observed d), each tested with the
    bootstrap-t (inner resamples, by default as many as the test); counts of rejections in the claim tail (p <= level
    and mean > 0, what the graph rejects at that local level), in the other tail and two-sided, at every local level the
    graph can reach. The simultaneous Monte Carlo bounds and the criterion are computed over all checks of a report."""
    if len(d) < 10 or n < 2:
        return None
    rng = np.random.default_rng(seed)
    c = d - d.mean()
    claim, other, no_p = Counter(), Counter(), 0
    for i in range(reps):
        x = c[rng.integers(0, len(c), n)]
        p = boot_t_test(x, inner, seed + 1 + i)
        if p is None:  # no test (constant resample): no rejection, as in the graph (p = 1), but counted
            no_p += 1
            continue
        for lv in LOCAL_LEVELS:
            if p <= lv:
                (claim if x.mean() > 0 else other)[lv] += 1
    return {"n": n, "reps": reps, "inner": inner, "source_firms": len(d), "replications_without_p": no_p,
            "claim_tail": {f"{lv:.4f}": claim[lv] for lv in LOCAL_LEVELS},
            "other_tail": {f"{lv:.4f}": other[lv] for lv in LOCAL_LEVELS},
            "claim_tail_rate": {f"{lv:.4f}": round(claim[lv] / reps, 5) for lv in LOCAL_LEVELS},
            "two_sided_rate": {f"{lv:.4f}": round((claim[lv] + other[lv]) / reps, 5) for lv in LOCAL_LEVELS},
            "reference_equal_split": {f"{lv:.4f}": round(lv / 2, 5) for lv in LOCAL_LEVELS}}


def h1_family_qualification(checks_out: list, reasons: list) -> dict:
    """Since 0.8: the qualification of the primary family (the three H1 under Holm) from the same bounds, computed with
    the correction for all the report's bounds (not relaxed); the report's ready_for_freezing is left as it is."""
    h1 = [x for x in checks_out if x["check"].split("/")[1] == "H1"]
    return {"bounds": len(h1), "bounds_within_levels": all(x["within_level"] for x in h1) if h1 else None,
            "qualified": (not reasons) and bool(h1) and all(x["within_level"] for x in h1),
            "note": "H1 bounds with the simultaneous correction over all bounds of the report; ready_for_freezing (all 36) "
                    "is unchanged"}


def level_criterion(checks: list, models: list, family_size: int = FAMILY_SIZE, conf: float = 0.95, k_by_model: dict | None = None,
                    split: str = "dev", k_from_first_report: bool = True) -> dict:
    """Simultaneous one-sided Clopper-Pearson upper bounds (Bonferroni over all claim-tail checks of the report) of the
    claim-tail rejection rate; the criterion is met when every bound is at most its local level. Not met means the
    calibration is not sufficiently confirmed (not proof of a faulty test); a changed procedure needs its own check
    before freezing; a met criterion covers the checked scenarios only."""
    from scipy.stats import beta
    rows = [(lab, lv, r["claim_tail"][f"{lv:.4f}"], r["reps"]) for lab, r in checks if r for lv in LOCAL_LEVELS]
    # since 0.7 (project lead's review, 18.11.2026): "met" only for a complete calibration with the plan's replications
    done = {lab for lab, r in checks if r}
    expected = {f"{m}/{c}" for m in models for c in LEVEL_CHECKS}
    reasons = []
    if len(models) < family_size:
        reasons.append(f"{family_size - len(models)} of the {family_size} models of the family missing from this report")
    if expected - done:
        reasons.append(f"checks missing or without data: {sorted(expected - done)}")
    few = sorted(lab for lab, r in checks if r and (r["reps"] < LEVEL_MIN_REPS or r.get("inner", 0) < LEVEL_MIN_INNER))
    if few:
        reasons.append(f"fewer than {LEVEL_MIN_REPS} outer replications or {LEVEL_MIN_INNER} inner resamples: {few}")
    k_by_model = k_by_model or {}
    no_k = sorted(m for m in models if not k_by_model.get(m))
    if no_k:  # a calibration at K = 1 does not approve K = 2 or 3 (project lead's review, 18.11.2026)
        reasons.append(f"no fixed K (--k NAME=K, the K planned for the test) for: {no_k}")
    if split != "dev":
        reasons.append("not development data")
    if not k_from_first_report:
        reasons.append("K not taken from a first report with --k-from (the plan's two-step procedure)")
    nop = sorted(lab for lab, r in checks if r and r.get("replications_without_p", 0) > 0.01 * r["reps"])
    if nop:  # e.g. a constant d: zero rejections would otherwise read as a met criterion
        reasons.append(f"more than 1% of the replications without a p value (no test performed): {nop}")
    if not rows:
        return {"bounds_within_levels": None, "ready_for_freezing": False, "status": "diagnostic",
                "not_ready_because": reasons or ["no checks"], "checks": []}
    m = len(rows)
    q = 1 - (1 - conf) / m
    out = []
    for lab, lv, k, nrep in rows:
        up = 1.0 if k >= nrep else float(beta.ppf(q, k + 1, nrep - k))
        out.append({"check": lab, "level": round(lv, 5), "claim_tail_rejections": k, "reps": nrep,
                    "rate": round(k / nrep, 5), "simultaneous_upper": round(up, 5), "within_level": up <= lv,
                    "warning_over_0.0125": bool(lv == LOCAL_LEVELS[0] and k / nrep > WARN_CLAIM_TAIL)})
    return {"checks": out, "simultaneous_confidence": conf, "bound_quantile_each": round(q, 6), "number_of_bounds": m,
            "expected_bounds": len(expected) * len(LOCAL_LEVELS),
            "K_by_model": {m: k_by_model.get(m) for m in models},
            "bounds_within_levels": all(x["within_level"] for x in out),
            "status": "calibration" if not reasons else "diagnostic",
            "not_ready_because": reasons,
            "ready_for_freezing": (not reasons) and all(x["within_level"] for x in out),
            "h1_family": h1_family_qualification(out, reasons),
            "warnings_alpha_over_3": [x["check"] for x in out if x["warning_over_0.0125"]],
            "rule": "bounds_within_levels: every simultaneous one-sided upper bound of the claim-tail rate is at most its local "
                    "level (diagnostic, any run). ready_for_freezing additionally needs a complete calibration: every model of "
                    "the family, H1 and H3, both sample sizes and the three levels (36 bounds), at least 4000 outer replications "
                    "and 10000 inner resamples, at most 1% replications without a p value, the K fixed per model with --k (the "
                    "K of the test; a result for K = 1 does not approve K = 2 or 3), on development data. Every simultaneous one-sided upper bound of the claim-tail rate is at most its local level "
                    "(0.0167, 0.025, 0.05); 0.0083 (alpha/3 split equally) is the reference, 0.0125 a diagnostic warning; "
                    "not met = calibration not sufficiently confirmed, not proof of a faulty test; a changed procedure needs "
                    "its own check before freezing; a met criterion covers the checked scenarios only"}


ERROR_CLASSES = (("cut by the token limit", "cut by the token limit"), ("refusal", "refusal"), ("incomplete", "incomplete"),
                 ("nested too deeply", "nested too deeply"), ("not one JSON object", "not one JSON object"),
                 ("keys must be", "wrong top-level keys"), ("must have exactly the keys", "wrong rating keys"),
                 ("is not an integer", "a count not an integer in 0-100"), ("all counts are 0", "all counts 0"),
                 ("reason must be", "reason not a text"))


def error_class(err: str) -> str:
    for needle, label in ERROR_CLASSES:
        if needle in err:
            return label
    return "other"


def text_features(t: str) -> dict:
    """Aggregate-only features of a failed answer's raw text (the text itself is never written)."""
    st = t.strip()
    i, j = st.find("{"), st.rfind("}")
    return {"empty": not st, "starts_with_brace": st.startswith("{"), "leading_text": i > 0, "trailing_text": 0 <= j < len(st) - 1,
            "code_fence": "```" in st, "has_firms_per_100": "firms_per_100" in st, "think_tag": "<think" in st.lower(),
            "no_brace": i < 0}


def dist_summary(x: np.ndarray) -> dict:
    """Descriptive distribution of per-firm differences (no claim)."""
    from scipy.stats import skew
    if len(x) == 0:
        return {"n": 0}
    xs = np.sort(x)
    w = max(1, math.ceil(0.05 * len(x)))
    tot = float(x.sum())
    return {"n": int(len(x)), "mean": r4(x.mean()), "sd": r4(x.std(ddof=1)) if len(x) > 1 else None,
            "skewness": r4(skew(x)) if len(x) > 2 and np.std(x) > 0 else None,
            "quantiles": {q: r4(np.quantile(x, float(q) / 100)) for q in ("1", "5", "10", "25", "50", "75", "90", "95", "99")},
            "min": r4(xs[0]), "max": r4(xs[-1]),
            "share_below_minus_1": r4((x < -1).mean()), "share_below_minus_2": r4((x < -2).mean()),
            "share_below_minus_4": r4((x < -4).mean()), "share_above_plus_1": r4((x > 1).mean()),
            "worst_5pct_firms": w, "sum_of_worst_5pct": r4(xs[:w].sum()), "sum_all": r4(tot),
            "mean_without_worst_5pct": r4(xs[w:].mean()) if len(x) > w else None,
            "mean_without_best_5pct": r4(xs[:-w].mean()) if len(x) > w else None}


def holm(ps: dict, alpha: float = ALPHA) -> dict:
    """Step-down Holm; a missing p value (None or NaN) counts as 1, so the order of entries does not matter."""
    clean = {k: (1.0 if v is None or (isinstance(v, float) and math.isnan(v)) else v) for k, v in ps.items()}
    out, k, stop = {}, len(clean), False
    for i, (name, p) in enumerate(sorted(clean.items(), key=lambda x: (x[1], x[0]))):
        rej = (not stop) and p <= alpha / (k - i)
        stop = stop or not rej
        out[name] = rej
    return out


def r4(x):
    return None if x is None else round(float(x), 4)


def nan_paths(o, path: str = "") -> list:
    """Paths of non-finite numbers in a report (a report with one is refused: it marks an inconsistent denominator)."""
    if isinstance(o, dict):
        return [q for k, v in o.items() for q in nan_paths(v, f"{path}/{k}")]
    if isinstance(o, (list, tuple)):
        return [q for i, v in enumerate(o) for q in nan_paths(v, f"{path}/{i}")]
    return [path] if isinstance(o, float) and not math.isfinite(o) else []


def load_run(name: str, d: Path, design: dict, items: set) -> dict:
    meta = json.loads((d / "run.json").read_text())
    cfg = meta["config"]
    if cfg.get("items_sha256") != design["items_sha256"]:
        sys.exit(f"run {name}: another build")
    if cfg.get("parser") != design.get("parser"):
        sys.exit(f"run {name}: parser {cfg.get('parser')!r}, the build's is {design.get('parser')!r}")
    if cfg.get("max_attempts") != MAX_ATTEMPTS:
        sys.exit(f"run {name}: max_attempts {cfg.get('max_attempts')}, the plan says {MAX_ATTEMPTS}")
    last = meta["sessions"][-1]
    if not last.get("finished_at") or last.get("stopped_reason"):
        sys.exit(f"run {name}: the last session did not finish ({last.get('stopped_reason') or 'no finished_at'})")
    valid, attempts, last_outcome, sums, served, n_valid_served, failed = {}, Counter(), {}, {}, Counter(), 0, []
    answers_raw = (d / "answers.jsonl").read_bytes()
    for ln in answers_raw.decode("utf-8").splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        if r.get("ok"):
            attempts[r["item"]] += 1
            if r.get("served_model"):  # every model answer, valid or not (code review 7)
                served[r["served_model"]] += 1
        last_outcome[r["item"]] = r.get("outcome") or ("valid" if r.get("valid") else "invalid")
        if r.get("valid"):
            valid[r["item"]] = np.array([r["answer"]["probabilities"][k] for k in SCALE])
            sums[r["item"]] = r["answer"].get("sum")
            n_valid_served += bool(r.get("served_model"))
        elif r.get("ok"):  # a model answer that failed (kept in memory for --diagnose only; never written out)
            failed.append({"item": r["item"], "outcome": r.get("outcome"), "error": r.get("error") or "", "text": r.get("text") or ""})
    unfinished = [i for i in items if i not in valid and attempts[i] < MAX_ATTEMPTS]
    if unfinished:
        sys.exit(f"run {name}: {len(unfinished)} items neither valid nor with {MAX_ATTEMPTS} model answers")
    if len(served) > 1:  # since 0.7: one run, one served model (project lead's review, 18.11.2026)
        sys.exit(f"run {name}: the API served more than one model {dict(served)}: answers of different models under one alias")
    if cfg.get("api") in ("anthropic", "openai-responses") and valid and n_valid_served != len(valid):
        sys.exit(f"run {name}: {len(valid) - n_valid_served} valid answers without served_model")
    return {"answers": valid, "sums": sums, "model": cfg.get("model"), "api": cfg.get("api"), "digest": cfg.get("model_digest"),
            "served_model": next(iter(served), None), "served_models_all_answers": dict(served), "config": cfg,
            "identity": {"config": {k: cfg.get(k) for k in FROZEN_CONFIG}, "items_sha256": cfg.get("items_sha256"),
                         "model_digest": cfg.get("model_digest"), "served_model": next(iter(served), None),
                         "answers_sha256": sha(answers_raw)},
            "outcomes_of_errors": dict(Counter(last_outcome[i] for i in items if i not in valid)), "failed_attempts": failed,
            "session_materials": [x.get("materials_sha256") for x in meta["sessions"]],
            "session_servers": [server_identity(x.get("server") or {}) for x in meta["sessions"]]}


def choose_lambda(rows: list, y: np.ndarray, feats: list, seed: int = SEED) -> float:
    """The penalty with the best 4-fold band log score, floored as in the final comparison (since 0.7: the same
    scoring function as the report; before, the cross-validation scored without the floor)."""
    fo = SC.folds(rows, 4, seed)
    best = None
    for lam in SC.LAMBDAS:
        s = 0.0
        try:
            for q in range(4):
                a, b = np.where(fo != q)[0], np.where(fo == q)[0]
                pp = logit(rows=[rows[i] for i in a], y=y[a], test=[rows[i] for i in b], feats=feats, lam=lam)
                s += scores(pp, y[b])["band"].sum()
        except SC.FitFailure:
            continue
        if best is None or s > best[1]:
            best = (lam, s)
    if best is None:
        raise SystemExit("no penalty with converged fits")
    return best[0]


def logit(rows: list, y: np.ndarray, test: list, feats: list, lam: float) -> np.ndarray:
    xtr, voc = SC.design(rows, feats, False)
    xte, _ = SC.design(test, feats, False, voc)
    b, c = SC.fit(xtr, y, 10, lam, "safe_analyze")
    z = xte @ b + c
    return np.exp(z - logsumexp(z, 1, keepdims=True))


FROZEN_CONFIG = ("script", "api", "model", "base_url", "max_tokens", "num_ctx", "think", "chat_template_kwargs", "effort",
                 "max_attempts", "temperature", "seed", "parser")  # every key required in a frozen entry (code review 7)


def check_frozen(name: str, r: dict, frozen: dict) -> None:
    """Since 0.7: a run must match the frozen settings of its model (project lead's review, 18.11.2026): the run
    settings in FROZEN_CONFIG, the weights digest (local model) or the served model (API), and a K of at least 1."""
    m = (frozen.get("models") or {}).get(name)
    if m is None:
        sys.exit(f"--frozen has no model {name!r}")
    cfg = r["config"]
    if not isinstance(m.get("config"), dict):
        sys.exit(f"--frozen for {name!r} has no config")
    missing = [k for k in FROZEN_CONFIG if k not in m["config"]]
    if missing:  # a key left out would not be checked at all
        sys.exit(f"--frozen for {name!r} lacks the settings {missing} (write null where a setting does not apply)")
    diff = {k: (cfg.get(k), m["config"][k]) for k in FROZEN_CONFIG if cfg.get(k) != m["config"][k]}
    if diff:
        sys.exit(f"run {name}: settings differ from the frozen ones {diff}")
    if m.get("model_digest") and cfg.get("model_digest") != m["model_digest"]:
        sys.exit(f"run {name}: weights digest {cfg.get('model_digest')} is not the frozen {m['model_digest']}")
    if m.get("served_model") and r.get("served_model") != m["served_model"]:
        sys.exit(f"run {name}: served model {r.get('served_model')!r} is not the frozen {m['served_model']!r}")
    if not (m.get("model_digest") or m.get("served_model")):
        sys.exit(f"--frozen for {name!r} names neither a weights digest nor a served model")
    if not isinstance(m.get("K"), int) or m["K"] < 1:
        sys.exit(f"--frozen for {name!r}: K must be an integer >= 1")


FROZEN_BUILD = ("split", "elicitation", "parser", "seed", "card_key_sha256", "prompt_sha256", "source_sha256", "records")


def check_frozen_build(design: dict, frozen: dict) -> None:
    """Since 0.8: on the test split the build and the materials must be the frozen ones (safe_make_frozen.py): the build's
    split, elicitation, parser, seed, card key and prompt module, and the sha256 of the analysis, build, prompt and runner
    scripts and the card key as they are now; a change after freezing stops the analysis."""
    if not str(frozen.get("status", "")).startswith("frozen"):
        sys.exit(f"--frozen is not approved (status: {frozen.get('status')!r}); safe_make_frozen --approved-on after the review")
    fb, mats = frozen.get("build"), frozen.get("materials_sha256")
    if not isinstance(fb, dict) or not isinstance(mats, dict) or not mats:
        sys.exit("--frozen has no build or materials_sha256")
    diff = {k: (design.get(k), fb.get(k)) for k in FROZEN_BUILD if design.get(k) != fb.get(k)}
    if diff:
        sys.exit(f"the build is not the frozen one {diff}")
    tb = frozen.get("test_build")  # since 0.8 (the project lead, 19.11.2026): this very test set, not another or a part
    if not isinstance(tb, dict):
        sys.exit("--frozen has no test_build: add the test build with safe_make_frozen --test-build before the calls")
    diff = {k: (design.get(k), tb.get(k)) for k in ("items_sha256", "key_json_sha256", "source_sha256", "records", "items")
            if design.get(k) != tb.get(k)}
    if diff:
        sys.exit(f"the build is not the frozen test set {diff}")
    changed = sorted(f for f, h in mats.items() if not (ROOT / f).exists() or sha((ROOT / f).read_bytes()) != h)
    if changed:
        sys.exit(f"materials changed after freezing: {changed}")


RUN_MATERIALS = ("scripts/run_b_k2.py", "scripts/llm_serving_check.py", "scripts/ollama_provenance.py", "scripts/safe_prompt.py")


def server_identity(facts: dict) -> dict:
    """Since 0.8 (code review 11): what of a session's server facts identifies the serving stack - the SDK versions, the
    API version header, the Ollama version and the model's show details and parameters, and the id and creation date the
    Models API reported (a model changed behind an alias usually changes them)."""
    m = facts.get("model") or {}
    show = facts.get("show") or {}
    return {"openai_sdk": facts.get("openai_sdk"), "anthropic_sdk": facts.get("anthropic_sdk"),
            "anthropic_version": facts.get("anthropic_version"), "ollama_version": facts.get("ollama_version"),
            "ollama_show": {"details": show.get("details"), "parameters": show.get("parameters")} if show else None,
            "model_id": m.get("id"), "model_created": m.get("created") if m.get("created") is not None else m.get("created_at")}


def check_run_materials(name: str, r: dict, frozen: dict) -> None:
    """Since 0.8: on the test split every session of a run must have recorded the sha256 of the runner's modules
    (run_b_k2 0.9.5) equal to the frozen ones, so a change of the runner, the request builder or the parser for the time
    of a run is seen even if it was undone before the analysis."""
    mats = frozen.get("materials_sha256") or {}
    for i, sm in enumerate(r["session_materials"]):
        if not isinstance(sm, dict) or not sm:
            sys.exit(f"run {name}: session {i + 1} recorded no materials (a runner before 0.9.5)")
        if set(sm) != set(RUN_MATERIALS):  # the whole set (code review 11, D1)
            sys.exit(f"run {name}: session {i + 1} recorded {sorted(sm)}, not the runner's modules {list(RUN_MATERIALS)}")
        diff = sorted(f for f, h in sm.items() if mats.get(f) != h)
        if diff:
            sys.exit(f"run {name}: session {i + 1} ran with materials other than the frozen ones: {diff}")
    want = ((frozen.get("models") or {}).get(name) or {}).get("server")
    if not isinstance(want, dict):
        sys.exit(f"--frozen has no server identity of {name} (safe_make_frozen --pilot-run)")
    for i, sv in enumerate(r["session_servers"]):  # code review 11, D2: the serving stack of every session
        diff = {k: (sv.get(k), want.get(k)) for k in set(sv) | set(want) if sv.get(k) != want.get(k)}
        if diff:
            sys.exit(f"run {name}: session {i + 1} ran on another serving stack than the frozen one {diff}")


def mde(sd: float, n: int, alpha: float, power: float = 0.8) -> float:
    return (normal.ppf(1 - alpha / 2) + normal.ppf(power)) * sd / math.sqrt(n)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--run", action="append", default=[], help="NAME=DIR, one per model, names unique")
    ap.add_argument("--safe", required=True)
    ap.add_argument("--audit", default=str(ROOT / "data/processed/safe_validation_audit_v0.1.json"),
                    help="for the test wave's size shares (no answers to the question are read from it)")
    ap.add_argument("--test-firms", type=int, default=692, help="the number of test-wave firms for the minimum detectable "
                    "effect (2026Q3 records answering q0b_3 with 1-10)")
    ap.add_argument("--average-draws", action="store_true", help="pilot diagnostic only: average the band log score over "
                    "all designed draws of each card (K differs between firms); not the test")
    ap.add_argument("--k", action="append", default=[], help="NAME=K: the fixed K of a model (plan F1): the first K draws "
                    "of full and segment are averaged and the build must have them for every firm; on the test split K "
                    "comes from --frozen")
    ap.add_argument("--settings-like", default=None, help="since 0.7: an earlier report (e.g. the second pilot's): every "
                    "run must have that report's settings of the same model (FROZEN_CONFIG but the runner version), weights "
                    "digest and served model (the third pilot: the same settings as the second)")
    ap.add_argument("--k-from", default=None, help="the first report (made without --level-check, --k and --average-draws, on "
                    "this build and these runs): K of every model is its stability.k_rule.recommended_K (the plan's two-step "
                    "procedure; required for ready_for_freezing)")
    ap.add_argument("--frozen", default=None, help="the frozen settings (JSON): per model name the model, API, served model "
                    "or weights digest, run settings and K; required on the test split")
    ap.add_argument("--level-check", action="store_true", help="also run the plan's level check: H1 and H3 of each model, "
                    "their own centred differences, at the expected complete cases and at --test-firms, claim tail at "
                    "0.0167 / 0.025 / 0.05, simultaneous Clopper-Pearson bounds (slow: with the defaults about 3-4 minutes "
                    "per hypothesis and sample size, about 45 minutes for three models)")
    ap.add_argument("--diagnose", action="store_true", help="development builds only: add a descriptive diagnosis per model "
                    "(distribution of d and d3, floor hits, concentration of the answers, technical errors by class, raw-text "
                    "features, repetition across draws, size and sector); aggregates only, no claim, other numbers unchanged")
    ap.add_argument("--level-reps", type=int, default=4000, help="outer replications of the level check (the plan: at least "
                    "4000; about 3-4 minutes per hypothesis and sample size at n = 692 with 10,000 inner resamples)")
    ap.add_argument("--level-inner", type=int, default=FLIPS, help="bootstrap resamples inside each replication (the plan: as "
                    "many as the test, 10,000)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    b = Path(a.build)
    design = json.loads((b / "design.json").read_text())
    if design.get("parser") == "safe_by_item":
        sys.exit("a build of the elicitation comparison (arms): use safe_elicitation_compare.py")
    if a.k and a.average_draws:
        sys.exit("--k and --average-draws exclude each other")
    k_of = {}
    for s_ in a.k:
        n_, _, v_ = s_.partition("=")
        if not n_ or not v_.isdigit() or int(v_) < 1:
            sys.exit(f"--k {s_!r}: NAME=K with K >= 1")
        k_of[n_] = int(v_)
    frozen = json.loads(Path(a.frozen).read_text()) if a.frozen else None
    if design["split"] == "test" and a.diagnose:
        sys.exit("--diagnose is for development builds only: refused on the test split")
    if design["split"] == "test" and a.level_check:  # the calibration belongs to the development data, before freezing
        sys.exit("--level-check is for development builds only (the calibration before freezing): refused on the test split")
    if design["split"] == "test":  # since 0.7: the test runs only under the frozen settings
        if frozen is None or a.k or a.average_draws:
            sys.exit("the test split needs --frozen (K from it) and takes neither --k nor --average-draws")
        check_frozen_build(design, frozen)
    if sha(gzip.open(b / "items.jsonl.gz").read()) != design["items_sha256"]:
        sys.exit("items.jsonl.gz does not match design.json")
    if sha((b / "key.json").read_bytes()) != design["key_json_sha256"]:
        sys.exit("key.json does not match design.json")
    key = json.loads((b / "key.json").read_text())
    names = [s.partition("=")[0] for s in a.run]
    if len(names) != len(set(names)) or not all(names):
        sys.exit("--run names must be unique and non-empty")
    if len(names) > FAMILY_SIZE:
        sys.exit(f"the plan's family has {FAMILY_SIZE} models")
    raw = Path(a.safe).read_bytes()
    if sha(raw) != design["source_sha256"]:
        sys.exit("--safe is not the extract the build was made from")
    dirs = [Path(s.partition("=")[2]).resolve() for s in a.run]
    if len(dirs) != len(set(dirs)):
        sys.exit("two --run entries point to the same directory")
    seen = Counter((k["wave"], k["permid"], k["variant"], k["repeat"]) for k in key.values())
    if any(c > 1 for c in seen.values()):  # since 0.7: a duplicated draw would overwrite another silently
        sys.exit(f"key.json: {sum(c > 1 for c in seen.values())} firm-card-draws appear more than once")
    runs = {s.partition("=")[0]: load_run(s.partition("=")[0], Path(s.partition("=")[2]), design, set(key)) for s in a.run}
    if frozen is not None:
        for n_, r_ in runs.items():
            check_frozen(n_, r_, frozen)
            if design["split"] == "test":
                check_run_materials(n_, r_, frozen)
            k_of[n_] = int(frozen["models"][n_]["K"])
    settings_source = None
    if a.settings_like:
        sl_raw = Path(a.settings_like).read_bytes()
        sl = json.loads(sl_raw)
        if (sl.get("build") or {}).get("items_sha256") == design["items_sha256"]:  # code review 9, D1
            sys.exit("--settings-like: a report of this same build is no reference for its settings")
        for n_, r_ in runs.items():
            fm = (sl.get("models") or {}).get(n_)
            if fm is None or fm.get("run_identity") is None:
                sys.exit(f"--settings-like: no run_identity of model {n_} in that report")
            ri, now = fm["run_identity"], r_["identity"]
            diff = {k: (ri["config"].get(k), now["config"].get(k)) for k in FROZEN_CONFIG
                    if k != "script" and ri["config"].get(k) != now["config"].get(k)}
            for k in ("model_digest", "served_model"):
                if ri.get(k) != now.get(k):
                    diff[k] = (ri.get(k), now.get(k))
            if diff:
                sys.exit(f"--settings-like: run {n_} differs from that report's settings {diff}")
        settings_source = {"report_sha256": sha(sl_raw), "report_build_items_sha256": (sl.get("build") or {}).get("items_sha256"),
                           "compared": [k for k in FROZEN_CONFIG if k != "script"]
                           + ["model_digest", "served_model"]}
    k_source = None
    if a.k_from:  # since 0.7 (project lead, 18.11.2026): K comes from the first report, never re-chosen after the calibration
        if a.k or a.average_draws or frozen is not None:
            sys.exit("--k-from excludes --k, --average-draws and --frozen")
        first_raw = Path(a.k_from).read_bytes()
        first = json.loads(first_raw)
        if first.get("build", {}).get("items_sha256") != design["items_sha256"]:
            sys.exit("--k-from: the first report is of another build")
        if first.get("level_check") is not None or first.get("draws_by_model"):
            sys.exit("--k-from: the first report must be made without --level-check and --k")
        if set(first.get("models", {})) != set(runs):
            sys.exit(f"--k-from: the first report has the models {sorted(first.get('models', {}))}, here {sorted(runs)}")
        for n_, r_ in runs.items():
            fm = first["models"][n_]
            if fm.get("draws") != "first draw (K = 1)":
                sys.exit(f"--k-from: model {n_} of the first report was not analysed on the first draw ({fm.get('draws')})")
            if fm.get("run_identity") is None:
                sys.exit(f"--k-from: model {n_} of the first report has no run_identity (a report before 0.7)")
            if fm["run_identity"] != r_["identity"]:  # the same run: settings, weights, served model and the answers file
                diff = sorted(k for k in r_["identity"] if fm["run_identity"].get(k) != r_["identity"][k])
                sys.exit(f"--k-from: model {n_} is not the run of the first report (differs in {diff})")
            kk = ((fm.get("stability") or {}).get("k_rule") or {}).get("recommended_K")
            if not isinstance(kk, int) or kk < 1:
                sys.exit(f"--k-from: the first report has no recommended_K for {n_}")
            k_of[n_] = kk
        k_source = {"first_report_sha256": sha(first_raw), "first_report_script": first.get("script"),
                    "recommended_K": dict(k_of)}
    unknown = set(k_of) - set(runs)
    if unknown:
        sys.exit(f"--k for models without a run: {sorted(unknown)}")
    ids = [(r["model"], r["digest"]) for r in runs.values()]
    if len(ids) != len(set(ids)):
        sys.exit("two runs of the same model: the family needs three different models")
    firms = sorted({(k["wave"], k["permid"]) for k in key.values()})
    info = {(k["wave"], k["permid"]): k for k in key.values()}
    y = np.array([int(info[f]["answer"]) - 1 for f in firms])
    size = {f: info[f]["size"] for f in firms}
    by = defaultdict(dict)
    for iid, k in key.items():
        by[(k["wave"], k["permid"])][(k["variant"], k["repeat"])] = iid
    rows = [r for r in csv.DictReader(gzip.decompress(raw).decode("utf-8").splitlines()) if r["wave"] in SC.WAVES]
    test_firms = {r["permid"] for r in rows if r["wave"] == SC.TEST}
    dev = [r for r in rows if r["wave"] != SC.TEST and r["permid"] not in test_firms]
    evaluated = {p for _, p in firms}
    train = [r for r in dev if (design["split"] == "test" or r["permid"] not in evaluated) and SC.norm(r["q0b_3"]) in SCALE]
    if len(train) < 100:
        sys.exit(f"{len(train)} training records: a development build must be a sample (--sample), not the whole sample")
    ytr = np.array([int(SC.norm(r["q0b_3"])) - 1 for r in train])
    rec = {(SC.WAVES[r["wave"]], r["permid"]): r for r in rows}
    erows = [rec[f] for f in firms]
    def _w(v):
        v = SC.norm(v)
        try:
            return float(v)
        except ValueError:
            return float("nan")
    wcommon = np.array([_w(r.get("wgtcommon")) for r in erows])
    feats = {"logit": SC.CARD, "logit_segment": SEGMENT_FIELDS, "logit_without_applications": NO_APPLICATIONS}
    lams = {m: choose_lambda(train, ytr, f) for m, f in feats.items()}
    stat = {"marginal": SC.predict(train, ytr, erows, 10, "marginal"), "rule": SC.predict(train, ytr, erows, 10, "rule")}
    stat.update({m: logit(train, ytr, erows, feats[m], lams[m]) for m in feats})
    stat_sc = {m: scores(p, y) for m, p in stat.items()}
    shares = json.loads(Path(a.audit).read_text())["waves"]["2026q3"]["size"]
    tot = sum(shares.values())
    size_share = {{"1-9": "1", "10-49": "2", "50-249": "3", "250+": "4"}[k]: v / tot for k, v in shares.items()}
    out = {"script": f"safe_analyze {VERSION}", "script_sha256": sha(Path(__file__).read_bytes()),
           "plan_sha256": sha((ROOT / "data/reference/safe_validation_plan.json").read_bytes()),
           "frozen_sha256": sha(Path(a.frozen).read_bytes()) if a.frozen else None, "draws_by_model": dict(k_of),
           "k_source": k_source, "settings_source": settings_source,
           "conventions": "log score: higher is better; log-loss: its negative; d and 'minus': differences of log scores",
           "build": {k: design.get(k) for k in ("split", "items_sha256", "records", "items", "card_key_sha256", "elicitation",
                                                "sample_by_size",
                                                "prompt_sha256", "target_answers_in_split")},
           "source_sha256": sha(raw), "audit_sha256": sha(Path(a.audit).read_bytes()), "firms": len(firms), "answers_1_10": dict(Counter(str(v + 1) for v in y)),
           "statistical": {"training_records": len(train), "penalties": lams,
                           "band_logloss": {m: r4(-s["band"].mean()) for m, s in stat_sc.items()},
                           "ten_logloss": {m: r4(-s["ten"].mean()) for m, s in stat_sc.items()},
                           "rps": {m: r4(s["rps"].mean()) for m, s in stat_sc.items()}},
           "models": {}}
    pvals, h3, level_checks = {}, {}, []
    for name, r in runs.items():
        ans = r["answers"]
        res = {"model": r["model"], "api": r["api"], "technical_errors": {}, "technical_error_outcomes": r["outcomes_of_errors"]}
        K = k_of.get(name)

        def draws_of(f, v):
            if v == "shuffled":
                return [0]
            if K is not None:
                return list(range(K))
            return sorted(k for (vv, k) in by[f] if vv == v) if a.average_draws else [0]

        if K is not None:
            short = sum(1 for f in firms for v in ("full", "segment") if any((v, k) not in by[f] for k in range(K)))
            if short:
                sys.exit(f"run {name}: K = {K}, but the build lacks draws for {short} firm-cards of full or segment")
        mode = (f"fixed K = {K}" if K is not None else "all designed draws (pilot diagnostic, K differs between firms)"
                if a.average_draws else "first draw (K = 1)")
        res["draws"] = mode
        res["served_model"] = r.get("served_model")
        res["run_identity"] = r["identity"]  # since 0.7: what --k-from compares between the two analyses
        res["served_models_all_answers"] = r.get("served_models_all_answers")

        def raw_sum(it):
            x = r["sums"].get(it)
            return None if x is None else x * (1 if design.get("parser") == "safe_counts" else 100)

        def draws(i: int, v: str, drop_sums: bool = False):
            """The answers of firm i to card v over the draws used (the first, or all with --average-draws), as a
            K x 10 array; None when any of these draws is a technical error (since 0.6: with K draws a firm-card is an
            error when any draw is) or, with drop_sums, when any draw's raw sum is outside SUM_RANGE."""
            f, got = firms[i], []
            for k in draws_of(firms[i], v):
                it = by[f].get((v, k))
                if it not in ans:
                    return None
                x = raw_sum(it)
                if drop_sums and x is not None and not (SUM_RANGE[0] <= x <= SUM_RANGE[1]):
                    return None
                got.append(ans[it])
            return np.array(got) if got else None

        D = {v: [draws(i, v) for i in range(len(firms))] for v in VARIANTS}
        ok = {v: np.array([x is not None for x in D[v]], dtype=bool) for v in VARIANTS}
        res["technical_errors"] = {v: int((~ok[v]).sum()) for v in VARIANTS}
        res["technical_error_definition"] = "a firm-card is an error when any draw used is invalid"
        sc = {}  # the first draw of each firm-card without errors (calibration, floor, stability)
        for v in VARIANTS:
            sc[v] = np.full((len(firms), 10), np.nan)
            for i in np.where(ok[v])[0]:
                sc[v][i] = D[v][i][0]
        all_sums = {}
        for v in VARIANTS:
            vals = []
            for i, f in enumerate(firms):
                for k in draws_of(f, v):
                    it = by[f].get((v, k))
                    if it in ans and raw_sum(it) is not None:
                        vals.append(raw_sum(it))
            all_sums[v] = vals
            if vals:
                va = np.array(vals, dtype=float)
                res.setdefault("raw_sums", {})[v] = {
                    "answers": len(vals), "share_exactly_100": r4((va == 100).mean()),
                    "share_outside_80_120": r4(((va < 80) | (va > 120)).mean()),
                    f"share_outside_{SUM_RANGE[0]}_{SUM_RANGE[1]}": r4(((va < SUM_RANGE[0]) | (va > SUM_RANGE[1])).mean()),
                    "min": r4(va.min()), "median": r4(np.median(va)), "max": r4(va.max())}
        res["raw_sums_draws"] = ("the draws used (" + mode + "), valid answers also of firm-cards that are errors: they "
                                 "describe the answers, not the denominator")

        def arm(v: str, kind: str = "band", fl: float = FLOOR, drop_sums: bool = False) -> np.ndarray:
            """Per-firm score of a card (kind band, ten or rps; floor fl): the mean over the draws used, NaN on errors."""
            outv = np.full(len(firms), np.nan)
            for i in range(len(firms)):
                P = draws(i, v, True) if drop_sums else D[v][i]
                if P is not None:
                    outv[i] = float(scores(P, np.full(len(P), y[i]), fl)[kind].mean())
            return outv
        share = {v: res["technical_errors"][v] / len(firms) for v in res["technical_errors"]}
        rng = np.random.default_rng(SEED)
        e_full, e_seg = (~ok["full"]).astype(float), (~ok["segment"]).astype(float)
        bs = [(lambda i: e_full[i].mean() - e_seg[i].mean())(rng.integers(0, len(firms), len(firms))) for _ in range(BOOT)]
        res["technical_error_shares"] = {v: r4(x) for v, x in share.items()}
        lo_e, hi_e = np.quantile(bs, 0.025), np.quantile(bs, 0.975)
        differential = bool(lo_e > 0 or hi_e < 0)
        res["technical_error_flag"] = {"over_threshold": any(x > ERROR_FLAG for x in share.values()), "threshold": ERROR_FLAG,
                                       "full_minus_segment": r4(e_full.mean() - e_seg.mean()), "ci95": [r4(lo_e), r4(hi_e)],
                                       "differential": differential,
                                       "equal_footing_sensitivity": ("errors_against_the_claim" if differential or
                                                                     any(x > ERROR_FLAG for x in share.values()) else None),
                                       "note": "with differential errors only the against-the-claim sensitivity stands on equal "
                                               "footing with the main result; the symmetric worst case then favours the claim"}
        A_full, A_seg = arm("full"), arm("segment")
        bi = np.where(~np.isnan(A_full) & ~np.isnan(A_seg))[0]
        res["coverage"] = {"eligible_firms": len(firms), "firms_in_the_main_comparison": int(len(bi)),
                           "share": r4(len(bi) / len(firms)),
                           "note": "the main result is the mean d among the firms with all used draws of both cards valid "
                                   "(complete cases at this K); a larger K or a model that fails more often removes more "
                                   "firms, so read it with the coverage and the error sensitivities"}
        sens = {}
        marg, worst = stat_sc["marginal"]["band"], worst_band(y)
        for label, f_full, f_seg in (("errors_as_marginal", marg, marg), ("errors_worst_symmetric", worst, worst),
                                     ("errors_against_the_claim", worst, marg)):
            full_all = np.where(~np.isnan(A_full), A_full, f_full)
            seg_all = np.where(~np.isnan(A_seg), A_seg, f_seg)
            dd = full_all - seg_all
            sens[label] = {"firms": len(firms), "mean_d": r4(dd.mean()), "p": boot_t_test(dd)}
        if len(bi) == 0:  # since 0.7 the error sensitivities are computed also here (project lead's review)
            res["primary"] = {"firms": 0, "mean_d": None, "ci95": None, "p": None, "sensitivity": sens,
                              "note": "no common firm: no main result; the technical-error diagnostics and the error "
                                      "sensitivities (all eligible firms) stay in the report"}
            pvals[name] = None
            out["models"][name] = res
            h3[name] = None
            continue
        d = A_full[bi] - A_seg[bi]
        p = boot_t_test(d) if len(bi) >= MIN_FIRMS else None
        pvals[name] = None if p is None else (p if d.mean() > 0 else 1.0)  # directional: a rejection only with d > 0
        Bf, Bs = arm("full", drop_sums=True), arm("segment", drop_sums=True)
        bb = np.where(~np.isnan(Bf) & ~np.isnan(Bs))[0]
        pooled_sums = [x for v in VARIANTS for x in all_sums[v]]
        out_share = (np.mean([not (SUM_RANGE[0] <= x <= SUM_RANGE[1]) for x in pooled_sums]) if pooled_sums else 0.0)
        sens["without_sums_outside_range"] = {"firms": int(len(bb)), "mean_d": r4((Bf[bb] - Bs[bb]).mean()) if len(bb) else None,
                                              "p": boot_t_test(Bf[bb] - Bs[bb]) if len(bb) >= MIN_FIRMS else None,
                                              "share_of_answers_outside": r4(out_share),
                                              "equal_footing": bool(out_share > SUM_FLAG),
                                              "note": f"answers with a raw sum outside {SUM_RANGE} treated as technical errors (plan F4)"}
        for fl in FLOORS_SENSITIVITY:
            dd = arm("full", fl=fl)[bi] - arm("segment", fl=fl)[bi]
            sens[f"floor_{fl}"] = {"firms": int(len(bi)), "mean_d": r4(dd.mean()), "note": "explanatory, no claim"}
        wi = wcommon[bi]
        wmean = r4(np.nansum(wi * d) / np.nansum(np.where(np.isnan(wi), 0, wi))) if np.nansum(wi) > 0 else None
        from scipy.stats import skew
        res["primary"] = {"firms": int(len(bi)), "mean_d": r4(d.mean()), "ci95": boot_t_ci(d), "ci95_percentile": boot_ci(d),
                          "p": p, "p_note": None if len(bi) >= MIN_FIRMS else f"fewer than {MIN_FIRMS} common firms: no test",
                          "draws": mode,
                          "estimand": "mean d among firms with all used draws of both cards valid (complete cases)",
                          "test": "two-sided bootstrap-t", "p_sign_flip": flip_test(d), "p_t": t_test(d),
                          "skewness_of_d": r4(skew(d)) if len(d) > 2 and np.std(d) > 0 else None,
                          "sensitivity": sens,
                          "mean_d_weighted_wgtcommon": wmean,
                          "weighted_note": "explanatory; SAFE weights are scaled within a wave, so in a development build "
                                           "(two waves) the weighted mean mixes two scales; in the test wave it does not"}
        res["references"] = {}
        for v in ("full", "segment", "shuffled"):
            vi = np.where(ok[v])[0]
            if len(vi):
                sv = arm(v)[vi]
                res["references"][v] = {"firms": int(len(vi)), "band_logloss": r4(-sv.mean()),
                                        **{f"minus_{m}": r4((sv - stat_sc[m]["band"][vi]).mean()) for m in stat_sc},
                                        "minus_marginal_ci95": boot_t_ci(sv - stat_sc["marginal"]["band"][vi])}
        res["ten_logloss"] = {v: r4(-arm(v, "ten")[bi].mean()) for v in ("full", "segment")}
        res["rps"] = {v: r4(arm(v, "rps")[bi].mean()) for v in ("full", "segment")}
        sh = np.where(ok["full"] & ok["shuffled"])[0]
        if len(sh):
            dd = A_full[sh] - arm("shuffled")[sh]
            res["full_minus_shuffled"] = {"firms": int(len(sh)), "mean": r4(dd.mean()), "ci95": boot_ci(dd),
                                          "note": "diagnostic, no test; an interval that includes 0 leaves open whether matching the "
                                                  "other features to this firm improves the score"}
        res["floor_acts"] = {v: r4((band_probs(sc[v][np.where(ok[v])[0]])[np.arange(int(ok[v].sum())), BAND_OF[y[ok[v]]]] < FLOOR).mean())
                             for v in sc if ok[v].any()}
        res["floor_acts_draws"] = "first draw"
        cal = {}
        for v in ("full", "segment"):
            vi = np.where(ok[v])[0]
            pb = band_probs(floor(sc[v][vi]))
            cal[v] = {"all": {bd: {"mean_predicted": r4(pb[:, k].mean()), "observed": r4((BAND_OF[y[vi]] == k).mean())}
                              for k, bd in enumerate(BANDS)}}
            for sz in sorted({size[f] for f in firms}):
                si = [j for j, i in enumerate(vi) if size[firms[i]] == sz]
                if si:
                    cal[v][sz] = {bd: {"mean_predicted": r4(pb[si, k].mean()), "observed": r4((BAND_OF[y[vi][si]] == k).mean())}
                                  for k, bd in enumerate(BANDS)}
        cal["draws"] = "first draw"
        res["calibration"] = cal
        res["by_size"], sds = {}, {}
        for sz in sorted({size[f] for f in firms}):
            idx = [j for j, i in enumerate(bi) if size[firms[i]] == sz]
            if idx:
                sds[sz] = float(np.std(d[idx], ddof=1)) if len(idx) > 1 else None
                res["by_size"][sz] = {"firms": len(idx), "mean_d": r4(d[idx].mean()), "sd_d": r4(sds[sz]),
                                      "full_minus_marginal": r4((A_full[bi][idx] - stat_sc["marginal"]["band"][bi][idx]).mean())}
        mus = {sz: float(np.mean(d[[j for j, i in enumerate(bi) if size[firms[i]] == sz]])) for sz in sds}
        # since 0.7: coverage by size class, carried to the test wave's size shares (project lead's review): the expected
        # firms in the test comparison are sum_k n share_k cov_k, and the retained sample has the shares share_k cov_k
        elig = Counter(size[f] for f in firms)
        kept = Counter(size[firms[i]] for i in bi)
        cov = {k: kept[k] / elig[k] for k in elig}
        cov_all = len(bi) / len(firms)
        cov_t = {k: cov.get(k, cov_all) for k in size_share}  # a class absent from the pilot: the overall coverage (flagged)
        n_test = a.test_firms
        n_eff = max(1, round(sum(n_test * size_share[k] * cov_t[k] for k in size_share)))
        keys_ = [k for k in sds if sds[k] is not None and k in size_share and cov_t[k] > 0]
        wsum = sum(size_share[k] * cov_t[k] for k in keys_)
        if wsum > 0:  # the total variance of d over the retained test sample's size mix (within + between classes),
            w_ = {k: size_share[k] * cov_t[k] / wsum for k in keys_}  # the variance an unstratified bootstrap sees
            mu_w = sum(w_[k] * mus[k] for k in keys_)
            within = sum(w_[k] * sds[k] ** 2 for k in keys_)
            between = sum(w_[k] * (mus[k] - mu_w) ** 2 for k in keys_)
            total = math.sqrt(within + between)
        else:
            within = between = total = None
        res["precision"] = {"sd_d_by_size": {k: r4(v) for k, v in sds.items()}, "mean_d_by_size": {k: r4(v) for k, v in mus.items()},
                            "coverage_by_size": {k: r4(v) for k, v in cov.items()},
                            "size_classes_without_pilot_coverage": sorted(k for k in size_share if k not in cov),
                            "sd_d_total_at_test_sizes": r4(total),
                            "variance_within_classes": r4(within), "variance_between_classes": r4(between),
                            "size_classes_without_sd": sorted(k for k in size_share if k not in keys_),
                            "test_firms_assumed": n_test, "coverage_of_this_run": r4(cov_all),
                            "test_firms_expected_in_the_comparison": n_eff,
                            "mde_80_alpha_0.05_over_3": r4(mde(total, n_eff, ALPHA / FAMILY_SIZE)) if total else None,
                            "mde_80_alpha_0.05": r4(mde(total, n_eff, ALPHA)) if total else None,
                            "note": "since 0.7 the SD is the total SD of d over the size mix of the retained test sample "
                                    "(test size shares times this run's coverage by size class; class SDs and class means, "
                                    "as the unstratified bootstrap of the test sees them) and n the sum over classes of the "
                                    "test firms times the coverage of that class; test firms from --test-firms, size shares "
                                    "from the audit (no answers read)"}
        stab = {}
        for v in ("full", "segment"):
            reps = [(i, k) for i, f in enumerate(firms) for (vv, k) in by[f] if vv == v and k > 0
                    and by[f][(vv, k)] in ans and ok[v][i]]
            if reps:
                pr = np.array([ans[by[firms[i]][(v, k)]] for i, k in reps])
                p0 = np.array([sc[v][i] for i, _ in reps])
                yi = np.array([y[i] for i, _ in reps])
                diff = scores(pr, yi)["band"] - scores(p0, yi)["band"]
                stab[v] = {"pairs": len(reps), "mean_abs_diff_of_probabilities": r4(np.abs(pr - p0).mean()),
                           "mean_abs_diff_of_band_log_score": r4(np.abs(diff).mean())}
        both_rep = [i for i, f in enumerate(firms) if ok["full"][i] and ok["segment"][i]
                    and by[f].get(("full", 1)) in ans and by[f].get(("segment", 1)) in ans]
        if len(both_rep) > 1:  # the noise of d itself: two independent draws of d on the same firms
            yi = y[both_rep]
            d0 = scores(sc["full"][both_rep], yi)["band"] - scores(sc["segment"][both_rep], yi)["band"]
            d1 = (scores(np.array([ans[by[firms[i]][("full", 1)]] for i in both_rep]), yi)["band"]
                  - scores(np.array([ans[by[firms[i]][("segment", 1)]] for i in both_rep]), yi)["band"])

            def var_d(j):  # var(d) of one draw, from both draws (since 0.6)
                return (np.var(d0[j], ddof=1) + np.var(d1[j], ddof=1)) / 2

            allj = np.arange(len(both_rep))
            noise = float(np.var(d1 - d0, ddof=1) / 2)
            rng = np.random.default_rng(SEED)
            rs = []
            for _ in range(BOOT):
                j = rng.integers(0, len(both_rep), len(both_rep))
                vd = var_d(j)
                if vd > 0:
                    rs.append(np.var((d1 - d0)[j], ddof=1) / 2 / vd)
            vd_all = var_d(allj)
            r_hat = noise / vd_all if vd_all > 0 else None
            r_up = float(np.quantile(rs, 0.8)) if rs else None
            if len(both_rep) < K_MIN_FIRMS or r_up is None:
                k_rec, basis = 3, f"fewer than {K_MIN_FIRMS} firms with repeats of both cards (or no variance): the conservative K"
            else:
                k_rec, basis = (1 if r_up < 0.2 else 2 if r_up < 0.5 else 3), "the 80% upper bound"
            stab["k_rule"] = {"firms": len(both_rep), "share_noise": r4(r_hat), "share_noise_upper_80": r4(r_up),
                              "recommended_K": k_rec, "basis": basis,
                              "rule": f"K = 1 if the 80% upper bound < 0.2, 2 if < 0.5, else 3, with at least {K_MIN_FIRMS} "
                                      "firms (plan F1); the test averages the band log scores of the K draws of each card"}
            stab["d"] = {"firms": len(both_rep), "noise_variance_of_one_draw_of_d": r4(noise),
                         "variance_of_d_on_these_firms": r4(vd_all),
                         "share_of_var_d_from_model_noise": r4(r_hat),
                         "note": "from repeats of both cards on the same firms; var(d) of one draw is the mean of the two "
                                 "draws' variances; averaging K draws divides the noise part by K"}
        designed = sum(1 for f in firms if ("full", 1) in by[f] and ("segment", 1) in by[f])
        if designed and len(both_rep) <= 1:  # since 0.6: the fallback also below two firms (user review, 18.11.2026)
            stab["k_rule"] = {"firms": len(both_rep), "firms_with_repeats_in_the_build": designed, "share_noise": None,
                              "share_noise_upper_80": None, "recommended_K": 3,
                              "basis": f"fewer than {K_MIN_FIRMS} firms with valid repeats of both cards: the conservative K",
                              "rule": f"K = 1 if the 80% upper bound < 0.2, 2 if < 0.5, else 3, with at least {K_MIN_FIRMS} "
                                      "firms (plan F1)"}
        if stab:
            if r["api"] == "ollama":
                stab["note"] = ("local model at temperature 0 with a fixed seed: near-deterministic, not exactly (the recorded "
                                "repeats differ); a retry after an invalid answer usually repeats it")
            res["stability"] = stab
        hi3 = np.where(~np.isnan(A_full))[0]
        d3 = A_full[hi3] - stat_sc["logit_segment"]["band"][hi3]
        p3 = boot_t_test(d3) if len(hi3) >= MIN_FIRMS else None
        h3[name] = None if p3 is None else (p3 if d3.mean() > 0 else 1.0)
        res["h3_full_vs_logit_segment"] = {"firms": int(len(hi3)), "mean": r4(d3.mean()) if len(hi3) else None,
                                           "ci95": boot_t_ci(d3) if len(hi3) >= MIN_FIRMS else None, "p": p3,
                                           "note": "exploratory since 19.11.2026: the full card vs the statistical size x "
                                                   "sector segmentation (logit on size and sector); estimates and intervals, "
                                                   "no confirmatory claim, no level passed to H1"}
        if a.diagnose:  # since 0.7 (project lead, 18.11.2026): a fixed descriptive scope, aggregates only, no claim
            dg = {"scope": "descriptive diagnosis of the development pilot; no test, no claim; the main numbers are unchanged",
                  "d_H1": dist_summary(d), "d3_H3": dist_summary(d3)}
            p_full_true = np.full(len(firms), np.nan)  # the first draw's probability of the true band, full card
            for i in np.where(ok["full"])[0]:
                p_full_true[i] = band_probs(sc["full"][i][None, :])[0, BAND_OF[y[i]]]
            hit = p_full_true < FLOOR  # the full card gave the observed band (almost) nothing
            for lab, idx, dv in (("H1", bi, d), ("H3", hi3, d3)):
                h_ = hit[idx]
                dg[f"floor_hits_{lab}"] = {"firms": int(h_.sum()), "share": r4(h_.mean()) if len(h_) else None,
                                           "mean_d_hits": r4(dv[h_].mean()) if h_.any() else None,
                                           "mean_d_others": r4(dv[~h_].mean()) if (~h_).any() else None,
                                           "share_of_the_sum_of_d": r4(dv[h_].sum() / dv.sum()) if h_.any() and dv.sum() != 0 else None,
                                           "by_size": {sz: int(sum(1 for j, i in enumerate(idx) if h_[j] and size[firms[i]] == sz))
                                                       for sz in sorted({size[f] for f in firms})},
                                           "true_band": {bd: int(sum(1 for j, i in enumerate(idx) if h_[j] and BAND_OF[y[i]] == k))
                                                         for k, bd in enumerate(BANDS)}}
            conc = {}
            for v in VARIANTS:
                vi = np.where(ok[v])[0]
                if not len(vi):
                    continue
                P = sc[v][vi]
                pb = band_probs(P)
                ent = -(np.where(P > 0, P * np.log(np.clip(P, 1e-12, 1)), 0)).sum(1)
                conc[v] = {"answers": int(len(vi)), "mean_ratings_at_zero": r4((P <= 0).sum(1).mean()),
                           "share_with_a_whole_band_at_zero": r4((pb <= 0).any(1).mean()),
                           "share_band_at_zero": {bd: r4((pb[:, k] <= 0).mean()) for k, bd in enumerate(BANDS)},
                           "mean_largest_share": r4(P.max(1).mean()), "mean_entropy": r4(ent.mean()),
                           "mean_probability_of_the_true_band": r4(pb[np.arange(len(vi)), BAND_OF[y[vi]]].mean()),
                           "observed_band_shares": {bd: r4((BAND_OF[y[vi]] == k).mean()) for k, bd in enumerate(BANDS)}}
            dg["concentration_first_draw"] = conc
            by_item = {iid: k_ for iid, k_ in key.items()}
            errs = {}
            for v in VARIANTS:
                fa = [x for x in r["failed_attempts"] if by_item[x["item"]]["variant"] == v]
                bad0 = [i for i in range(len(firms)) if (v, 0) in by[firms[i]] and by[firms[i]][(v, 0)] not in ans]
                other = Counter()
                for i in bad0:
                    for (vv, k_), iid in by[firms[i]].items():
                        if vv == v and k_ > 0:
                            other["valid" if iid in ans else "invalid"] += 1
                feats = Counter()
                for x in fa:
                    feats.update(k_ for k_, val in text_features(x["text"]).items() if val)
                lens = [len(x["text"]) for x in fa]
                errs[v] = {"failed_model_answers": len(fa), "firm_cards_failed_on_the_first_draw": len(bad0),
                           "classes": dict(Counter(error_class(x["error"]) for x in fa)),
                           "outcomes": dict(Counter(x["outcome"] for x in fa)),
                           "text_features": dict(feats),
                           "text_length_quantiles": ({q: int(np.quantile(lens, q)) for q in (0.0, 0.5, 1.0)} if lens else None),
                           "other_draws_of_the_failed_first_draws": dict(other),
                           "failed_first_draws_by_size": dict(Counter(size[firms[i]] for i in bad0)),
                           "failed_first_draws_by_sector": dict(Counter(info[firms[i]]["sector"] or "not given" for i in bad0)),
                           "failed_first_draws_by_true_band": dict(Counter(BANDS[BAND_OF[y[i]]] for i in bad0))}
            dg["technical_errors"] = errs
            res["diagnosis"] = dg
        if a.level_check:  # since 0.7: the plan's scenarios (project lead, 18.11.2026): each hypothesis' own centred
            def n_carried(mask):  # differences at the expected test firms (coverage by size class) and at 692
                cv = {k: sum(1 for i in range(len(firms)) if mask[i] and size[firms[i]] == k) / elig[k] for k in elig}
                return max(2, round(sum(a.test_firms * size_share[k] * cv.get(k, float(mask.mean())) for k in size_share)))
            m1 = np.zeros(len(firms), dtype=bool)
            m1[bi] = True
            lc = {}
            for hyp, dv, mask in (("H1", d, m1), ("H3", d3, ~np.isnan(A_full))):
                if len(dv) < MIN_FIRMS:
                    lc[hyp] = f"fewer than {MIN_FIRMS} firms: no check"
                    continue
                for lab, n_ in (("expected_complete_cases", n_carried(mask)), ("all_eligible", a.test_firms)):
                    r_ = calibration_check(dv, n_, a.level_reps, a.level_inner)
                    lc[f"{hyp}/{lab}"] = r_
                    level_checks.append((f"{name}/{hyp}/{lab}", r_))
            res["level_check"] = lc
        out["models"][name] = res
    out["level_check"] = (level_criterion(level_checks, names, k_by_model={n: k_of.get(n) for n in names}, split=design["split"],
                                          k_from_first_report=k_source is not None) if a.level_check else None)
    family = {n: pvals.get(n) for n in names}
    missing = FAMILY_SIZE - len(family)
    for k in range(missing):
        family[f"(missing model {k + 1})"] = None
    rej = holm(family)
    h3fam = {n: h3.get(n) for n in family}
    # since 0.8 (the project lead's decision of 19.11.2026, before the confirmatory analysis): the primary family is the
    # three H1 under Holm at alpha 0.05 with the directional rule; H3 is exploratory (estimates and intervals, its p shown
    # without any claim) and passes no level to the H1 - the graph of 0.5-0.7 is no longer used, because the calibration of
    # H3 was not confirmed for two models and in the graph a rejected H3 passes its level on to the other H1
    out["primary_family"] = {"comparisons": "H1 full vs segment per model (primary); H3 full vs logit_segment per model "
                                            "(exploratory, no claim, no level passed)",
                             "family_size": FAMILY_SIZE, "models_missing_from_this_report": missing,
                             "p_directional_H1": family,
                             "procedure": "Holm over the three H1 at alpha 0.05; p values directional (p if the mean difference "
                                          "> 0, else 1); since 0.8, the project lead's decision of 19.11.2026 (the graph with H3 "
                                          "is no longer used)",
                             "holm_rejected": rej,
                             "supports_claim": {n: rej[n] for n in family},
                             "h3_exploratory": {"p_directional_shown_without_claim": h3fam,
                                                "note": "estimates and intervals in each model's h3_full_vs_logit_segment; no "
                                                        "confirmatory success is declared and no level is passed to H1"},
                             "alpha": ALPHA, "scope": "a rejection supports the claim for that model only: the full profile "
                                                      "improves the forecast over the short card in that model; it is not a "
                                                      "proof of an advantage over the statistical model nor of a value for "
                                                      "product decisions"}
    out["fits"] = {"total": len(SC.DIAG), "failed": sum(not (x["success"] and x["finite"]) for x in SC.DIAG)}
    bad = nan_paths(out)
    if bad:
        sys.exit(f"non-finite values in the report: {bad[:10]}")
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1, allow_nan=False) + "\n")
    print(json.dumps({"statistical": out["statistical"]["band_logloss"],
                      "models": {n: {k: m["primary"].get(k) for k in ("firms", "mean_d", "ci95", "p")} for n, m in out["models"].items()},
                      "supports_claim": out["primary_family"]["supports_claim"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
