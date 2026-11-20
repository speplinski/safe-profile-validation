#!/usr/bin/env python3
"""SAFE validation: the comparison of three ways of eliciting the distribution (plan "elicitation_comparison"; no model call).

Input: the build of safe_build.py --arms A B C --variants full segment --only-repeat-firms-of <the second pilot's key>
(--build), the second pilot's build (--reference-build, for the check that every card is that pilot's card), one run of
the build per model (--run NAME=DIR, the three models of the family) and the second pilot's run of the same model
(--reference-run NAME=DIR: the settings, the weights digest and the served model must be the same, the runner version
and the parser aside; its arm-A answers are auxiliary: stability of A between the runs), and the SAFE extract (for the
marginal distribution of the training data, as in safe_analyze).

Selection (the plan, fixed before the calls): S(arm) = mean over the models (equal weights) and the two cards (equal
weights) of the mean over the firms of the band log score of the arm (floor 0.01) minus that of the marginal; an item
without a valid answer after max_attempts is scored as the marginal (sensitivities: worst, and complete cases common
to the three arms); an arm with more than 5% technical errors for any model and card is not eligible (none eligible:
"no eligible arm", no automatic choice); the paired differences of S for all three pairs by a bootstrap over firms
(2,000 resamples, every resample keeping all answers of a firm together) with family-wise 95% protection (Bonferroni);
an eligible arm drops out when another eligible arm has a demonstrated advantage over it; among the remaining the
smallest change is chosen, in the order A, B, C (a conservative choice under an unresolved difference, not a proof of
equivalence). Reported
separately and never ranking: technical errors by class, calibration by band, confident misses (true band < 0.05) and
their share of the sum of log scores, d = full - segment by arm (descriptive), the stability of A. Aggregates only.

    python3 scripts/safe_elicitation_compare.py --build <abc build> --reference-build <pilot-2 build> \\
        --run qwen=<dir> --run haiku=<dir> --run luna=<dir> \\
        --reference-run qwen=<dir> --reference-run haiku=<dir> --reference-run luna=<dir> \\
        --safe data/local/safe/safe_poland_all_waves.csv.gz --out data/processed/safe_elicitation_compare_v0.1.json
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import safe_analyze as A  # noqa: E402
import safe_prompt as P  # noqa: E402
import safe_signal_check as SC  # noqa: E402

VERSION = "0.4.1"
ARMS = ("A", "B", "C")
ARM_MODES = {"A": "counts", "B": "counts_nozero", "C": "probabilities"}
ARM_PARSERS = {"A": "safe_counts", "B": "safe_counts", "C": "safe_probabilities"}
CARDS = ("full", "segment")
ERROR_LIMIT = 0.05
BOOT = 2000
PAIR_ALPHA = 0.05  # family-wise over the three pairs
CONFIDENT_MISS = 0.05
NOT_COMPARED = ("script", "parser")  # the runner version (0.9.4 reads the parser per item) and the parser differ by design


def read_build(b: Path) -> tuple[dict, dict, dict]:
    design = json.loads((b / "design.json").read_text())
    raw = gzip.open(b / "items.jsonl.gz").read()
    if A.sha(raw) != design["items_sha256"]:
        sys.exit(f"{b}: items do not match design.json")
    if A.sha((b / "key.json").read_bytes()) != design.get("key_json_sha256"):
        sys.exit(f"{b}: key.json does not match design.json")
    items = {json.loads(ln)["item"]: json.loads(ln) for ln in raw.decode("utf-8").splitlines() if ln.strip()}
    return design, json.loads((b / "key.json").read_text()), items


def runs_arg(xs: list) -> dict:
    out = {}
    for s in xs:
        n, _, d = s.partition("=")
        if not n or not d or n in out:
            sys.exit(f"{s!r}: NAME=DIR, names unique")
        out[n] = Path(d)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("build", "reference-build", "safe", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--run", action="append", default=[])
    ap.add_argument("--reference-run", action="append", default=[])
    a = ap.parse_args()
    design, key, items = read_build(Path(a.build))
    if design.get("parser") != "safe_by_item" or sorted(design.get("arms") or {}) != list(ARMS) \
            or sorted(design.get("variants_built") or []) != sorted(CARDS) or design.get("split") != "dev":
        sys.exit("--build: not the comparison build (dev, --arms A B C --variants full segment)")
    rdesign, rkey, ritems = read_build(Path(a.reference_build))
    # since 0.4 (project lead, 19.11.2026): the same SAFE extract and the same true answers as the second pilot
    if rdesign.get("source_sha256") != design.get("source_sha256"):
        sys.exit("the build and the reference build are not from the same SAFE extract (source_sha256)")

    def answers(k_: dict, label: str) -> dict:
        out_ = {}
        for r_ in k_.values():
            f_ = (r_["wave"], r_["permid"])
            if out_.setdefault(f_, r_["answer"]) != r_["answer"]:
                sys.exit(f"{label}: one firm with two answers in key.json")
        return out_
    ya, yr = answers(key, "--build"), answers(rkey, "--reference-build")
    repeated = {(r_["wave"], r_["permid"]) for r_ in rkey.values() if r_["repeat"] > 0}  # what --only-repeat-firms-of keeps
    if set(ya) != repeated:
        sys.exit("the build's firms are not the reference build's repeated firms (safe_build --only-repeat-firms-of)")
    diff_y = sum(ya[f] != yr[f] for f in ya)
    if diff_y:
        sys.exit(f"{diff_y} firms have another true answer than in the reference build")
    ref_card = {(k["wave"], k["permid"], k["variant"]): ritems[i]["messages"][1]["content"]
                for i, k in rkey.items() if k["repeat"] == 0}
    pkey = P.load_key()
    ref_sys = {it["messages"][0]["content"] for it in ritems.values()}
    if len(ref_sys) != 1 or ref_sys != {P.system_prompt(pkey, "counts")}:  # arm A is the second pilot's instruction run again
        sys.exit("the reference build's system prompt is not arm A's (the current counts prompt)")
    for iid, k in key.items():  # the same cards as the second pilot, the current prompt of each arm, the arm's parser
        m = {x["role"]: x["content"] for x in items[iid]["messages"]}
        if m["user"] != ref_card.get((k["wave"], k["permid"], k["variant"])):
            sys.exit("a card differs from the reference build's card")
        if m["system"] != P.system_prompt(pkey, ARM_MODES[k["arm"]]) or items[iid].get("parser") != ARM_PARSERS[k["arm"]]:
            sys.exit(f"an item of arm {k['arm']} has another prompt or parser than its arm")
    cells = Counter((k["wave"], k["permid"], k["variant"], k["arm"]) for k in key.values())
    if any(c != 1 for c in cells.values()):
        sys.exit("key.json: a firm-card-arm appears more than once")
    firms = sorted({(k["wave"], k["permid"]) for k in key.values()})
    if len(cells) != len(firms) * len(CARDS) * len(ARMS):
        sys.exit("the build does not hold every firm x card x arm")
    runs, refs = runs_arg(a.run), runs_arg(a.reference_run)
    if len(runs) != A.FAMILY_SIZE or set(runs) != set(refs):
        sys.exit(f"--run and --reference-run: the {A.FAMILY_SIZE} models of the family, the same names")
    for label, dd in (("--run", runs), ("--reference-run", refs)):
        if len({d.resolve() for d in dd.values()}) != len(dd):
            sys.exit(f"{label}: two models point to the same directory")
    R, RR = {}, {}
    for n in runs:
        R[n] = A.load_run(n, runs[n], design, set(key))
        RR[n] = A.load_run(n, refs[n], rdesign, set(rkey))
        c, rc = R[n]["config"], RR[n]["config"]
        diff = {k: (rc.get(k), c.get(k)) for k in A.FROZEN_CONFIG if k not in NOT_COMPARED and c.get(k) != rc.get(k)}
        if diff:
            sys.exit(f"run {n}: settings differ from the second pilot's run {diff}")
        if c.get("model_digest") != rc.get("model_digest") or R[n]["served_model"] != RR[n]["served_model"]:
            sys.exit(f"run {n}: another weights digest or served model than the second pilot's run")
    ident = [(R[n]["model"], R[n]["config"].get("model_digest"), R[n]["served_model"]) for n in R]
    if len(set(ident)) != len(ident):
        sys.exit("two runs of the same model: the family needs three different models")
    # the marginal of the training data, as in safe_analyze (development records without the evaluated firms)
    raw = Path(a.safe).read_bytes()
    if A.sha(raw) != design["source_sha256"]:
        sys.exit("--safe is not the extract the build was made from")
    rows = [r for r in csv.DictReader(gzip.decompress(raw).decode("utf-8").splitlines()) if r["wave"] in SC.WAVES]
    test_firms = {r["permid"] for r in rows if r["wave"] == SC.TEST}
    evaluated = {p for _, p in firms}
    train = [r for r in rows if r["wave"] != SC.TEST and r["permid"] not in test_firms and r["permid"] not in evaluated
             and SC.norm(r["q0b_3"]) in A.SCALE]
    ytr = np.array([int(SC.norm(r["q0b_3"])) - 1 for r in train])
    info = {(k["wave"], k["permid"]): k for k in key.values()}
    y = np.array([int(info[f]["answer"]) - 1 for f in firms])
    rec = {(SC.WAVES[r["wave"]], r["permid"]): r for r in rows}
    marg_p = SC.predict(train, ytr, [rec[f] for f in firms], 10, "marginal")
    marg = A.scores(marg_p, y)["band"]
    worst = A.worst_band(y)
    cell_item = {(k["wave"], k["permid"], k["variant"], k["arm"]): i for i, k in key.items()}
    out = {"script": f"safe_elicitation_compare {VERSION}", "script_sha256": A.sha(Path(__file__).read_bytes()),
           "plan_sha256": A.sha((A.ROOT / "data/reference/safe_validation_plan.json").read_bytes()),
           "build_items_sha256": design["items_sha256"], "reference_build_items_sha256": rdesign["items_sha256"],
           "source_sha256": design["source_sha256"], "true_answers_checked_against_the_reference": True,
           "firms": len(firms), "training_records": len(train), "models": {}, "selection": {}}
    sc = {}  # (model, card, arm) -> per-firm band log score minus the marginal's, by rule
    err = {}
    for n, r in R.items():
        ans = r["answers"]
        m = {"model": r["model"], "served_model": r["served_model"], "run_identity": r["identity"], "cards": {}}
        for v in CARDS:
            m["cards"][v] = {}
            for arm in ARMS:
                ok = np.array([cell_item[(*f, v, arm)] in ans for f in firms])
                P_ = np.full((len(firms), 10), np.nan)
                for i, f in enumerate(firms):
                    if ok[i]:
                        P_[i] = ans[cell_item[(*f, v, arm)]]
                s_ok = np.full(len(firms), np.nan)
                s_ok[ok] = A.scores(P_[ok], y[ok])["band"]
                sc[(n, v, arm)] = {"marginal": np.where(ok, s_ok, marg) - marg, "worst": np.where(ok, s_ok, worst) - marg,
                                   "ok": ok, "raw": s_ok}
                err[(n, v, arm)] = float((~ok).mean())
                pb = A.band_probs(A.floor(P_[ok])) if ok.any() else np.zeros((0, 3))
                true_p = A.band_probs(P_[ok])[np.arange(int(ok.sum())), A.BAND_OF[y[ok]]] if ok.any() else np.zeros(0)
                miss = true_p < CONFIDENT_MISS
                cal = {bd: {"mean_predicted": A.r4(pb[:, k].mean()) if len(pb) else None,
                            "observed": A.r4((A.BAND_OF[y[ok]] == k).mean()) if ok.any() else None} for k, bd in enumerate(A.BANDS)}
                fa = [x for x in r["failed_attempts"] if key[x["item"]]["variant"] == v and key[x["item"]]["arm"] == arm]
                m["cards"][v][arm] = {
                    "valid": int(ok.sum()), "technical_error_share": A.r4(err[(n, v, arm)]),
                    "failed_model_answers_by_class": dict(Counter(A.error_class(x["error"]) for x in fa)),
                    "band_logscore_minus_marginal": A.r4(sc[(n, v, arm)]["marginal"].mean()),
                    "band_logscore_minus_marginal_errors_worst": A.r4(sc[(n, v, arm)]["worst"].mean()),
                    "calibration_by_band": cal,
                    "calibration_largest_gap": (A.r4(max(abs(c["mean_predicted"] - c["observed"]) for c in cal.values()))
                                                if ok.any() else None),
                    "confident_misses": {"share": A.r4(miss.mean()) if ok.any() else None,
                                         "share_of_the_sum_of_log_scores": (A.r4(s_ok[ok][miss].sum() / s_ok[ok].sum())
                                                                            if ok.any() and s_ok[ok].sum() != 0 else None)}}
        m["d_full_minus_segment_by_arm"] = {}  # descriptive, never a criterion
        for arm in ARMS:
            both = sc[(n, "full", arm)]["ok"] & sc[(n, "segment", arm)]["ok"]
            dd = sc[(n, "full", arm)]["raw"][both] - sc[(n, "segment", arm)]["raw"][both]
            m["d_full_minus_segment_by_arm"][arm] = {"firms": int(both.sum()), "mean": A.r4(dd.mean()) if both.any() else None}
        # auxiliary: arm A of this run vs the second pilot's first draw of the same cards
        rans, stab = RR[n]["answers"], {}
        for v in CARDS:
            ritem = {(k["wave"], k["permid"]): i for i, k in rkey.items() if k["variant"] == v and k["repeat"] == 0}
            both = [i for i, f in enumerate(firms) if sc[(n, v, "A")]["ok"][i] and ritem[f] in rans]
            if both:
                prev = A.scores(np.array([rans[ritem[firms[i]]] for i in both]), y[both])["band"]
                now = sc[(n, v, "A")]["raw"][both]
                stab[v] = {"firms": len(both), "mean_now_minus_pilot2": A.r4((now - prev).mean()),
                           "mean_abs_difference": A.r4(np.abs(now - prev).mean())}
        m["stability_of_A_vs_the_second_pilot"] = stab
        out["models"][n] = m
    names = sorted(R)

    def S(rule: str, idx: np.ndarray, arms=ARMS) -> dict:
        return {arm: float(np.mean([sc[(n, v, arm)][rule][idx].mean() for n in names for v in CARDS])) for arm in arms}

    def S_common() -> dict:
        """Complete cases common to the three arms, per model and card; a model-card without common firms (e.g. one arm
        with only errors) is left out and named, and the mean is over the other cells (since 0.3, code review 8)."""
        res, empty = {}, []
        for n in names:
            for v in CARDS:
                if not np.all([sc[(n, v, b)]["ok"] for b in ARMS], axis=0).any():
                    empty.append(f"{n}/{v}")
        for arm in ARMS:
            vals = [sc[(n, v, arm)]["marginal"][np.all([sc[(n, v, b)]["ok"] for b in ARMS], axis=0)].mean()
                    for n in names for v in CARDS if f"{n}/{v}" not in empty]
            res[arm] = A.r4(np.mean(vals)) if vals else None
        return {"S": res, "cells_without_common_firms": empty,
                "cells_used": len(names) * len(CARDS) - len(empty)}
    allidx = np.arange(len(firms))
    main = S("marginal", allidx)
    eligible = [arm for arm in ARMS if all(err[(n, v, arm)] <= ERROR_LIMIT for n in names for v in CARDS)]
    pairs = [("A", "B"), ("A", "C"), ("B", "C")]
    rng = np.random.default_rng(A.SEED)
    diffs = {pr: [] for pr in pairs}
    for _ in range(BOOT):  # each resample keeps together all answers of a firm (every model, card and arm)
        sb = S("marginal", rng.integers(0, len(firms), len(firms)))
        for x, z in pairs:
            diffs[(x, z)].append(sb[x] - sb[z])
    q = PAIR_ALPHA / len(pairs) / 2  # Bonferroni over the three pairs, two-sided
    comp = {}
    for x, z in pairs:
        lo, hi = float(np.quantile(diffs[(x, z)], q)), float(np.quantile(diffs[(x, z)], 1 - q))
        comp[f"{x}-{z}"] = {"difference": A.r4(main[x] - main[z]), "ci_familywise_95": [A.r4(lo), A.r4(hi)],
                            "demonstrated_advantage": x if lo > 0 else z if hi < 0 else None}
    beaten = {}
    for x, z in pairs:  # an eligible arm drops out when another eligible arm has a demonstrated advantage over it
        w = comp[f"{x}-{z}"]["demonstrated_advantage"]
        loser = z if w == x else x if w == z else None
        if w in eligible and loser in eligible:
            beaten.setdefault(loser, []).append(w)
    remaining = [arm for arm in eligible if arm not in beaten]
    chosen = next((arm for arm in ARMS if arm in remaining), None)
    out["selection"] = {
        "S_errors_as_marginal": {arm: A.r4(v) for arm, v in main.items()},
        "S_errors_worst": {arm: A.r4(v) for arm, v in S("worst", allidx).items()},
        "S_complete_cases_common_to_the_three_arms": S_common(),
        "technical_error_shares": {f"{n}/{v}/{arm}": A.r4(err[(n, v, arm)]) for n in names for v in CARDS for arm in ARMS},
        "eligible": eligible, "pairwise": comp, "excluded_by_a_demonstrated_advantage": beaten,
        "remaining": remaining, "chosen": chosen,
        "result": (f"arm {chosen}" if chosen else "no eligible arm"),
        "rule": "S = mean over the models and the two cards (equal weights) of the mean band log score minus the marginal's, "
                "invalid answers as the marginal; eligible: at most 5% technical errors for every model and card (none "
                "eligible: no eligible arm, no automatic choice); all three pairs by a paired bootstrap over firms with "
                "family-wise 95% protection (Bonferroni); an eligible arm drops out when another eligible arm has a "
                "demonstrated advantage over it; among the remaining the smallest change A, B, C: a conservative choice "
                "under an unresolved difference, not a proof of equivalence; zeros and d are not criteria"}
    bad = A.nan_paths(out)
    if bad:
        sys.exit(f"non-finite values in the report: {bad[:10]}")
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1, allow_nan=False) + "\n")
    print(json.dumps(out["selection"], ensure_ascii=False))


if __name__ == "__main__":
    main()
