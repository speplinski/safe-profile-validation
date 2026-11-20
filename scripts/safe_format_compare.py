#!/usr/bin/env python3
"""SAFE validation: two ways of eliciting the distribution (the probabilities instruction of pilot 1 and the counts
instruction, each with its own rule on zeros and answer format) on the same firms and cards (no model call).

Input: for one model, a probabilities build and its run (--old-build, --old-run) and a counts build of the same records
and its run (--new-build, --new-run). Since 0.2 both builds must be made with safe_build.py --only-repeat-firms-of
<the first pilot's key.json> and the first pilot's --seed, one with --elicitation probabilities and one with counts,
from the same card key: the script refuses builds with another card key, the wrong elicitation or parser, or any
matched pair whose card (the user message) differs, and checks that each system prompt is the current one of its
format; since 0.3 each run is read as in safe_analyze (finished, complete, one served model) and the two runs must have
the same settings (all frozen ones but the parser), the same weights digest and the same served model, so the two arms differ in the instruction and the answer format only. (The first pilot's own build cannot be
used: its card key differs from 0.5 on.) Items are matched by firm (wave, permid), variant and draw 0. Per format and variant: the mean entropy of the ten-value distribution, the mean
largest share, the mean mass on 1-3, the mean band log score (floored as in safe_analyze), and the paired differences
counts - probabilities with a percentile bootstrap interval over firms. Descriptive, aggregates only.

    python3 scripts/safe_format_compare.py --old-build <dir> --old-run <dir> --new-build <dir> --new-run <dir> \
        --out data/processed/safe_format_compare_<model>_v0.1.json
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import safe_analyze as A  # noqa: E402
import safe_prompt as P  # noqa: E402

VERSION = "0.3"


FORMATS = {"probabilities": "safe_probabilities", "counts": "safe_counts"}


def load(build: Path, run: Path, fmt: str) -> tuple[dict, dict, dict]:
    design = json.loads((build / "design.json").read_text())
    raw = gzip.open(build / "items.jsonl.gz").read()
    if hashlib.sha256(raw).hexdigest() != design["items_sha256"]:
        sys.exit(f"{build}: items do not match design.json")
    if hashlib.sha256((build / "key.json").read_bytes()).hexdigest() != design.get("key_json_sha256"):
        sys.exit(f"{build}: key.json does not match design.json")
    if design.get("elicitation") != fmt or design.get("parser") != FORMATS[fmt]:
        sys.exit(f"{build}: not a {fmt} build (elicitation {design.get('elicitation')}, parser {design.get('parser')})")
    if not design.get("only_repeat_firms_of_an_earlier_build"):
        sys.exit(f"{build}: not built with --only-repeat-firms-of")
    if design.get("card_key_sha256") != hashlib.sha256(P.KEY_PATH.read_bytes()).hexdigest():
        sys.exit(f"{build}: built from another card key than the current one")
    key = json.loads((build / "key.json").read_text())
    cfg = json.loads((run / "run.json").read_text())["config"]
    if cfg.get("parser") != FORMATS[fmt]:
        sys.exit(f"{run}: parser {cfg.get('parser')}, a {fmt} run needs {FORMATS[fmt]}")
    # since 0.3 the run is read as in the main analysis (project lead's review, 18.11.2026): this build's run, the
    # plan's max_attempts, the last session finished, every item valid or with max_attempts model answers, one served
    # model over every model answer
    r = A.load_run(fmt, run, design, set(key))
    cfg = r["config"]
    sysp = P.system_prompt(P.load_key(), fmt)
    msgs = {}
    for ln in raw.decode("utf-8").splitlines():
        it = json.loads(ln)
        m = {x["role"]: x["content"] for x in it["messages"]}
        if m.get("system") != sysp:
            sys.exit(f"{build}: an item's system prompt is not the current {fmt} prompt")
        msgs[it["item"]] = m["user"]
    ans = r["answers"]
    by, cards = {}, {}
    for iid, k in key.items():
        if k["repeat"] == 0:
            cards[(k["wave"], k["permid"], k["variant"])] = msgs[iid]
            if iid in ans:
                by[(k["wave"], k["permid"], k["variant"])] = (ans[iid], int(k["answer"]) - 1)
    return by, cards, {"model": cfg.get("model"), "model_digest": cfg.get("model_digest"), "served_model": r["served_model"],
                       "parser": cfg.get("parser"), "items_sha256": design["items_sha256"],
                       "settings": {k: cfg.get(k) for k in A.FROZEN_CONFIG if k != "parser"}}


def stats(p: np.ndarray, y: np.ndarray) -> dict:
    q = np.clip(p, 1e-12, 1)
    ent = -(np.where(p > 0, p * np.log(q), 0)).sum(1)
    return {"entropy": ent, "largest_share": p.max(1), "mass_1_3": p[:, :3].sum(1), "band_log_score": A.scores(p, y)["band"]}


def answers_of(build: Path) -> dict:
    """(wave, permid) -> the record's answer in the build's key (every item of a firm must agree)."""
    out = {}
    for k in json.loads((build / "key.json").read_text()).values():
        f = (k["wave"], k["permid"])
        if out.setdefault(f, k["answer"]) != k["answer"]:
            sys.exit(f"{build}: one firm with two answers in key.json")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("old-build", "old-run", "new-build", "new-run", "out"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    old, co, mo = load(Path(a.old_build), Path(a.old_run), "probabilities")
    new, cn, mn = load(Path(a.new_build), Path(a.new_run), "counts")
    diff = {k: (mo["settings"][k], mn["settings"][k]) for k in mo["settings"] if mo["settings"][k] != mn["settings"][k]}
    if diff:  # since 0.3: the same generation settings (all of the frozen ones but the parser)
        sys.exit(f"the two runs differ in their settings {diff}: not a comparison of elicitations")
    if mo["model_digest"] != mn["model_digest"]:
        sys.exit("the two runs are of different weights (model_digest)")
    if mo["served_model"] != mn["served_model"]:
        sys.exit(f"the API served different models: {mo['served_model']} vs {mn['served_model']}")
    if set(co) != set(cn):
        sys.exit("the two builds do not hold the same firm-cards")
    yo, yn = answers_of(Path(a.old_build)), answers_of(Path(a.new_build))
    if yo != yn:  # since 0.2: the same cards must carry the same survey answer (user review, 18.11.2026)
        sys.exit(f"{sum(yo.get(k) != yn.get(k) for k in set(yo) | set(yn))} firms have different answers to the question "
                 "in the two builds: not the same records")
    diff = [k for k in co if co[k] != cn[k]]
    if diff:
        by_v = {v: sum(k[2] == v for k in diff) for v in ("full", "segment", "shuffled")}
        sys.exit(f"{len(diff)} cards differ between the builds {by_v}: not a comparison of formats")
    out = {"script": f"safe_format_compare {VERSION}", "model": mo["model"], "probabilities": mo, "counts": mn,
           "cards_checked_identical": len(co), "variants": {}}
    rng = np.random.default_rng(A.SEED)
    for v in ("full", "segment", "shuffled"):
        keys = sorted(k for k in old if k[2] == v and k in new)
        if not keys:
            continue
        y = np.array([old[k][1] for k in keys])
        so = stats(np.array([old[k][0] for k in keys]), y)
        sn = stats(np.array([new[k][0] for k in keys]), y)
        res = {"firms": len(keys), "firms_without_valid_answers_in_both": sum(k[2] == v for k in co) - len(keys)}
        for m in so:
            dd = sn[m] - so[m]
            bs = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(A.BOOT)]
            res[m] = {"probabilities": A.r4(so[m].mean()), "counts": A.r4(sn[m].mean()), "counts_minus_probabilities": A.r4(dd.mean()),
                      "ci95": [A.r4(np.quantile(bs, 0.025)), A.r4(np.quantile(bs, 0.975))]}
        out["variants"][v] = res
    out["note"] = ("descriptive: the same records and cards (checked), the current system prompt of each elicitation; the "
                   "arms differ in the whole instruction (including the rule on zeros) and the answer format, so this compares "
                   "two ways of eliciting the distribution, not the number format alone; firms with a technical error in "
                   "either arm are left out")
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(out["variants"], ensure_ascii=False))


if __name__ == "__main__":
    main()
