#!/usr/bin/env python3
"""SAFE validation: write the frozen settings of the confirmatory analysis (data/reference/safe_frozen.json) from the
third development pilot's reports (no model call). The file is a candidate until the project lead approves it.

Inputs: the pilot's first report (K basis; run identities), its calibration report (K taken with --k-from; the same run
identities) and the H1-family qualification (safe_h1_qualification.py; qualified). Per model: every setting of
safe_analyze.FROZEN_CONFIG as the pilot ran it, the weights digest (local model) or the served model (API; the
identifier the API reported, an alias where it reported one: it is not changed to a dated name), and K. The test build:
the test split, arm B (elicitation counts_nozero), the seed given here, max(K) draws of full and segment for every firm,
the card key's and the prompt module's sha256 as in the pilot's build (checked against the files now). The primary family
(Holm over the three H1, alpha 0.05, directional), H3 exploratory. The sha256 of the analysis, build, prompt and runner
scripts and of the card key: safe_analyze checks them on the test split, so a change after freezing stops the analysis.

    python3 scripts/safe_make_frozen.py --first data/processed/safe_dev_pilot3_v0.1.json \\
        --calibration data/processed/safe_dev_pilot3_calibration_v0.1.json \\
        --qualification data/processed/safe_h1_family_qualification_v0.1.json --test-seed 20261121 \\
        --out data/reference/safe_frozen.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import safe_analyze as A  # noqa: E402
import safe_prompt as P  # noqa: E402

VERSION = "0.4.1"
MATERIALS = ("scripts/safe_analyze.py", "scripts/safe_build.py", "scripts/safe_prompt.py", "scripts/run_b_k2.py",
             "scripts/llm_serving_check.py", "scripts/ollama_provenance.py", "scripts/safe_signal_check.py",
             "data/reference/safe_card_key.json", "data/reference/safe_validation_plan.json")  # since 0.3 (code review 10)


def pilot_server(d: Path, ri: dict) -> dict:
    """The serving stack of a pilot run (every session the same), after checking it is the run of the report."""
    meta = json.loads((d / "run.json").read_text())
    if meta["config"].get("items_sha256") != ri["items_sha256"] or A.sha((d / "answers.jsonl").read_bytes()) != ri["answers_sha256"]:
        sys.exit(f"--pilot-run {d}: not the run of the report (items or answers differ)")
    ids = [A.server_identity(x.get("server") or {}) for x in meta["sessions"]]
    if any(x != ids[0] for x in ids):
        sys.exit(f"--pilot-run {d}: the sessions ran on different serving stacks")
    return ids[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("first", "calibration", "qualification", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--test-seed", type=int, required=True)
    ap.add_argument("--pilot-run", action="append", default=[], help="since 0.4, NAME=DIR: the third pilot's run of each model "
                    "(its answers.jsonl must be the one in the report); the serving stack of its sessions (SDK versions, API "
                    "version, Ollama version and model show, the Models API's id and creation date) is frozen and the "
                    "confirmatory runs must match it")
    ap.add_argument("--approved-on", default=None, help="the date of the project lead's approval (YYYY-MM-DD): with a complete "
                    "candidate the status becomes 'frozen'; the confirmatory analysis requires it")
    ap.add_argument("--test-firms", type=int, default=692, help="the full test sample (2026Q3 records answering q0b_3 with 1-10)")
    ap.add_argument("--test-build", default=None, help="since 0.2: the test build made by the frozen command (no model "
                    "call yet); its source, items and key sha256 and its size are written into the frozen file, and the "
                    "analysis requires them; without it the file is not complete for the test")
    a = ap.parse_args()
    raws = {k: Path(getattr(a, k)).read_bytes() for k in ("first", "calibration", "qualification")}
    first, cal, qual = (json.loads(raws[k]) for k in ("first", "calibration", "qualification"))
    if first.get("level_check") is not None or first.get("draws_by_model"):
        sys.exit("--first: not a first report (made without --level-check and --k)")
    if cal.get("k_source", {}).get("first_report_sha256") != A.sha(raws["first"]):
        sys.exit("--calibration: its K does not come from this first report")
    if qual.get("calibration_report_sha256") != A.sha(raws["calibration"]) or qual.get("h1_family_qualified") is not True:
        sys.exit("--qualification: not the qualification of this calibration, or the H1 family is not qualified")
    if first["build"]["items_sha256"] != cal["build"]["items_sha256"] or first["build"].get("elicitation") != "counts_nozero":
        sys.exit("the reports are not of the same arm-B build")
    names = sorted(first["models"])
    if names != sorted(cal["models"]) or len(names) != A.FAMILY_SIZE:
        sys.exit("the two reports do not have the same three models")
    k_by = cal["k_source"]["recommended_K"]
    pr = {}
    for x in a.pilot_run:  # code review 12, D3: a name outside the family is an error, not silently skipped
        n_, _, d_ = x.partition("=")
        if n_ not in names or not d_ or n_ in pr:
            sys.exit(f"--pilot-run {x!r}: NAME=DIR with NAME one of {names}, each once")
        pr[n_] = d_
    if a.approved_on:  # code review 12, D2: a date, YYYY-MM-DD
        import datetime as _dt
        try:
            if _dt.date.fromisoformat(a.approved_on).isoformat() != a.approved_on:
                raise ValueError
        except ValueError:
            sys.exit(f"--approved-on {a.approved_on!r}: a date YYYY-MM-DD")
    import re
    runner = re.search(r'^VERSION = "([^"]+)"', (A.ROOT / "scripts/run_b_k2.py").read_text(), re.M).group(1)
    pilot_runner = {first["models"][n]["run_identity"]["config"]["script"] for n in names}
    models = {}
    for n in names:
        ri = first["models"][n]["run_identity"]
        if ri != cal["models"][n]["run_identity"]:
            sys.exit(f"{n}: the two reports are of different runs")
        if not (ri.get("model_digest") or ri.get("served_model")):
            sys.exit(f"{n}: neither a weights digest nor a served model in the run identity")
        cfg = {k: ri["config"].get(k) for k in A.FROZEN_CONFIG}
        cfg["script"] = runner  # the confirmatory runs' runner (records its modules' sha256 per session); see runner_note
        server = pilot_server(Path(pr[n]), ri) if n in pr else None
        models[n] = {"config": cfg, "model_digest": ri.get("model_digest"), "server": server,
                     "served_model": ri.get("served_model"), "K": int(k_by[n])}
    b = first["build"]
    now_key, now_prompt = A.sha(P.KEY_PATH.read_bytes()), A.sha(Path(P.__file__).read_bytes())
    if (b.get("card_key_sha256"), b.get("prompt_sha256")) != (now_key, now_prompt):
        sys.exit("the card key or the prompt module changed since the pilot's build")
    test_build = None
    if a.test_build:  # since 0.2 (the project lead, 19.11.2026): bind the analysis to this very test set
        tb = Path(a.test_build)
        d = json.loads((tb / "design.json").read_text())
        import gzip
        if A.sha(gzip.open(tb / "items.jsonl.gz").read()) != d["items_sha256"] or A.sha((tb / "key.json").read_bytes()) != d["key_json_sha256"]:
            sys.exit("--test-build: items or key do not match its design.json")
        want = {"split": "test", "elicitation": "counts_nozero", "parser": "safe_counts", "seed": a.test_seed,
                "repeats": max(m["K"] for m in models.values()) - 1, "sample_per_size": None, "sample_by_size": None,
                "only_repeat_firms_of_an_earlier_build": None, "card_key_sha256": now_key, "prompt_sha256": now_prompt,
                "source_sha256": first["source_sha256"], "records": a.test_firms}
        diff = {k: (d.get(k), v) for k, v in want.items() if d.get(k) != v}
        if diff:
            sys.exit(f"--test-build is not the frozen command's full test set: {diff}")
        if d.get("repeat_firms") != a.test_firms:
            sys.exit(f"--test-build: repeats for {d.get('repeat_firms')} firms, the frozen K needs all {a.test_firms}")
        # since 0.3 (code review 10): the full set from the build itself, not only from the parameter
        if d.get("excluded_firms_from_earlier_builds") not in (0, None) or (d.get("target_answers_in_split") or {}).get("1-10") != d["records"]:
            sys.exit("--test-build: not every test record answering 1-10 is in the build (exclusions or a sample)")
        test_build = {k: d[k] for k in ("items_sha256", "key_json_sha256", "source_sha256", "records", "items", "repeat_firms")}
    complete = test_build is not None and all(m["server"] for m in models.values())
    if a.approved_on and not complete:
        sys.exit("--approved-on: the candidate is not complete (test build and pilot runs needed)")
    out = {"status": (f"frozen (approved by the project lead on {a.approved_on}); not changed afterwards" if a.approved_on else
                      "candidate (19.11.2026): frozen when the project lead approves it; then not changed" if complete else
                      "incomplete candidate: " + "; ".join(x for x in (
                          None if test_build else "no test build yet (--test-build after the test build)",
                          None if all(m["server"] for m in models.values()) else "no serving stack of the pilot runs (--pilot-run)")
                          if x)),
           "script": f"safe_make_frozen {VERSION}", "script_sha256": A.sha(Path(__file__).read_bytes()),
           "sources": {k: A.sha(raws[k]) for k in raws},
           "pilot_build_items_sha256": b["items_sha256"],
           "models": models,
           "runner_note": (f"the confirmatory runs use run_b_k2 {runner}; the third pilot ran {sorted(pilot_runner)}; 0.9.5 differs "
                           "from 0.9.4 only in recording the sha256 of its modules per session (no change of the requests or "
                           "the parsing)"),
           "served_model_note": "the identifier the API reported in the pilot (an alias where it reported one); not changed "
                                "to a dated name, so a silent change of the model behind an alias is not detected",
           "build": {"split": "test", "elicitation": "counts_nozero", "parser": "safe_counts", "seed": a.test_seed,
                     "draws_of_full_and_segment": max(m["K"] for m in models.values()),
                     "command": f"safe_build.py --split test --elicitation counts_nozero --seed {a.test_seed} --repeats "
                                f"{max(m['K'] for m in models.values()) - 1} --repeat-firms 100000",
                     "card_key_sha256": now_key, "prompt_sha256": now_prompt, "source_sha256": first["source_sha256"],
                     "records": a.test_firms},
           "test_build": test_build,
           "family": {"primary": "Holm over the three H1 at alpha 0.05, directional (p if the mean d > 0, else 1)",
                      "h3": "exploratory: estimates and intervals, no confirmatory claim, no level passed to H1"},
           "materials_sha256": {f: A.sha((A.ROOT / f).read_bytes()) for f in MATERIALS}}
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({"models": {n: {"K": m["K"], "id": m["model_digest"] or m["served_model"]} for n, m in models.items()},
                      "build": out["build"]["command"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
