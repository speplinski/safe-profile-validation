#!/usr/bin/env python3
"""SAFE validation: a check of the serving stack before the confirmatory runs (code review round 12, D1).

The confirmatory analysis refuses a run whose settings, runner modules or serving stack (SDK versions,
anthropic-version, Ollama version and model show, the Models API's model id and creation date) differ from the frozen
ones - after the run, when it is too late. This script checks the same before the runs, on a preflight run of ONE item
per model, made on a COPY of the frozen test build in its own directory (never the real run directory: its session would
stay in run.json):

    cp -r data/local/safe_runs/test_build data/local/safe_runs/preflight_haiku
    python3 scripts/run_b_k2.py run --out data/local/safe_runs/preflight_haiku <the frozen settings> --limit 1
    python3 scripts/safe_preflight.py --frozen data/reference/safe_frozen.json --run haiku=data/local/safe_runs/preflight_haiku

A pass means the real runs, started now with the same settings, will meet the frozen stack; an automatic update of
Ollama or of the anthropic / openai packages between the preflight and the runs would still break them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import safe_analyze as A  # noqa: E402

VERSION = "0.1"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--run", action="append", default=[], help="NAME=DIR, a preflight run per model")
    a = ap.parse_args()
    frozen = json.loads(Path(a.frozen).read_text())
    tb = frozen.get("test_build") or {}
    if not tb:
        sys.exit("--frozen has no test_build yet: make the test build and add it with safe_make_frozen --test-build first")
    if not a.run:
        sys.exit("no --run given")
    for s in a.run:
        name, _, d = s.partition("=")
        meta = json.loads((Path(d) / "run.json").read_text())
        cfg = meta["config"]
        if cfg.get("items_sha256") != tb.get("items_sha256"):
            sys.exit(f"{name}: the preflight is not a copy of the frozen test build")
        served = None
        p_ans = Path(d) / "answers.jsonl"
        if p_ans.exists():
            for ln in p_ans.read_text(encoding="utf-8").splitlines():
                if ln.strip() and json.loads(ln).get("served_model"):
                    served = json.loads(ln)["served_model"]
                    break
        if cfg.get("api") in ("anthropic", "openai-responses") and served is None:
            sys.exit(f"{name}: no answer with a served model in the preflight (did the one call go through?)")
        r = {"config": cfg, "served_model": served,
             "session_materials": [x.get("materials_sha256") for x in meta["sessions"]],
             "session_servers": [A.server_identity(x.get("server") or {}) for x in meta["sessions"]]}
        A.check_frozen(name, r, frozen)  # the run settings, weights digest, served model, K
        A.check_run_materials(name, r, frozen)  # the runner modules and the serving stack of every session
        print(f"{name}: settings, runner modules and serving stack match the frozen ones")


if __name__ == "__main__":
    main()
