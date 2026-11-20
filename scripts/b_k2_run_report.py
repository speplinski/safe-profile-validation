#!/usr/bin/env python3
"""What a B-K2 run cost and gave, from its own files (no model call): outcomes, cost, throughput and the diagnostic
repeats of whole 2x2 profiles.

    python3 scripts/b_k2_run_report.py --run data/processed/b_runs/<label> [--pricing <price file>]

- Outcomes per row kept apart (valid, refusal with its category, cut, invalid_format, transport), for first attempts
  and for retries separately, by kind (pair, pair_repeat), function and version; 'brak podstaw' (a null rating
  inside a valid answer) counted apart; transport retries and rate-limit waits from run.json.
- Cost (Messages API rows with usage): input, cache write, cache read and output tokens summed and priced with the
  recorded price file, each request on its own (above 100,000 input tokens, cache included, the higher row);
  the cache-write price is the 5-minute one (the runner sets no cache_control, so it should stay 0). Cost per valid
  item. Since 0.3 also Responses API rows (run.json api openai-responses): their input_tokens include the cache reads
  and writes, so the input counted here is input minus both, cache written split by retention (cache_write 30 minutes,
  cache_write_12h), priced with llm_serving_check.openai_cost; the default price file follows the run's API.
- Throughput: unique items with a valid answer per hour of the sessions' wall time (started_at to finished_at, which
  includes errors, retries and waits; a session without finished_at, e.g. stopped by a cap or Ctrl-C, up to the time of
  its last row; not given when a session's length is unknown).
- 'Brak podstaw' counted in every valid answer, by group (first or retry, pair or pair_repeat).
- Cost: each row's stored cost (priced when it was asked); the report prices only rows without one, counted apart.
- Repeats (kind pair_repeat, repeat_of): the FIRST attempt of each repeat against the FIRST attempt of its item; per
  item the rating of the item's function (equal, difference); per profile the interaction HH-HL-LH+LL from first
  answers and from repeats. A pair with a missing first answer is reported as missing, never filled.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import b_k2_plan as bp  # noqa: E402
import run_b_k2 as R  # noqa: E402

VERSION = "0.3.2"


def cost(rows: list, price: dict | None, price_name: str, api: str = "anthropic") -> dict:
    """Tokens and USD by kind (input, cache write, cache read, output). Each row's own stored cost (priced when it was
    asked, with its price file named) is summed as it is; a row without one is priced with the given file and counted
    apart, so a later price file never reprices the history."""
    from llm_serving_check import anthropic_cost, openai_cost
    oai = api == "openai-responses"
    price_fn = openai_cost if oai else anthropic_cost
    tot = {"input": 0, "cache_write": 0, "cache_read": 0, "output": 0} | ({"cache_write_12h": 0} if oai else {})
    usd = {k: 0.0 for k in tot}
    n, total, unpriced, missing, files = 0, 0.0, 0, 0, {}
    for r in rows:
        if not r.get("ok") or r.get("prompt_tokens") is None:
            continue
        n += 1
        if oai:  # input_tokens include the cache reads and writes
            read, write, w12 = r.get("cached_tokens") or 0, r.get("cache_write_tokens") or 0, r.get("cache_write_12h_tokens") or 0
            tot["input"] += max(0, (r.get("prompt_tokens") or 0) - read - write)
            tot["cache_read"] += read
            tot["cache_write"] += write - w12
            tot["cache_write_12h"] += w12
            tot["output"] += r.get("completion_tokens") or 0
        else:
            for k, f in (("input", "prompt_tokens"), ("cache_write", "cache_creation_input_tokens"),
                         ("cache_read", "cache_read_input_tokens"), ("output", "completion_tokens")):
                tot[k] += r.get(f) or 0
        c = r.get("cost")
        if not c:
            unpriced += 1
            if price is None:  # no stored cost and the model missing in the price file: counted, never guessed
                missing += 1
                continue
            c = {**price_fn(r, price, by_kind=True), "pricing": price_name + " (applied by the report)"}
        for k, v in (c.get("by_kind") or {}).items():
            usd[k] = usd.get(k, 0.0) + v
        total += c["usd"]
        files[c.get("pricing")] = files.get(c.get("pricing"), 0) + 1
    return {"calls_with_usage": n, "tokens": tot, "usd": {k: round(v, 6) for k, v in usd.items()}, "usd_total": round(total, 6),
            "rows_without_stored_cost": unpriced, "rows_priced_by_the_report": unpriced - missing,
            "rows_not_priced_model_missing_in_the_price_file": missing, "complete": missing == 0, "price_files": files}


def first_answers(rows: list) -> dict:
    """item -> its first model attempt's row (attempt 1)."""
    out = {}
    for r in rows:
        if r.get("attempt") == 1 and r["item"] not in out:
            out[r["item"]] = r
    return out


def repeats(items: dict, first: dict) -> dict:
    def rating(k, fn):
        r = first.get(k)
        return (r["answer"]["ratings"].get(fn) if r and r.get("valid") else bp.INVALID)
    per_item, profiles = [], {}
    for k, it in items.items():
        if it["kind"] != "pair_repeat":
            continue
        fn, o = it["function"], it["repeat_of"]
        a, b = rating(o, fn), rating(k, fn)
        ok = isinstance(a, int) and isinstance(b, int)
        kind = ("missing" if bp.INVALID in (a, b) else "both_no_basis" if a is None and b is None else
                "no_basis_flip" if (a is None) != (b is None) else "equal" if a == b else "different")
        per_item.append({"item": o, "first": a, "repeat": b, "equal": (a == b) if ok else None, "diff": (b - a) if ok else None,
                         "kind": kind})
        profiles.setdefault((fn, it["firm_id"], it["size"]), {})[it["version"]] = (a, b)
    prof = []
    for (fn, f, s), v in sorted(profiles.items()):
        first_r = {ver: x[0] for ver, x in v.items()}
        rep_r = {ver: x[1] for ver, x in v.items()}
        full = lambda d: all(isinstance(d.get(x), int) for x in bp.VERSIONS)  # noqa: E731
        i1 = bp.interaction(first_r) if full(first_r) else None
        i2 = bp.interaction(rep_r) if full(rep_r) else None
        prof.append({"function": fn, "firm_id": f, "size": s, "interaction_first": i1, "interaction_repeat": i2,
                     "diff": (i2 - i1) if i1 is not None and i2 is not None else None})
    both = [x for x in per_item if x["equal"] is not None]
    pd = [p for p in prof if p["diff"] is not None]
    cnt = {}
    for x in per_item:
        cnt[x["kind"]] = cnt.get(x["kind"], 0) + 1
    return {"items": len(per_item), "by_kind": cnt, "items_both_valid": len(both),
            "rating_equal_share": (sum(x["equal"] for x in both) / len(both)) if both else None,
            "rating_mean_abs_diff": (sum(abs(x["diff"]) for x in both) / len(both)) if both else None,
            "profiles": len(prof), "profiles_both_complete": len(pd),
            "interaction_equal_share": (sum(p["diff"] == 0 for p in pd) / len(pd)) if pd else None,
            "interaction_mean_abs_diff": (sum(abs(p["diff"]) for p in pd) / len(pd)) if pd else None,
            "by_item": per_item, "by_profile": prof,
            "note": "diagnostic only: the repeats never replace a first answer and add no independent firm"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--pricing", default=None, help="default: the price file of the run's API (run_b_k2.PRICING)")
    ap.add_argument("--out", help="default: <run>/run_report.json")
    a = ap.parse_args()
    run = Path(a.run)
    items = {it["item"]: it for it in R.read_jsonl(run / "items.jsonl.gz")}
    rows = R.read_jsonl(run / "answers.jsonl")
    meta = json.loads((run / "run.json").read_text())
    first = first_answers(rows)
    out = {"script": f"b_k2_run_report {VERSION}", "run": run.name, "config": meta["config"]}
    oc = {}
    for r in rows:
        it = items.get(r["item"], {})
        key = ("first" if r.get("attempt") == 1 else "retry" if r.get("ok") else "transport_row", it.get("kind"))
        d = oc.setdefault(f"{key[0]} {key[1]}", {})
        o = r.get("outcome") or ("valid" if r.get("valid") else "?")
        d[o] = d.get(o, 0) + 1
    out["outcomes"] = oc
    ref = {}
    for r in rows:
        if r.get("outcome") == "refusal":
            ref[r.get("refusal_category") or "none"] = ref.get(r.get("refusal_category") or "none", 0) + 1
    out["refusal_categories"] = ref
    tiers = {}
    for r in rows:  # Responses API: the tier that served each call (the price file is for 'default', Standard)
        if r.get("ok") and "service_tier" in r:
            tiers[str(r.get("service_tier"))] = tiers.get(str(r.get("service_tier")), 0) + 1
    if tiers:
        out["service_tiers"] = tiers
    by = {}
    for k, r in first.items():
        it = items[k]
        if it["kind"] != "pair":
            continue
        d = by.setdefault(it["function"], {}).setdefault(it["version"], {"n": 0, "valid": 0, "no_basis": 0})
        d["n"] += 1
        if r.get("valid"):
            d["valid"] += 1
            d["no_basis"] += r["answer"]["ratings"].get(it["function"]) is None
    out["first_attempts_by_function_version"] = by
    nb = {}
    for r in rows:  # 'brak podstaw' in every valid answer, by group: never only in the main first attempts
        if not r.get("valid"):
            continue
        it = items.get(r["item"], {})
        g = f"{'first' if r.get('attempt') == 1 else 'retry'} {it.get('kind')}"
        d = nb.setdefault(g, {"valid_answers": 0, "no_basis_item_function": 0, "no_basis_any_function": 0})
        d["valid_answers"] += 1
        rt = r["answer"].get("ratings")
        if rt is None:  # a SAFE build (explicit probabilities): no "no basis" to count
            continue
        d["no_basis_item_function"] += rt.get(it.get("function")) is None if it.get("function") else 0
        d["no_basis_any_function"] += any(v is None for v in rt.values())
    out["no_basis_by_group"] = nb
    out["sessions"] = [s.get("summary") for s in meta["sessions"]]
    if items and all(it.get("kind") == "safe" for it in items.values()):  # a SAFE build: the B-K2 fields do not apply
        out["note_safe_build"] = ("a SAFE validation build: the fields about main pair items, 'no basis' and functions are "
                                  "B-K2 measures and do not apply; use the outcomes, times and cost")
        out.pop("no_basis_by_group", None)
    api = meta["config"].get("api") or "anthropic"
    price_file = Path(a.pricing) if a.pricing else Path(R.PRICING.get(api, R.PRICING["anthropic"]))
    pricing = json.loads(price_file.read_text()) if price_file.exists() else {}
    model = meta["config"]["model"]
    if any(r.get("cost") for r in rows) or model in pricing.get("models", {}):
        # the stored costs always count; the price file is only for rows without a stored cost
        c = cost(rows, pricing.get("models", {}).get(model), price_file.name, api)
        valid_items = {r["item"] for r in rows if r.get("valid")}
        main_valid = {k for k in valid_items if items.get(k, {}).get("kind") == "pair"}
        c.update(pricing_file=price_file.name, pricing_read_on=pricing.get("read_on"),
                 usd_per_valid_item=round(c["usd_total"] / len(valid_items), 6) if valid_items else None,
                 usd_per_valid_main_item_all_calls=round(c["usd_total"] / len(main_valid), 6) if main_valid else None,
                 note="usd_total includes every call (retries, repeats, invalid answers)")
        out["cost"] = c
    secs, open_sessions, unknown = 0.0, 0, 0
    starts = [dt.datetime.fromisoformat(s["started_at"]) for s in meta["sessions"] if s.get("started_at")]
    stamps = sorted(dt.datetime.fromisoformat(r["at"]) for r in rows if r.get("at"))
    for i, s in enumerate(meta["sessions"]):
        if not s.get("started_at"):
            continue
        t0 = dt.datetime.fromisoformat(s["started_at"])
        end = s.get("finished_at") or s.get("stopped_at")  # a stopped session ends at its stop
        if end:
            secs += (dt.datetime.fromisoformat(end) - t0).total_seconds()
            open_sessions += not s.get("finished_at")
            continue
        nxt = starts[i + 1] if i + 1 < len(starts) else None  # a session without finished_at: up to its last row
        mine = [t for t in stamps if t >= t0 and (nxt is None or t < nxt)]
        open_sessions += 1
        if mine:
            secs += (mine[-1] - t0).total_seconds()
        else:
            unknown += 1
    valid_items = {r["item"] for r in rows if r.get("valid")}
    main_valid = {k for k in valid_items if items.get(k, {}).get("kind") == "pair"}
    out["throughput"] = {"wall_s": secs, "unique_valid_items": len(valid_items), "unique_valid_main_items": len(main_valid),
                         "valid_main_items_per_hour": round(len(main_valid) / secs * 3600, 1) if secs and not unknown else None, "sessions_without_finished_at": open_sessions,
                         "sessions_of_unknown_length": unknown,
                         "valid_items_per_hour": round(len(valid_items) / secs * 3600, 1) if secs and not unknown else None,
                         "note": "a stopped session counts up to stopped_at; one without finished_at or stopped_at up to its last row's "
                                 "time ('at'); with a session of unknown length (no row time) the rate is not given",
                         "calls": len(rows), "latency_median_s": sorted(r.get("latency") or 0 for r in rows)[len(rows) // 2] if rows else None}
    if any(it["kind"] == "pair_repeat" for it in items.values()):
        out["repeats"] = repeats(items, first)
    dest = Path(a.out) if a.out else run / "run_report.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: out[k] for k in ("outcomes", "refusal_categories", "throughput")}, ensure_ascii=False))
    if "cost" in out:
        print(f"cost USD {out['cost']['usd_total']} ({out['cost']['usd']}), per valid item {out['cost']['usd_per_valid_item']}")
    if "repeats" in out:
        rp = out["repeats"]
        print(f"repeats: ratings equal {rp['rating_equal_share']} over {rp['items_both_valid']} items; interaction equal "
              f"{rp['interaction_equal_share']} over {rp['profiles_both_complete']} profiles")
    print(f"written {dest}")


if __name__ == "__main__":
    main()
