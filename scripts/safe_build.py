#!/usr/bin/env python3
"""SAFE validation: build the item set of one split for run_b_k2.py (no model call).

Splits (docs/walidacja-safe.md): "dev" = waves 2024Q3 and 2025Q3 without any observation of a firm that is in the
test wave; "test" = wave 2026Q3. Only records that answered the held-out question q0b_3 with 1-10 (the analysis
population; "don't know"/NA are counted in the manifest and not asked). Per record three items, one per card
variant (safe_prompt.VARIANTS); the shuffled card takes its other block from a donor of the same split and the same
size x sector cell (another firm, drawn with the seed; a cell of one record takes a donor from the same size class,
flagged in the item's key row). Since 0.2 the evaluated records are one per firm: a firm seen in both development
waves keeps one of its records (seeded), so the scored observations are independent firms. --sample N draws N firms
per size class (dev pilot); --repeats R adds R repeats of the full card for a fixed seeded subsample of
--repeat-firms firms (stability). Item ids are random and carry no record information. An output directory inside the
repository must be under data/local/ (record-level cards stay out of git). Since 0.4: --elicitation counts (the
default; firms out of 100, parser safe_counts) or probabilities (the first pilot's prompt); --repeat-variants (default
full and segment, so the model noise of the main difference can be measured on the same firms); --exclude-key keeps
the firms of earlier builds out of the evaluated sample (they may still be donors); the extract must have every card
column (a missing column would read as "not asked"), and a record without any donor stops the build. Since 0.5:
--only-repeat-firms-of evaluates exactly the repeated firms of an earlier build (to compare formats on the same firms).
Since 0.6: with --only-repeat-firms-of the shuffled cards take the earlier build's donors (the key's donor_permid and,
from 0.6 on, donor_wave; an earlier key without donor_wave needs the earlier build's --seed, which picks the donor's
record), so the cards are the same; every size class of the split must still have records (with --sample: N firms) after the
exclusions (a class that the exclusions empty stops the build instead of disappearing); a firm both kept by
--only-repeat-firms-of and excluded, or an empty selection, stops the build.

Outputs in --out (record-level, so the directory belongs under data/local/, not in git):
  items.jsonl.gz   {"item", "kind": "safe", "messages"} in a seeded random order (only messages reach the model; kind
                   lets the B-K2 run report read the build)
  design.json      items_sha256 (of the uncompressed JSONL), parser "safe_probabilities", counts, seeds, the key's and
                   the prompt module's sha256
  key.json         item -> split, wave, permid, variant, donor, repeat, the record's answer to q0b_3 (for scoring)
A manifest without record-level data (counts and checksums) is printed for the repository. For the test split the
counts of the target are only "1-10", "99" and "missing" (since 0.3: its distribution stays unseen until the analysis).

    python3 scripts/safe_build.py --safe data/local/safe/safe_poland_all_waves.csv.gz --split dev --sample 30 \
        --repeats 1 --repeat-firms 20 --out data/local/safe_runs/dev_pilot_build
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import safe_prompt as P  # noqa: E402

VERSION = "0.8"
WAVES = {"32": "2024q3", "36": "2025q3", "40": "2026q3"}
ARMS = {"A": ("counts", "safe_counts"), "B": ("counts_nozero", "safe_counts"), "C": ("probabilities", "safe_probabilities")}
TEST = "40"
TARGET = "q0b_3"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--safe", required=True)
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261118)
    ap.add_argument("--sample", type=int, default=None, help="records per size class (default: all)")
    ap.add_argument("--sample-by-size", nargs="+", default=None, metavar="CLASS=N",
                    help="since 0.8: records per size class given separately, every class of the split named once "
                         "(e.g. 1=30 2=30 3=30 4=11, the third pilot's design); excludes --sample")
    ap.add_argument("--repeats", type=int, default=0)
    ap.add_argument("--repeat-firms", type=int, default=0)
    ap.add_argument("--repeat-variants", nargs="+", default=["full", "segment"], choices=list(P.VARIANTS),
                    help="the variants repeated for the --repeat-firms firms (since 0.4 both arms of the main comparison)")
    ap.add_argument("--elicitation", choices=["counts", "counts_nozero", "probabilities"], default="counts",
                    help="counts (since 0.4: firms out of 100, parser safe_counts) or the first pilot's probabilities")
    ap.add_argument("--arms", nargs="+", default=None, choices=list(ARMS),
                    help="since 0.7, the elicitation comparison (plan elicitation_comparison): one item per firm, card and "
                         "arm (A counts with the current zero rule, B counts with the first pilot's zero rule, C probabilities), "
                         "interleaved; the parser is named per item (safe_by_item); needs --only-repeat-firms-of and --variants")
    ap.add_argument("--variants", nargs="+", default=None, choices=list(P.VARIANTS),
                    help="with --arms: the card variants to build (the comparison: full segment)")
    ap.add_argument("--only-repeat-firms-of", default=None,
                    help="key.json of an earlier build: evaluate exactly its firms that had repeats (a comparison of "
                         "formats on the same firms); --sample must not be given")
    ap.add_argument("--exclude-key", nargs="*", default=[],
                    help="key.json of earlier builds whose firms must not be evaluated here (a fresh pilot sample)")
    a = ap.parse_args()
    by_n = None
    if a.sample_by_size is not None:  # since 0.8
        if a.sample is not None or a.only_repeat_firms_of:
            sys.exit("--sample-by-size excludes --sample and --only-repeat-firms-of")
        by_n = {}
        for x in a.sample_by_size:
            c_, _, n_ = x.partition("=")
            if not c_ or not n_.isdigit() or int(n_) < 1 or c_ in by_n:
                sys.exit(f"--sample-by-size {x!r}: CLASS=N with N >= 1, each class once")
            by_n[c_] = int(n_)
    if a.arms is not None:
        if not a.only_repeat_firms_of or not a.variants or a.repeats:
            sys.exit("--arms needs --only-repeat-firms-of and --variants, and takes no --repeats")
        if len(set(a.arms)) != len(a.arms) or len(set(a.variants)) != len(a.variants):
            sys.exit("--arms / --variants: a value named twice")
    elif a.variants is not None:
        sys.exit("--variants only with --arms")
    if len(set(a.repeat_variants)) != len(a.repeat_variants):  # since 0.6: a duplicate would give two items one key
        sys.exit("--repeat-variants: a variant named twice")
    out = Path(a.out)
    repo = Path(__file__).resolve().parents[1]
    if out.resolve().is_relative_to(repo) and not out.resolve().is_relative_to(repo / "data/local"):
        sys.exit("the build holds record-level cards: --out inside the repository must be under data/local/")
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} is not empty")
    raw = Path(a.safe).read_bytes()
    reader = csv.DictReader(gzip.decompress(raw).decode("utf-8").splitlines())
    key = P.load_key()
    need = ({"permid", "wave", TARGET} | {f["safe"] for f in key["fields"]} | {i["safe"] for i in key["sources_of_financing"]["items"]}
            | {i["safe"] for i in key["applications"]["items"]} | {i["outcome"] for i in key["applications"]["items"]})
    missing_cols = sorted(need - set(reader.fieldnames or []))
    if missing_cols:  # a missing column would be rendered as "not asked": refuse instead
        sys.exit(f"the extract lacks card columns: {missing_cols}")
    rows = [r for r in reader if r["wave"] in WAVES]
    test_firms = {r["permid"] for r in rows if r["wave"] == TEST}
    pool = ([r for r in rows if r["wave"] == TEST] if a.split == "test" else
            [r for r in rows if r["wave"] != TEST and r["permid"] not in test_firms])
    answers = Counter(P.norm(r[TARGET]) or "missing" for r in pool)
    if a.split == "test":  # since 0.3: the test wave's distribution of the answers is not written or printed
        answers = Counter("1-10" if v in P.SCALE else v for v in answers.elements())
    pool = [r for r in pool if P.norm(r[TARGET]) in P.SCALE]
    excluded = set()
    for kp in a.exclude_key:
        excluded |= {k["permid"] for k in json.loads(Path(kp).read_text()).values()}
    donors_pool = pool[:]  # donors may come from excluded firms (they are not evaluated here)
    rng = random.Random(a.seed)
    per_firm = defaultdict(list)
    for r in sorted(donors_pool, key=lambda r: (r["permid"], r["wave"])):
        per_firm[r["permid"]].append(r)
    pool = [rs[0] if len(rs) == 1 else rng.choice(rs) for _, rs in sorted(per_firm.items())]
    firms_in_two_waves = sum(len(rs) > 1 for rs in per_firm.values())
    by_size = defaultdict(list)
    only = None
    if a.only_repeat_firms_of:
        if a.sample is not None:
            sys.exit("--only-repeat-firms-of evaluates a fixed set of firms: no --sample")
        code = {v: k for k, v in WAVES.items()}
        only = {(code[k["wave"]], k["permid"]) for k in json.loads(Path(a.only_repeat_firms_of).read_text()).values()
                if k["repeat"] > 0}
        if not only:
            sys.exit("--only-repeat-firms-of: the key has no repeated firms")
        clash = sorted(p_ for _, p_ in only if p_ in excluded)
        if clash:
            sys.exit(f"--only-repeat-firms-of and --exclude-key: {len(clash)} firms are both kept and excluded")
        earlier = json.loads(Path(a.only_repeat_firms_of).read_text())
        donor_of = {}
        for k in earlier.values():
            if k["variant"] == "shuffled" and k["repeat"] == 0:
                donor_of[(code[k["wave"]], k["permid"])] = (k["donor_permid"], code.get(k.get("donor_wave")))
        if any(w is None for _, w in donor_of.values()):  # an earlier key without donor_wave: the seed picks the record
            ed = Path(a.only_repeat_firms_of).parent / "design.json"
            if not ed.exists() or json.loads(ed.read_text()).get("seed") != a.seed:
                sys.exit("--only-repeat-firms-of: the earlier key has no donor_wave; give the earlier build's --seed "
                         "(its design.json must sit next to key.json)")
    # the evaluated records: one per firm, or exactly the earlier build's records (same wave) with --only-repeat-firms-of
    evaluated = ([r for r in donors_pool if (r["wave"], r["permid"]) in only] if only is not None else pool)
    if only is not None and len(evaluated) != len(only):
        sys.exit("--only-repeat-firms-of: some records of the earlier build are not in this split")
    classes = sorted({P.norm(r["d1_rec"]) for r in evaluated})  # the size classes before the exclusions
    for r in evaluated:
        if r["permid"] in excluded:
            continue
        by_size[P.norm(r["d1_rec"])].append(r)
    if by_n is not None and set(by_n) != set(classes):
        sys.exit(f"--sample-by-size must name exactly the size classes of the split {classes}, got {sorted(by_n)}")
    chosen = []
    for s in classes:
        rs = sorted(by_size[s], key=lambda r: (r["wave"], r["permid"]))
        want = by_n[s] if by_n is not None else a.sample
        if not rs and want is None:  # since 0.6 also without --sample (code review round 6, D4)
            sys.exit(f"size class {s}: 0 records left after the exclusions")
        if want is not None:
            if len(rs) < want:
                sys.exit(f"size class {s}: {len(rs)} records, {want} asked")
            rs = rng.sample(rs, want)
        chosen += rs
    if not chosen:
        sys.exit("no record left to evaluate")
    cells = defaultdict(list)
    for r in pool:  # donors come from the whole split, not only the sample
        cells[(P.norm(r["d1_rec"]), P.norm(r["d3_rec"]))].append(r)
    items, keyrows, flags = [], {}, Counter()
    used = set()

    def new_id() -> str:
        while True:
            u = "%012x" % rng.getrandbits(48)
            if u not in used:
                used.add(u)
                return u

    rep_firms = set()
    if a.repeats:
        ids = sorted((r["wave"], r["permid"]) for r in chosen)
        rep_firms = set(rng.sample(ids, min(a.repeat_firms, len(ids))))
    by_rec = {(d["wave"], d["permid"]): d for d in donors_pool}
    by_permid = {d["permid"]: d for d in pool}
    for r in sorted(chosen, key=lambda r: (r["wave"], r["permid"])):
        if only is not None:  # since 0.6: the earlier build's donor, so the shuffled card is the same
            if (r["wave"], r["permid"]) not in donor_of:
                sys.exit("--only-repeat-firms-of: the earlier key has no shuffled card for a kept firm")
            dp, dw = donor_of[(r["wave"], r["permid"])]
            donor = by_rec.get((dw, dp)) if dw is not None else by_permid.get(dp)
            if donor is None:
                sys.exit("--only-repeat-firms-of: an earlier donor is not in this split")
            fallback = (P.norm(donor["d3_rec"]) != P.norm(r["d3_rec"]))
            cell = [donor]
        else:
            cell = [d for d in cells[(P.norm(r["d1_rec"]), P.norm(r["d3_rec"]))] if d["permid"] != r["permid"]]
            fallback = not cell
        if fallback:
            flags["donor_from_size_class_only"] += 1
            if only is None:
                cell = [d for d in pool if P.norm(d["d1_rec"]) == P.norm(r["d1_rec"]) and d["permid"] != r["permid"]]
        if not cell:
            sys.exit(f"no donor for a record of size class {P.norm(r['d1_rec'])}: the split is too small for shuffled cards")
        donor = rng.choice(sorted(cell, key=lambda d: (d["wave"], d["permid"])))
        if a.arms is not None:  # since 0.7: one item per card and arm, first draw only
            plan = [(v, 0, arm) for v in a.variants for arm in a.arms]
        else:
            plan = [(v, 0, None) for v in P.VARIANTS]
            if (r["wave"], r["permid"]) in rep_firms:
                plan += [(v, k, None) for v in a.repeat_variants for k in range(1, a.repeats + 1)]
        for v, k, arm in plan:
            iid = new_id()
            mode = ARMS[arm][0] if arm else a.elicitation
            it = {"item": iid, "kind": "safe", "messages": P.messages(r, v, key, donor if v == "shuffled" else None, mode)}
            if arm:
                it["parser"] = ARMS[arm][1]
            items.append(it)
            keyrows[iid] = {"split": a.split, "wave": WAVES[r["wave"]], "permid": r["permid"], "variant": v, "arm": arm,
                            "repeat": k, "donor_permid": donor["permid"] if v == "shuffled" else None,
                            "donor_wave": WAVES[donor["wave"]] if v == "shuffled" else None,
                            "donor_from_size_class_only": fallback if v == "shuffled" else None,
                            "size": P.norm(r["d1_rec"]), "sector": P.norm(r["d3_rec"]), "answer": P.norm(r[TARGET])}
    rng.shuffle(items)
    text = "".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items).encode("utf-8")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "items.jsonl.gz", "wb") as raw_f, gzip.GzipFile(fileobj=raw_f, mode="wb", mtime=0) as fh:
        fh.write(text)
    (out / "key.json").write_text(json.dumps(keyrows, ensure_ascii=False, indent=0) + "\n")
    design = {"script": f"safe_build {VERSION}", "script_sha256": sha(Path(__file__).read_bytes()),
              "source_sha256": sha(raw), "split": a.split, "seed": a.seed, "sample_per_size": a.sample, "sample_by_size": by_n,
              "repeats": a.repeats, "repeat_firms": len(rep_firms), "parser": ("safe_by_item" if a.arms else {"counts": "safe_counts", "counts_nozero": "safe_counts", "probabilities": "safe_probabilities"}[a.elicitation]),
              "arms": {arm: ARMS[arm][0] for arm in a.arms} if a.arms else None, "variants_built": a.variants,
              "elicitation": None if a.arms else a.elicitation, "repeat_variants": a.repeat_variants,
              "only_repeat_firms_of_an_earlier_build": len(only) if only is not None else None,
              "excluded_firms_from_earlier_builds": len(excluded & {r["permid"] for r in donors_pool}),
              "items": len(items), "records": len(chosen), "items_sha256": sha(text),
              "firms_in_two_waves_one_record_kept": firms_in_two_waves,
              "key_json_sha256": sha((out / "key.json").read_bytes()),
              "card_key_sha256": sha(P.KEY_PATH.read_bytes()), "prompt_sha256": sha(Path(P.__file__).read_bytes()),
              "target_answers_in_split": dict(answers), "records_by_size": dict(Counter(P.norm(r["d1_rec"]) for r in chosen)),
              "variants": dict(Counter(k["variant"] for k in keyrows.values())), "flags": dict(flags)}
    (out / "design.json").write_text(json.dumps(design, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: design[k] for k in ("split", "records", "items", "items_sha256", "target_answers_in_split",
                                              "records_by_size", "variants", "flags")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
