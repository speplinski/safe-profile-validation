#!/usr/bin/env python3
"""B-K2 runner: build the items, ask the model (one separate conversation per item), analyse both families of the
plan (data/reference/b_k2_outcome_pairs.json; docs/tor-b-etap1.md).

    python3 scripts/run_b_k2.py build   --out data/processed/b_runs/<label> [--pair-per-class 10] [--pop-per-class 10]
    python3 scripts/run_b_k2.py run     --out <same> --model <tag> [--api ollama|openai|anthropic|openai-responses] [--base-url ...]
    python3 scripts/run_b_k2.py analyze --out <same>

build: profiles are drawn ONCE for every active firm of the sample (GUS scope, weight classes round(weight_active, 6)):
independent and replicates 1-5 of mrit_rank and stress_coupled (layer2.draw_population), saved with their SHA-256;
cases are chosen only from the saved profiles. Pair trials: per function, base profiles (independent) that qualify
(b_k2_plan.pair_versions), drawn per size class; each gives four items (LL, HL, LH, HH). Population cases: active
firms with B2B clients, drawn per size class; each gives 11 items (independent, five replicates of each variant).
The achieved coupling is computed on the case set (profile_check.achieved_coupling(..., subset=...)), and the
classes each population test keeps follow from it (b_k2_plan.population_contrasts). Every item holds its messages.

run: items in a random order (seeded), every item a fresh conversation without ids; an answer is valid only if
b_k2_prompt.parse accepts it; an invalid one is asked again (a fresh conversation) up to --max-attempts; resumable;
one run per directory at a time (lock); provenance in run.json (home directory masked). Since 0.3: before every write
the path answers.jsonl must still name the file appended to, at the size left by the last write (inode and size); a
file replaced, truncated or removed during the run (e.g. by a sync or a checkout) stops the run at once (also after the
last write, before finished_at): every answer of the session is then saved to answers.jsonl.orphan-<time>, since a sync
that took its copy earlier leaves later answers only in the unlinked file, and a resume asks again what the path lacks.
The progress counts each pass apart (pass 1: the items to ask in this session; then each pass of retries: an invalid
answer, or a call that failed in transport after its transport retries, which is no model attempt), so a count never
exceeds its denominator. Since 0.4 every answer row also holds what the server reports about the call: prompt tokens
taken from the prefix cache and the server's durations of loading, prompt evaluation, generation and in total
(llm_serving_check.server_timing; None where the server does not report them; a None duration from Ollama can also
mean exactly 0, which Ollama omits). Since 0.5 the OpenAI-compatible path needs --chat-template-kwargs given explicitly
(e.g. '{"enable_thinking": false}' for vLLM and Qwen; '{}' to leave the server's default on purpose); it must match
--think, is sent with every request and recorded in run.json, so stage A and the run cannot send different templates.
--stage-a-report <b_k2_input_identity report> checks before the first call that the report passes on this build, is
of this server type, has this run's request settings and model, and (vLLM) that the body this run sends for each of
its cards has the SHA-256 the report recorded; its SHA-256 is kept in run.json.
Since 0.6, --api anthropic (the Messages API through the official anthropic SDK; the key only from the environment
variable ANTHROPIC_API_KEY, never written anywhere): the body is llm_serving_check.anthropic_chat_body (the system
message as the top-level system field, thinking {"type": "disabled"}, output_config {"effort": --effort}, no
temperature, top_p or top_k: Claude Haiku 5.5 rejects non-default values). The SDK's own retries are off
(max_retries 0): transport errors (429, 5xx, timeouts, connection) are retried only by this runner, up to
--transport-retries per call, waiting at least the server's retry-after, each retry counted in the row
(transport_attempts) and in the session's summary; a 4xx other than 429 stops the run at once (a configuration
error). A RateGate shared by the threads reads the rate-limit headers and makes every next call wait for the window's
reset when the requests or tokens left fall below what the running calls need; the waits are printed and counted.
Every row has an outcome kept apart: valid, refusal (stop_reason refusal, with its category), cut (the token limit),
invalid_format (the parser refused the text) or transport; 'brak podstaw' is a rating inside a valid answer. Rows
of the Messages API also hold the four usage counts (input, cache write, cache read, output), the request id and the
model that served the call; run.json holds the model's facts from the Models API and the SDK's version.
--max-usd (required with --api anthropic, as in WorkTrace's runner): the spend so far, from the usage of every row
in answers.jsonl priced with --pricing (data/reference/anthropic_pricing_*.json), cumulative over resumed sessions;
the check runs right before every send (at the start, after the rate-limit wait, before every retry), so once the
cap is reached no request is sent (calls already sent finish and are saved), the run stops with a message and the
same command with a higher cap continues; the cap and the spend are recorded per session. Since 0.7 every row keeps
its own cost (by kind, with the price file's name and SHA-256) at the rates in force when it was asked, so a later
price file never reprices the history; a fatal error or a streak of transport errors stops new calls and the calls
already sent are received and saved before the run stops; every row carries the time it was written ('at'). A session
stopped this way (a fatal error, a streak, the spend cap) gets stopped_at and stopped_reason instead of finished_at, so
the run never reads as finished. Ctrl-C and the lost-file stop still end at once (os._exit): a call in flight is then
not saved, its cost is not counted, and a resume asks it again.
Since 0.9 (0.9.1: the resume check of the parser and the data/local guard for SAFE runs), run reads the answer parser from the build's design.json ("parser"; absent = "b_k2", b_k2_prompt.parse):
"safe_probabilities" (safe_prompt.parse, the SAFE validation's explicit probabilities 1-10) or, since 0.9.2,
"safe_counts" (safe_prompt.parse_counts: firms out of 100 with an identical card) for item sets of safe_build.py; the parser is recorded in the run's configuration when it is not b_k2, a resume with another parser is
refused (also when design.json lost the key), a SAFE run directory inside the repository must be under data/local/
(record-level cards stay out of git); analyze refuses such builds.
Since 0.8, --api openai-responses (the OpenAI Responses API through the official openai SDK; the key only from the
environment variable OPENAI_API_KEY, never written anywhere), with everything said above for --api anthropic (the
SDK's retries off, the runner's visible retries, a 4xx other than 429 stops the run, the rate-limit gate, the outcomes,
--max-usd with each row's stored cost) except: the body is llm_serving_check.openai_responses_body (the messages as
they are in input, the system message included, reasoning {"effort": --effort} given explicitly, no sampling
parameters, no response format, store false, service_tier 'default'); a 429 for an exhausted quota
(insufficient_quota) is fatal, not retried; the gate reads x-ratelimit-remaining/reset-{requests,tokens}; a refusal is
a refusal item in the output or an incomplete response for content_filter, a cut an incomplete response for
max_output_tokens; rows hold the usage counts (input with cache reads and writes, cached, cache written by retention,
output with reasoning), the request and response ids, the model and service tier that served the call; prices from
--pricing (data/reference/openai_pricing_*.json, the default for this API), the long-context row above 272,000 input
tokens. Prompt caching is the API's default (not disabled, not steered). A response with a status other than completed
that is neither a cut nor a refusal is its own outcome, incomplete (never valid); one without usage is a transport
error. An error message is stored without anything that looks like a key, and not at all for 401 and 403. As before,
a run directory is resumed only by the same runner version (config 'script'): a new version starts a new directory.

analyze: the pair family (b_k2_plan.pair_test per function, pair_family over the four) and the population family
(per firm: rating variant minus independent per replicate, minus the main effects of the four coupled fields
estimated within firms, mean over replicates; weighted mean with size classes weighted by their share of active
firms; sign-flip test; Holm over the four; exploratory contrast apart). A failed or invalid answer is missing
('invalid'), never 'brak podstaw'.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import datetime as dt
import gzip
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import b_k2_plan as bp  # noqa: E402
import b_k2_prompt as bpr  # noqa: E402
import profile_check as pc  # noqa: E402
from dnba.population import card as card_mod, layer2 as l2  # noqa: E402
from ollama_provenance import mask_home  # noqa: E402

VERSION = "0.9.5"
TEST_ENV = "DNBA_B_K2_TEST"  # --firms-limit only in tests
FIRMS = ROOT / "data/population/firms_regon_2026q2_krs_strata-size_class-min500_n10000_seed42.csv"
AUTH = ROOT / "data/population/authority_regon_2026q2_krs_strata-size_class-min500_n10000_seed42_krs20261107.jsonl.gz"
VARIANTS = ("mrit_rank", "stress_coupled")
COUPLED = ("largest_b2b_client_share", "overdue_receivables_share", "overdue_payables_share", "liquidity_declared")


def sha(b: bytes | str) -> str:
    return hashlib.sha256(b.encode() if isinstance(b, str) else b).hexdigest()


def role_of(firm: dict, authority: dict | None) -> str:
    for p in (authority or {}).get("persons", []):
        if p.get("function") in card_mod.FUNCTION_TEXT:
            return card_mod.FUNCTION_TEXT[p["function"]]
    return "właściciel" if firm["legal_form"] in ("099", "019") else "prezes zarządu"


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def population(limit: int | None = None):
    firms = list(csv.DictReader(FIRMS.open(encoding="utf-8")))
    if limit:
        firms = firms[:limit]
    auth = {x["firm_id"]: x for x in map(json.loads, gzip.open(AUTH))}
    aw = {r["firm_id"]: float(r["weight_active"]) for r in csv.DictReader(open(os.path.splitext(FIRMS)[0] + ".active_weights.csv"))
          if r.get("scope", "gus_dpn") == "gus_dpn"}
    active = [f for f in firms if aw.get(f["firm_id"], 0.0) > 0]
    return active, auth, aw




RUN_MATERIALS = ("scripts/run_b_k2.py", "scripts/llm_serving_check.py", "scripts/ollama_provenance.py", "scripts/safe_prompt.py")


def run_materials() -> dict:
    """Since 0.9.5: the sha256 of the modules that build, send and parse the requests, as they are in this session (the
    SAFE confirmatory analysis compares them with the frozen ones, so a change for the time of a run is seen)."""
    return {f: sha((ROOT / f).read_bytes()) for f in RUN_MATERIALS}


def git_state() -> dict:
    try:
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip())
    except Exception:
        dirty = None
    return {"commit": git_commit(), "dirty": dirty}


def read_jsonl(path: Path) -> list[dict]:
    """Rows of a JSONL file (plain or .gz). A damaged line stops the run (repair_tail mends only an interrupted last
    write, under the lock, before anything is appended)."""
    if not path.exists():
        return []
    raw = gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(encoding="utf-8")
    out = []
    with raw as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                sys.exit(f"{path.name}: line {n} is damaged; an interrupted last write is mended by 'run', any other damage "
                         "needs a look by hand (restore the file)")
    return out


def repair_tail(path: Path) -> str | None:
    """Before appending (under the run's lock): a file whose last write was interrupted ends without a newline. A
    complete last record gets its newline; an incomplete one is moved to <name>.broken-<time> and cut off. Returns
    what was done, or None."""
    if not path.exists() or path.stat().st_size == 0:
        return None
    data = path.read_bytes()
    if data.endswith(b"\n"):
        return None
    cut = data.rfind(b"\n") + 1
    tail = data[cut:]
    try:
        json.loads(tail.decode("utf-8"))
        with path.open("ab") as fh:
            fh.write(b"\n")
            fh.flush()
            os.fsync(fh.fileno())
        return "a complete last record without its newline: newline added"
    except (json.JSONDecodeError, UnicodeDecodeError):
        stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
        (path.parent / f"{path.name}.broken-{stamp}").write_bytes(tail)
        with path.open("r+b") as fh:
            fh.truncate(cut)
            fh.flush()
            os.fsync(fh.fileno())
        return f"an incomplete last record moved to {path.name}.broken-{stamp}"


def item(kind: str, f: dict, a: dict | None, prof: dict, order: list, weight: float, **extra) -> dict:
    role = role_of(f, a)
    card = card_mod.render(f, prof, a, experimental=True)
    return {"kind": kind, "firm_id": f["firm_id"], "size": f["size_class"], "weight": weight, "role": role, "order": order, **extra,
            "messages": [{"role": "system", "content": bpr.system_prompt(role, order)},
                         {"role": "user", "content": bpr.USER_MESSAGE.format(card=card)}],
            "fields": {k: prof.get(k) for k in COUPLED}}


def shares(firms: list, aw: dict) -> dict:
    tot = sum(aw[f["firm_id"]] for f in firms)
    return {s: (sum(aw[f["firm_id"]] for f in firms if f["size_class"] == s) / tot if tot else 0.0) for s in bp.SIZES}


def draw_profiles(a):
    if a.firms_limit and os.environ.get(TEST_ENV) != "1":
        sys.exit("--firms-limit is for tests only")
    active, auth, aw = population(a.firms_limit)
    wclass = {f["firm_id"]: round(aw[f["firm_id"]], 6) for f in active}
    indep = l2.draw_population(active, auth, "independent", wclass)
    reps = {d: [l2.draw_population(active, auth, d, wclass, replicate=r, base=indep) for r in pc.REPLICATES] for d in VARIANTS}
    return active, auth, aw, wclass, indep, reps


def population_pool(active, indep) -> dict:
    """Population cases: active firms with B2B clients (the receivables fields apply), by size class."""
    return {s: [f for f, p in zip(active, indep) if f["size_class"] == s and p["activity_status"] == "active"
                and p.get("overdue_receivables_share") is not None] for s in bp.SIZES}


def build(a) -> None:
    out = Path(a.out)
    if (out / "items.jsonl.gz").exists():
        sys.exit(f"{out} already has items: build writes only into a new directory")
    out.mkdir(parents=True, exist_ok=True)
    plan = bp.load_plan()
    ids = [f["id"] for f in plan["functions"]]
    active, auth, aw, wclass, indep, reps = draw_profiles(a)
    prof_lines = [json.dumps({"firm_id": f["firm_id"], "dependence": "independent", "replicate": 0, "profile": p}, ensure_ascii=False, sort_keys=True)
                  for f, p in zip(active, indep)]
    prof_lines += [json.dumps({"firm_id": f["firm_id"], "dependence": d, "replicate": r, "profile": p}, ensure_ascii=False, sort_keys=True)
                   for d in VARIANTS for r, rr in zip(pc.REPLICATES, reps[d]) for f, p in zip(active, rr)]
    prof_bytes = ("\n".join(prof_lines) + "\n").encode()
    with open(out / "profiles.jsonl.gz", "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as fh:
        fh.write(prof_bytes)
    rng = random.Random(a.seed)
    idx = {f["firm_id"]: i for i, f in enumerate(active)}
    items, pair_share = [], {}
    # pair trials: base profiles from the saved independent profiles that qualify for the function's pair; simple
    # random sampling inside a class, firms weighted by weight_active inside it in the analysis (the target: the
    # qualifying active firms)
    for fn in plan["functions"]:
        pool = {s: [] for s in bp.SIZES}
        for f, p in zip(active, indep):
            vs = bp.pair_versions(p, f, fn["pair"], plan)
            if vs:
                pool[f["size_class"]].append((f, vs))
        pair_share[fn["id"]] = shares([f for s in bp.SIZES for f, _ in pool[s]], aw)
        for s in bp.SIZES:
            for f, vs in rng.sample(pool[s], min(a.pair_per_class, len(pool[s]))):
                order = bpr.order_for(int(f["firm_seed"]) % 2 ** 32, ids)
                for v, q in vs.items():
                    items.append(item("pair", f, auth.get(f["firm_id"]), q, order, aw[f["firm_id"]], function=fn["id"],
                                      pair=fn["pair"], version=v, item=f"pair/{fn['id']}/{f['firm_id']}/{v}"))
    pool = population_pool(active, indep)
    pop_share = shares([f for s in bp.SIZES for f in pool[s]], aw)
    cases = [f for s in bp.SIZES for f in rng.sample(pool[s], min(a.pop_per_class, len(pool[s])))]
    for f in cases:
        order = bpr.order_for(int(f["firm_seed"]) % 2 ** 32, ids)
        i = idx[f["firm_id"]]
        items.append(item("population", f, auth.get(f["firm_id"]), indep[i], order, aw[f["firm_id"]], dependence="independent",
                          replicate=0, item=f"population/{f['firm_id']}/independent/0"))
        for d in VARIANTS:
            for r, rr in zip(pc.REPLICATES, reps[d]):
                items.append(item("population", f, auth.get(f["firm_id"]), rr[i], order, aw[f["firm_id"]], dependence=d, replicate=r,
                                  item=f"population/{f['firm_id']}/{d}/{r}"))
    copula = {d: [l2.draw(f, auth.get(f["firm_id"]), dependence=d) for f in active] for d in VARIANTS}
    achieved = pc.achieved_coupling(active, aw, wclass, indep, reps, copula, subset={f["firm_id"] for f in cases})
    contrasts = bp.population_contrasts(plan, achieved)
    items_bytes = "".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in items).encode()
    with open(out / "items.jsonl.gz", "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as fh:
        fh.write(items_bytes)
    design = {"script": VERSION, "layer2_version": l2.VERSION, "git": git_state(), "seed": a.seed,
              "pair_per_class": a.pair_per_class, "pop_per_class": a.pop_per_class, "firms_limit": a.firms_limit,
              "weights": "inside a class: weight_active of the firm (simple random sampling inside a class); classes: share of "
                         "the qualifying active firms (pair: per function; population: firms with B2B clients)",
              "class_share_pair": pair_share, "class_share_population": pop_share,
              "profiles_sha256": sha(prof_bytes), "items_sha256": sha(items_bytes), "files_sha256": current_files(),
              "counts": {"pair_items": sum(x["kind"] == "pair" for x in items), "population_items": sum(x["kind"] == "population" for x in items),
                         "population_cases": len(cases)},
              "population_contrasts": contrasts, "achieved_coupling_case_set": achieved}
    (out / "design.json").write_text(json.dumps(design, ensure_ascii=False, indent=1))
    print(f"{len(items)} items ({design['counts']}) -> {out}")
    for c in contrasts:
        print(f"  {c['role']:11s} {c['function']:24s} {c['variant']:15s} keep {c['keep']}")


def current_files() -> dict:
    return {k: sha(Path(p).read_bytes()) for k, p in (("plan", bp.PLAN), ("key", bpr.KEY), ("prompt_script", bpr.__file__),
                                                      ("plan_script", bp.__file__), ("runner", __file__))}


def plan_sizes(a) -> None:
    """Without any model call: how often each population test keeps each class, over many draws of the case set, for
    a grid of case-set sizes (the rule is checked on the case set, so one lucky seed proves nothing)."""
    plan = bp.load_plan()
    active, auth, aw, wclass, indep, reps = draw_profiles(a)
    copula = {d: [l2.draw(f, auth.get(f["firm_id"]), dependence=d) for f in active] for d in VARIANTS}
    pool = population_pool(active, indep)
    res = {"draws": a.draws, "grid": a.grid, "pool": {s: len(v) for s, v in pool.items()}, "keep_rate": {}}
    for n in a.grid:
        tally = {}
        for k in range(a.draws):
            rng = random.Random(a.seed * 1000 + n * 7 + k)
            cases = {f["firm_id"] for s in bp.SIZES for f in rng.sample(pool[s], min(n, len(pool[s])))}
            ach = pc.achieved_coupling(active, aw, wclass, indep, reps, copula, subset=cases)
            for c in bp.population_contrasts(plan, ach):
                key = f"{c['role']}|{c['function']}|{c['variant']}"
                t = tally.setdefault(key, {s: 0 for s in bp.SIZES} | {"any": 0})
                for s in c["keep"]:
                    t[s] += 1
                t["any"] += bool(c["keep"])
        res["keep_rate"][str(n)] = {key: {s: v / a.draws for s, v in t.items()} for key, t in tally.items()}
        print(f"n={n}: " + "; ".join(f"{key.split('|')[1]} any {t['any'] / a.draws:.2f}" for key, t in tally.items()), flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "plan_sizes.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


class Lock:
    def __init__(self, path: Path):
        self.path = path

    def __enter__(self):
        import fcntl
        self.fh = open(self.path, "w")
        try:
            fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            sys.exit(f"another process holds {self.path.parent} (a run in progress)")
        return self

    def __exit__(self, *e):
        self.fh.close()


def safe_url(url: str) -> str:
    """The server address as recorded: a local one as is, any other reduced to 'remote' and its port (no host names)."""
    from urllib.parse import urlparse
    u = urlparse(url)
    return url if u.hostname in ("127.0.0.1", "localhost", "::1") else f"{u.scheme}://remote:{u.port or ''}"


def anthropic_client(a):
    import anthropic
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("--api anthropic needs the key in the environment variable ANTHROPIC_API_KEY")
    return anthropic.Anthropic(max_retries=0, **({"base_url": a.base_url} if a.base_url_given else {}))


PAID = ("anthropic", "openai-responses")  # APIs billed per token: a spend cap, a stored cost per row, a rate-limit gate


def openai_client(a):
    import openai
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("--api openai-responses needs the key in the environment variable OPENAI_API_KEY")
    return openai.OpenAI(max_retries=0, **({"base_url": a.base_url} if a.base_url_given else {}))


def server_facts(a) -> dict:
    if a.api == "openai-responses":
        import openai
        client = openai_client(a)
        try:
            m = client.models.retrieve(a.model)
        except openai.APIConnectionError as e:
            sys.exit(f"no connection to the API at {safe_url(str(client.base_url))} ({type(e).__name__}: {e.__cause__!r}); "
                     f"check the network, HTTPS_PROXY and OPENAI_BASE_URL")
        except openai.APIStatusError as e:
            sys.exit(f"the Models API answered HTTP {e.status_code} for {a.model!r}: "
                     + ("the key is not accepted" if e.status_code in (401, 403) else "the model is not available to this key"
                        if e.status_code == 404 else str(getattr(e, "message", ""))[:200]))
        return {"openai_sdk": openai.__version__,
                "model": {"id": m.id, "created": getattr(m, "created", None), "owned_by": getattr(m, "owned_by", None)}}
    if a.api == "anthropic":
        import anthropic
        client = anthropic_client(a)
        try:
            m = client.models.retrieve(a.model)
        except anthropic.APIConnectionError as e:  # no answer from the server at all: the network, a proxy, a base URL
            sys.exit(f"no connection to the API at {client.base_url} ({type(e).__name__}: {e.__cause__!r}); check the network, "
                     f"HTTPS_PROXY and ANTHROPIC_BASE_URL")
        except anthropic.APIStatusError as e:
            sys.exit(f"the Models API answered HTTP {e.status_code} for {a.model!r}: "
                     + ("the key is not accepted" if e.status_code in (401, 403) else "the model is not available to this key"
                        if e.status_code == 404 else str(getattr(e, "message", ""))[:200]))
        return {"anthropic_sdk": anthropic.__version__, "anthropic_version": anthropic_client(a).default_headers.get("anthropic-version"),
                "model": {"id": m.id, "display_name": getattr(m, "display_name", None),
                "created_at": str(getattr(m, "created_at", None)), "max_input_tokens": getattr(m, "max_input_tokens", None),
                "max_tokens": getattr(m, "max_tokens", None)}}
    if a.api != "ollama":
        return {}
    from ollama_provenance import api_facts, manifest
    f = api_facts(a.base_url, a.model)
    try:
        f["manifest"] = manifest(a.model)
    except Exception as e:
        f["manifest_error"] = type(e).__name__
    return mask_home(f)


EFFORTS = {"anthropic": ("low", "medium", "high"), "openai-responses": ("none", "low", "medium", "high", "xhigh", "max")}
PRICING = {"anthropic": ROOT / "data/reference/anthropic_pricing_2026-11-15.json",
           "openai-responses": ROOT / "data/reference/openai_pricing_2026-11-17.json"}
TIMING = ("prompt_cached_tokens", "load_s", "prompt_eval_s", "eval_s", "total_s")  # since 0.4; None if not reported
ANTHROPIC_FIELDS = ("cache_creation_input_tokens", "cache_read_input_tokens", "stop_reason", "refusal_category",
                    "request_id", "served_model", "empty_content")  # since 0.6 (empty_content since 0.7)
OPENAI_FIELDS = ("cached_tokens", "cache_write_tokens", "cache_write_30m_tokens", "cache_write_12h_tokens", "reasoning_tokens",
                 "stop_reason", "incomplete_reason", "refusal_category", "refusal_text", "request_id", "response_id",
                 "served_model", "service_tier", "empty_content")  # since 0.8


def request_body(a, messages: list) -> dict:
    """The body ask() sends for these messages (the same constructors chat and chat_ollama use)."""
    from llm_serving_check import anthropic_chat_body, ollama_chat_body, openai_chat_body, openai_responses_body
    if a.api == "openai-responses":
        return openai_responses_body(a.model, messages, a.max_tokens, a.effort)
    if a.api == "ollama":
        return ollama_chat_body(a.model, messages, a.max_tokens, a.seed, a.think, a.num_ctx)
    if a.api == "anthropic":
        return anthropic_chat_body(a.model, messages, a.max_tokens, a.effort)
    return openai_chat_body(a.model, messages, a.max_tokens, a.seed, chat_template_kwargs=a.ctk)


def check_stage_a(a, out: Path, design: dict) -> str:
    if design.get("parser", "b_k2") != "b_k2":
        sys.exit("--stage-a-report: stage A applies to B-K2 builds only (see the SAFE plan, stage_a)")
    """The stage A report a.stage_a_report matches this run: it passes on this build, is of this server type, and has
    this run's request settings and model; for vLLM every card's recorded request SHA-256 is the body this run sends."""
    import b_k2_input_identity as bi
    p = Path(a.stage_a_report)
    rep = json.loads(p.read_text())
    items, rk, items_sha = bi.load_build(out)
    probs = bi.check_report(rep, items, rk)
    if probs or rep.get("items_sha256") != design["items_sha256"]:
        sys.exit(f"--stage-a-report does not pass on this build: {probs or 'another build'}")
    want = {"ollama": "ollama", "openai": "vllm"}.get(a.api)
    if want is None:
        sys.exit(f"--stage-a-report: no stage A exists for --api {a.api} (the API renders the conversation itself)")
    if rep.get("server") != want:
        sys.exit(f"--stage-a-report is of server {rep.get('server')!r}, this run uses {want!r}")
    req = ({"think": a.think, "num_ctx": a.num_ctx, "max_tokens": a.max_tokens, "seed": a.seed} if a.api == "ollama" else
           {"chat_template_kwargs": a.ctk, "max_tokens": a.max_tokens, "seed": a.seed, "temperature": 0})
    if rep.get("request") != req or rep.get("model") != a.model:
        sys.exit(f"--stage-a-report was made with other request settings or model: {rep.get('request')} {rep.get('model')!r} "
                 f"vs {req} {a.model!r}")
    if a.api == "openai":
        for c in rep["cards"]:
            body = request_body(a, items[c["item"]]["messages"])
            if bi.sha(json.dumps(body, ensure_ascii=False, sort_keys=True).encode()) != c.get("request_sha256"):
                sys.exit(f"--stage-a-report: {c['item']}: this run's request body differs from the one stage A sent")
    return bi.sha(p.read_bytes())


PARSERS = ("b_k2", "safe_probabilities", "safe_counts", "safe_by_item")
ITEM_PARSERS = ("safe_probabilities", "safe_counts")  # safe_by_item: each item names one (since 0.9.4)


def parser_of(name: str):
    """The answer parser a build names in design.json ("parser"; absent = b_k2): b_k2_prompt.parse, or
    safe_prompt.parse for the SAFE validation (explicit probabilities 1-10)."""
    if name == "b_k2":
        return bpr.parse
    if name == "safe_probabilities":
        import safe_prompt
        return safe_prompt.parse
    if name == "safe_counts":
        import safe_prompt
        return safe_prompt.parse_counts
    raise ValueError(f"unknown parser {name!r}")


def ask(a, it: dict, attempt: int) -> dict:
    """One fresh conversation. Transport errors (no answer from the model) are retried here with backoff; they never
    count as an attempt of the model."""
    import time
    from llm_serving_check import chat, chat_anthropic, chat_ollama, chat_openai_responses

    def blocked() -> bool:
        """No request may be sent: the run is halting (a fatal error, a streak of transport errors) or the spend cap is
        reached. Checked right before every send: at the start, after the rate-limit wait and before every retry."""
        h = getattr(a, "halt", None)
        if h is not None and h.is_set():
            return True
        if a.api in PAID:
            with a.spend_lock:
                return a.spent >= a.max_usd
        return False
    if blocked():
        return None  # this call does not start
    tries, r = 0, None
    for k in range(a.transport_retries + 1):
        if k and blocked():
            break  # no retry: the last failed attempt is the row
        if a.api == "ollama":
            tries = k + 1
            r = chat_ollama(a.base_url, a.model, it["messages"], a.max_tokens, a.seed, a.timeout, a.think, a.num_ctx)
        elif a.api in PAID:
            a.gate.wait()
            if blocked():
                if r is None:
                    return None  # nothing was sent for this item
                break
            tries = k + 1
            send = chat_anthropic if a.api == "anthropic" else chat_openai_responses
            r = send(a.client, request_body(a, it["messages"]), a.timeout, a.gate)
        else:
            tries = k + 1
            r = chat(a.base_url, a.model, it["messages"], a.max_tokens, a.seed, a.timeout, os.environ.get("OPENAI_API_KEY", "none"),
                     chat_template_kwargs=a.ctk)
        if r.get("ok") or r.get("fatal"):
            break
        if k < a.transport_retries:
            wait = max(min(60, 2 ** k), r.get("retry_after") or 0)
            if a.api in PAID and r.get("retry_after"):
                a.gate.after(r["retry_after"])
            print(f"  transport: {it['item']}: {r.get('error')}; retry {k + 1}/{a.transport_retries} in {wait:.0f} s", flush=True)
            time.sleep(wait)
    row = {"item": it["item"], "attempt": attempt if r.get("ok") else None, "ok": bool(r.get("ok")), "latency": r.get("latency"),
           "prompt_tokens": r.get("prompt_tokens"), "completion_tokens": r.get("completion_tokens"),
           "finish_reason": r.get("finish_reason"), "text": r.get("text"), "error": r.get("error"), "valid": False,
           "transport_attempts": tries, "outcome": "transport", "at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           **{k: r.get(k) for k in TIMING}}
    if a.api in PAID:
        row.update({k: r.get(k) for k in (ANTHROPIC_FIELDS if a.api == "anthropic" else OPENAI_FIELDS)})
        from llm_serving_check import anthropic_cost, openai_cost
        cost_fn = getattr(a, "cost_fn", None) or (anthropic_cost if a.api == "anthropic" else openai_cost)
        c = cost_fn(row, a.price, by_kind=True)  # priced once, at the rates in force now, and kept with the row
        row["cost"] = {**c, "pricing": a.pricing_name, "pricing_sha256": a.pricing_sha}
        with a.spend_lock:
            a.spent += c["usd"]
    if r.get("fatal"):
        row["fatal"] = True
        row["error"] = f"{r.get('error')}: {r.get('detail') or ''}".strip()
    lock = getattr(a, "streak_lock", None)
    if lock is not None:  # the stop is decided here, as each result comes in, not when the main thread gets to it
        with lock:
            a.streak_n = 0 if r.get("ok") else a.streak_n + 1
            if not a.halt.is_set() and (r.get("fatal") or a.streak_n >= a.stop_after_transport_errors):
                a.halt_reason = (f"{it['item']}: {row['error']} (not retried: a request or account error); fix it, then resume"
                                 if r.get("fatal") else f"{a.streak_n} transport errors in a row (server down?): stopped; resume later")
                a.halt.set()
    if r.get("ok") and r.get("finish_reason") == "refusal":
        row["error"] = f"refusal: {r.get('refusal_category')}"
        row["outcome"] = "refusal"
    elif r.get("ok") and r.get("finish_reason") == "length":
        row["error"] = "cut by the token limit"
        row["outcome"] = "cut"
    elif r.get("ok") and r.get("finish_reason") == "incomplete":  # Responses API: a status other than completed
        row["error"] = f"incomplete response: {r.get('stop_reason')} {r.get('incomplete_reason')}"
        row["outcome"] = "incomplete"
    elif r.get("ok"):
        try:
            pname = it["parser"] if getattr(a, "parser", "b_k2") == "safe_by_item" else getattr(a, "parser", "b_k2")
            row["answer"] = parser_of(pname)(r.get("text") or "")
            row["valid"] = True
            row["outcome"] = "valid"
        except ValueError as e:
            row["error"] = str(e)[:200]
            row["outcome"] = "invalid_format"
    return row


def run(a) -> None:
    import fcntl
    out = Path(a.out)
    design = json.loads((out / "design.json").read_text())
    raw_items = gzip.open(out / "items.jsonl.gz").read()  # since 0.9.3: the bytes the run sends are the bytes checked
    if sha(raw_items) != design.get("items_sha256"):
        sys.exit("items.jsonl.gz does not match design.json (items_sha256): the items were changed after the build; "
                 "nothing is sent")
    items = [json.loads(ln) for ln in raw_items.decode("utf-8").splitlines() if ln.strip()]
    if design.get("firms_limit") and os.environ.get(TEST_ENV) != "1":
        sys.exit("this build used --firms-limit (tests only): not for a real run")
    a.parser = design.get("parser", "b_k2")
    if a.parser not in PARSERS:
        sys.exit(f"design.json names an unknown parser {a.parser!r}")
    if a.parser == "safe_by_item":  # since 0.9.4: a build mixing elicitations; the parser of every item is checked first
        bad = [it["item"] for it in items if it.get("parser") not in ITEM_PARSERS]
        if bad:
            sys.exit(f"safe_by_item: {len(bad)} items without a known parser {ITEM_PARSERS}")
    if a.parser.startswith("safe_"):
        repo = Path(__file__).resolve().parents[1]
        o = out.resolve()
        if o.is_relative_to(repo) and not o.is_relative_to(repo / "data/local"):
            sys.exit("a SAFE build holds record-level cards: its run directory must be under data/local/ (not in git)")
    with Lock(out / ".lock"):
        facts = server_facts(a)
        digest = facts.get("model_digest") or (facts.get("manifest") or {}).get("weights_digest")
        if a.api == "ollama" and not digest:
            sys.exit(f"no model {a.model!r} served at {safe_url(a.base_url)}: {facts.get('tags_error') or facts.get('ollama_version_error') or 'not in /api/tags'}")
        stage_a = check_stage_a(a, out, design) if a.stage_a_report else None
        anth, oai = a.api == "anthropic", a.api == "openai-responses"
        paid = a.api in PAID
        if paid:
            import threading
            from llm_serving_check import OpenAIRateGate, RateGate, anthropic_cost, openai_cost
            a.client = anthropic_client(a) if anth else openai_client(a)
            a.gate = RateGate(a.concurrency) if anth else OpenAIRateGate(a.concurrency, output_per_call=a.max_tokens)
            a.cost_fn = anthropic_cost if anth else openai_cost
            prices = json.loads(Path(a.pricing).read_text())["models"]
            if a.model not in prices:
                sys.exit(f"{a.pricing} has no prices for {a.model!r}")
            a.price, a.spend_lock = prices[a.model], threading.Lock()
            a.pricing_name, a.pricing_sha = Path(a.pricing).name, sha(Path(a.pricing).read_bytes())
        env_url, default_url = (("ANTHROPIC_BASE_URL", "https://api.anthropic.com") if anth else
                                ("OPENAI_BASE_URL", "https://api.openai.com/v1"))
        config = {"script": VERSION, "api": a.api, "model": a.model,
                  "base_url": (safe_url(a.base_url) if a.base_url_given else safe_url(os.environ[env_url])
                               if os.environ.get(env_url) else default_url) if paid else safe_url(a.base_url),
                  "max_tokens": a.max_tokens, "num_ctx": a.num_ctx if a.api == "ollama" else None,
                  "think": None if paid else a.think, "chat_template_kwargs": a.ctk, "stage_a_report_sha256": stage_a,
                  "thinking": {"type": "disabled"} if anth else None, "effort": a.effort if paid else None,
                  "sampling": "not sent (the API's defaults)" if paid else None,
                  "seed": a.seed, "seed_note": ("the seed sets the order of the items (the Messages API takes no seed)" if anth else
                                                "the seed sets the order of the items (the Responses API takes no seed)" if oai else
                                                "the seed sets the order of the items and the model's options.seed"),
                  "temperature": None if paid else 0, "max_attempts": a.max_attempts, "concurrency": a.concurrency,
                  "timeout": a.timeout,
                  "items_sha256": design["items_sha256"], "model_digest": digest}
        if a.parser != "b_k2":  # only for other builds, so a B-K2 run's configuration keeps its fields
            config["parser"] = a.parser
        if oai:  # only for this API, so the configuration of earlier runs (and their resume check) is unchanged
            config.update(reasoning={"effort": a.effort}, store=False, service_tier="default", response_format="not sent (plain text)",
                          prompt_caching="the API's default (not disabled, not steered)")
        rj = out / "run.json"
        now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
        if rj.exists():
            meta = json.loads(rj.read_text())
            diff = {k: (meta["config"].get(k), v) for k, v in config.items() if meta["config"].get(k) != v}
            if meta["config"].get("parser", "b_k2") != a.parser:  # also when design.json lost its "parser" key
                diff["parser"] = (meta["config"].get("parser", "b_k2"), a.parser)
            if diff:
                sys.exit(f"{out} holds a run with another configuration (or another model digest): {diff}")
            meta["sessions"].append({"started_at": now, "git": git_state(), "server": facts, "materials_sha256": run_materials()})
        else:
            meta = {"config": config, "sessions": [{"started_at": now, "git": git_state(), "server": facts,
                                                    "materials_sha256": run_materials()}]}
        rj.write_text(json.dumps(mask_home(meta), ensure_ascii=False, indent=1))
        ap = out / "answers.jsonl"
        fixed = repair_tail(ap)
        if fixed:
            print(f"answers.jsonl: {fixed}")
            meta["sessions"][-1]["repaired_tail"] = fixed
            rj.write_text(json.dumps(mask_home(meta), ensure_ascii=False, indent=1))
        rows = read_jsonl(ap)
        if paid:
            # each row keeps its own cost at the rates in force when it was asked; a row without one (none so far) is
            # priced with the current file and counted apart
            a.spent = sum((r.get("cost") or {}).get("usd", 0.0) for r in rows)
            unpriced = [r for r in rows if r.get("ok") and r.get("prompt_tokens") is not None and not r.get("cost")]
            a.spent += sum(a.cost_fn(r, a.price) for r in unpriced)
            if unpriced:
                print(f"{len(unpriced)} earlier rows have no stored cost: priced with {Path(a.pricing).name}", flush=True)
            meta["sessions"][-1].update(max_usd=a.max_usd, spent_usd_at_start=round(a.spent, 6), transport_retries=a.transport_retries,
                                        pricing=Path(a.pricing).name, pricing_sha256=sha(Path(a.pricing).read_bytes()))
            rj.write_text(json.dumps(mask_home(meta), ensure_ascii=False, indent=1))
            print(f"spent so far ${a.spent:.4f} of the cap ${a.max_usd:.2f}", flush=True)
            if a.spent >= a.max_usd:
                meta["sessions"][-1].update(stopped_at=meta["sessions"][-1]["started_at"], stopped_reason="spend cap reached at the start")
                rj.write_text(json.dumps(mask_home(meta), ensure_ascii=False, indent=1))
                sys.exit(f"the spend so far (${a.spent:.4f}) reaches the cap ${a.max_usd:.2f}: raise --max-usd to continue")
        done = {r["item"] for r in rows if r.get("valid")}
        tries = {}
        for r in rows:
            if r.get("ok"):  # only answers the model gave count as attempts
                tries[r["item"]] = tries.get(r["item"], 0) + 1
        todo = [it for it in items if it["item"] not in done and tries.get(it["item"], 0) < a.max_attempts]
        random.Random(a.seed).shuffle(todo)  # random order; the same order on a resume
        if a.limit:
            todo = todo[:a.limit]
        streak, asked, capped = 0, [], False
        import threading
        a.halt, a.halt_reason, a.streak_lock, a.streak_n = threading.Event(), None, threading.Lock(), 0
        import time
        t0 = time.perf_counter()
        ps = {"pass": 1, "n": 0, "of": len(todo), "valid": 0, "t0": t0}  # the counter of the current pass only

        def progress(r) -> None:
            """One line per answer, counted inside its pass (pass 1: the items to ask in this session; then each pass of
            retries after an invalid answer apart), so that the count never exceeds its denominator: count, valid, this
            answer's time, elapsed, estimated time left of the pass (its mean so far)."""
            ps["n"] += 1
            ps["first" if ps["pass"] == 1 else "retried"] = ps.get("first" if ps["pass"] == 1 else "retried", 0) + 1
            ps["valid"] += bool(r["valid"])
            now = time.perf_counter()
            left = (now - ps["t0"]) / ps["n"] * max(0, ps["of"] - ps["n"])
            hms = lambda x: f"{int(x // 3600)}:{int(x % 3600 // 60):02d}:{int(x % 60):02d}"  # noqa: E731
            name = "pass 1" if ps["pass"] == 1 else f"retries (pass {ps['pass']})"
            print(f"  {name}: {ps['n']}/{ps['of']} valid {ps['valid']} | {r.get('latency') or 0:.0f} s | elapsed {hms(now - t0)} | "
                  f"pass left about {hms(left)}" + ("" if r["valid"] else f" | {(r.get('error') or r.get('finish_reason') or '')[:60]}"),
                  flush=True)
        with ap.open("a", encoding="utf-8") as fh:
            inode, size = os.fstat(fh.fileno()).st_ino, os.fstat(fh.fileno()).st_size

            def same_file() -> bool:
                """We hold the lock, so nobody else may write: the path must still name the file we append to, at the size
                we left it (a sync or a checkout that replaces or truncates answers.jsonl would make later answers go to
                a lost inode)."""
                try:
                    st = os.stat(ap)
                except FileNotFoundError:
                    return False
                return st.st_ino == inode and st.st_size == size

            def stop_lost_file() -> None:
                """The path no longer names the file appended to: save every answer of this session (held in memory) to
                answers.jsonl.orphan-<time>, say what the path holds now, stop at once. A resume asks again whatever the
                path lacks (answers written after a sync took its copy are in the orphan file, not in the path)."""
                try:
                    now_size = os.stat(ap).st_size
                except FileNotFoundError:
                    now_size = None
                orphan = ap.with_name(f"answers.jsonl.orphan-{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}")
                with open(orphan, "x", encoding="utf-8") as of:
                    for r in asked:
                        of.write(json.dumps(r, ensure_ascii=False) + "\n")
                    of.flush()
                    os.fsync(of.fileno())
                print(f"\n{ap} was replaced, truncated or removed during the run: stopped. The file appended to had {size} "
                      f"bytes; the path now has {'none (removed)' if now_size is None else f'{now_size} bytes'}. Answers "
                      f"written after the change happened (e.g. after a sync took its copy) may be missing from the path: all "
                      f"{len(asked)} answers of this session are saved in {orphan.name}. A resume with the same command asks "
                      f"again what the path lacks.", file=sys.stderr, flush=True)
                os._exit(3)  # without waiting for the calls in progress
            while todo:
                ex = cf.ThreadPoolExecutor(max_workers=a.concurrency)
                try:
                    futs = {ex.submit(ask, a, it, tries.get(it["item"], 0) + 1): it for it in todo}
                    again = []
                    for fut in cf.as_completed(futs):
                        it, r = futs[fut], fut.result()
                        if r is None:  # not started: the spend cap or a halt
                            capped = capped or not a.halt.is_set()
                            continue
                        if not same_file():
                            ex.shutdown(wait=False, cancel_futures=True)
                            asked.append(r)  # this answer was not written: it goes to the orphan file with the others
                            stop_lost_file()
                        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                        fh.flush()
                        os.fsync(fh.fileno())
                        size = os.fstat(fh.fileno()).st_size
                        asked.append(r)
                        progress(r)
                        if r["ok"]:
                            streak = 0
                            tries[it["item"]] = tries.get(it["item"], 0) + 1
                        else:
                            streak += 1
                            if r.get("fatal") and not a.halt.is_set():
                                a.halt_reason = f"{it['item']}: {r.get('error')} (not retried: a request or account error); fix it, then resume"
                                a.halt.set()  # no new call starts; the calls in flight are received and saved below
                            if streak >= a.stop_after_transport_errors and not a.halt.is_set():
                                a.halt_reason = f"{streak} transport errors in a row (server down?): stopped; resume later"
                                a.halt.set()
                        if not r["valid"] and tries.get(it["item"], 0) < a.max_attempts and not a.limit:
                            again.append(it)
                except KeyboardInterrupt:  # one Ctrl-C: stop at once; every answer so far is on disk
                    ex.shutdown(wait=False, cancel_futures=True)
                    print(f"\ninterrupted: {sum(x['valid'] for x in asked)} valid answers of this session are saved "
                          f"({len(asked)} asked); the answer in progress is asked again on resume; resume with the same command",
                          flush=True)
                    os._exit(130)  # without waiting for the call in progress
                ex.shutdown(wait=True)
                todo = [] if (capped or a.halt.is_set()) else again
                ps.update({"pass": ps["pass"] + 1, "n": 0, "of": len(todo), "valid": 0, "t0": time.perf_counter()})
            if not same_file():  # a change after the last write: never mark such a run finished
                stop_lost_file()
        rows = read_jsonl(ap)
        valid = {r["item"] for r in rows if r.get("valid")}
        meta = json.loads(rj.read_text())
        now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
        stop_reason = a.halt_reason or (f"spend cap ${a.max_usd:.2f}" if paid and capped else None)
        if stop_reason:  # a stopped session is never 'finished': readers that require a finished run refuse it
            meta["sessions"][-1].update(stopped_at=now, stopped_reason=stop_reason)
        else:
            meta["sessions"][-1]["finished_at"] = now
        rj.write_text(json.dumps(mask_home(meta), ensure_ascii=False, indent=1))
        if asked:
            lat = sorted(r["latency"] or 0 for r in asked)
            fr = {}
            for r in asked:
                fr[r.get("finish_reason") or r.get("error") or "?"] = fr.get(r.get("finish_reason") or r.get("error") or "?", 0) + 1
            print(f"asked {len(asked)} (pass 1: {ps.get('first', 0)}, retries: {ps.get('retried', 0)}): valid {sum(r['valid'] for r in asked)}, finish/error {fr}, latency median "
                  f"{lat[len(lat) // 2]:.1f}s max {lat[-1]:.1f}s, completion tokens max "
                  f"{max((r.get('completion_tokens') or 0) for r in asked)}")
            oc = {}
            for r in asked:
                oc[r.get("outcome") or "?"] = oc.get(r.get("outcome") or "?", 0) + 1
            extra = sum(max(0, (r.get("transport_attempts") or 1) - 1) for r in asked)
            gate = getattr(a, "gate", None)
            print(f"outcomes {oc}; transport retries {extra}" +
                  (f"; rate-limit waits {gate.waits} ({gate.waited_s:.0f} s)" if gate is not None else ""))
            meta = json.loads(rj.read_text())
            meta["sessions"][-1]["summary"] = {"asked": len(asked), "outcomes": oc, "transport_retries": extra,
                                               "rate_limit_waits": gate.waits if gate is not None else None,
                                               "rate_limit_waited_s": round(gate.waited_s, 1) if gate is not None else None}
            rj.write_text(json.dumps(mask_home(meta), ensure_ascii=False, indent=1))
        print(f"valid {len(valid)} of {len(items)} items; answers in {ap}")
        if paid:
            meta = json.loads(rj.read_text())
            meta["sessions"][-1]["spent_usd_at_end"] = round(a.spent, 6)
            rj.write_text(json.dumps(mask_home(meta), ensure_ascii=False, indent=1))
            print(f"spent ${a.spent:.4f} of the cap ${a.max_usd:.2f}")
        if a.halt_reason:
            sys.exit(a.halt_reason)
        if paid and capped:
            sys.exit(f"stopped at the spend cap ${a.max_usd:.2f}: the same command with a higher --max-usd continues")
        if len(valid) < len(items) and not a.limit:
            sys.exit(f"{len(items) - len(valid)} items without a valid answer")


def ratings_of(out: Path) -> tuple[dict, dict, dict]:
    """item -> item; item -> its valid answer's ratings; item -> 'invalid' reason for items without a valid answer."""
    items = {it["item"]: it for it in read_jsonl(out / "items.jsonl.gz")}
    ans, err = {}, {}
    for r in read_jsonl(out / "answers.jsonl"):
        if r.get("valid"):
            ans[r["item"]] = r["answer"]["ratings"]
        else:
            err[r["item"]] = r.get("error") or r.get("finish_reason") or "invalid"
    return items, ans, {k: v for k, v in err.items() if k not in ans}


def rating(ans: dict, k: str, function: str):
    """A valid rating (1-5), None ('brak podstaw'), or INVALID (no valid answer)."""
    return ans[k].get(function) if k in ans else bp.INVALID


def onehot(fields: dict) -> np.ndarray:
    cats = {"largest_b2b_client_share": l2.CONCENTRATION, "overdue_receivables_share": l2.OVERDUE,
            "overdue_payables_share": l2.OVERDUE, "liquidity_declared": l2.LIQUIDITY_CATS}
    v = []
    for k in COUPLED:
        v += [1.0 if fields.get(k) == c else 0.0 for c in cats[k][1:]]  # the first category is the reference
    return np.array(v)


def population_test(items: dict, ans: dict, function: str, variant: str, keep: list, class_share: dict) -> dict:
    """Variant minus independent per firm and replicate, minus the main effects of the coupled fields estimated within
    firms UNDER THE NULL (no variant effect: least squares of the differences on the differences of the category
    indicators, without intercept), mean over the firm's replicates; weighted mean over firms (inside a class by
    weight_active, classes by their share among the kept classes); sign-flip test over firms on these null-restricted
    residuals. Missing data reported: firms dropped (independent rating null or invalid) and nulls and invalid answers
    by replicate of the variant."""
    by_firm = {}
    for k, it in items.items():
        if it["kind"] == "population" and it["size"] in keep:
            by_firm.setdefault(it["firm_id"], {})[(it["dependence"], it["replicate"])] = (k, it)
    dropped = {"independent_null": 0, "independent_invalid": 0}
    missing = {str(r): {"null": 0, "invalid": 0} for r in pc.REPLICATES}
    dy, dx, fid, wt, sz = [], [], [], {}, {}
    for f, its in by_firm.items():
        k0, base = its[("independent", 0)]
        y0 = rating(ans, k0, function)
        if y0 is None or y0 == bp.INVALID:
            dropped["independent_null" if y0 is None else "independent_invalid"] += 1
            continue
        for r in pc.REPLICATES:
            k, it = its[(variant, r)]
            y = rating(ans, k, function)
            if y is None or y == bp.INVALID:
                missing[str(r)]["null" if y is None else "invalid"] += 1
                continue
            dy.append(y - y0)
            dx.append(onehot(it["fields"]) - onehot(base["fields"]))
            fid.append(f)
        wt[f], sz[f] = base["weight"], base["size"]
    out = {"function": function, "variant": variant, "keep": keep, "dropped_firms": dropped, "missing_by_replicate": missing}
    firms = sorted(set(fid))
    if not firms:
        return {**out, "n_firms": 0, "estimate": None, "p": 1.0}
    dy, dx = np.array(dy, float), np.array(dx)
    cls = sorted({sz[f] for f in firms})
    tot_w = {s: sum(wt[f] for f in firms if sz[f] == s) for s in cls}
    tot_share = sum(class_share[s] for s in cls)
    wf = {f: class_share[sz[f]] / tot_share * wt[f] / tot_w[sz[f]] for f in firms}
    w_obs = np.array([wf[f] for f in fid])
    beta0 = np.linalg.lstsq(dx * np.sqrt(w_obs)[:, None], dy * np.sqrt(w_obs), rcond=None)[0]
    resid = dy - dx @ beta0
    d = np.array([resid[[i for i, g in enumerate(fid) if g == f]].mean() for f in firms])
    wd = np.array([wf[f] for f in firms])
    return {**out, "n_firms": len(firms), "n_by_class": {s: sum(1 for f in firms if sz[f] == s) for s in cls},
            "class_weights": {s: class_share[s] / tot_share for s in cls},
            "estimate": float(np.average(d, weights=wd)), "p": bp.sign_flip_p(d, wd),
            "kish_n": float(wd.sum() ** 2 / (wd ** 2).sum())}


def descriptives(items: dict, ans: dict, err: dict, plan: dict) -> dict:
    """Every function and class, kept or not by the rules: ratings by version and by dependence, nulls, ratings at 1
    and 5, invalid answers, within-firm differences; and the technical errors."""
    out = {"pair": {}, "population": {}, "errors": {}}
    for k, e in err.items():
        out["errors"][e[:60]] = out["errors"].get(e[:60], 0) + 1
    ids = [f["id"] for f in plan["functions"]]

    def summary(vals):
        v = [x for x in vals if isinstance(x, int)]
        return {"n": len(vals), "valid": len(v), "null": sum(x is None for x in vals), "invalid": sum(x == bp.INVALID for x in vals),
                "mean": float(np.mean(v)) if v else None, "sd": float(np.std(v)) if v else None,
                "at_1": sum(x == 1 for x in v), "at_5": sum(x == 5 for x in v)}
    for fn in ids:
        for s in bp.SIZES:
            g = [(k, it) for k, it in items.items() if it["kind"] == "pair" and it["function"] == fn and it["size"] == s]
            by = {v: summary([rating(ans, k, fn) for k, it in g if it["version"] == v]) for v in bp.VERSIONS}
            bases = {}
            for k, it in g:
                bases.setdefault(it["firm_id"], {})[it["version"]] = rating(ans, k, fn)
            inter = [bp.interaction(b) for b in bases.values() if all(isinstance(b.get(v), int) for v in bp.VERSIONS)]
            out["pair"].setdefault(fn, {})[s] = {"by_version": by, "interaction_mean": float(np.mean(inter)) if inter else None,
                                                 "interaction_sd": float(np.std(inter)) if inter else None, "n_complete": len(inter)}
            pop = [(k, it) for k, it in items.items() if it["kind"] == "population" and it["size"] == s]
            by_dep = {d: summary([rating(ans, k, fn) for k, it in pop if it["dependence"] == d]) for d in ("independent", *VARIANTS)}
            diffs = {}
            for d in VARIANTS:
                x = []
                firms = sorted({it["firm_id"] for _, it in pop})  # a fixed order: the sums do not depend on the hash seed
                for f in firms:
                    y0 = rating(ans, f"population/{f}/independent/0", fn)
                    ys = [rating(ans, f"population/{f}/{d}/{r}", fn) for r in pc.REPLICATES]
                    ys = [y for y in ys if isinstance(y, int)]
                    if isinstance(y0, int) and ys:
                        x.append(float(np.mean(ys)) - y0)
                diffs[d] = {"n_firms": len(x), "mean": float(np.mean(x)) if x else None, "sd": float(np.std(x)) if x else None}
            out["population"].setdefault(fn, {})[s] = {"by_dependence": by_dep, "within_firm_difference": diffs}
    return out


def analyze_core(items: dict, ans: dict, err: dict, design: dict, plan: dict, with_descriptives: bool = True) -> dict:
    """Both families from items and answers (item -> ratings of a valid answer; items without one are missing): the
    analysis used by 'analyze' and by the simulation (scripts/b_k2_sim.py)."""
    pair = {}
    for fn in plan["functions"]:
        bases = {}
        for k, it in items.items():
            if it["kind"] == "pair" and it["function"] == fn["id"]:
                b = bases.setdefault(it["firm_id"], {"size": it["size"], "weight": it["weight"], "r": {}})
                b["r"][it["version"]] = rating(ans, k, fn["id"])
        bl = list(bases.values())
        pair[fn["id"]] = bp.pair_test([b["r"] for b in bl], [b["size"] for b in bl], design["class_share_pair"][fn["id"]],
                                      [b["weight"] for b in bl])
    pop, expl = {}, []
    for c in design["population_contrasts"]:
        t = population_test(items, ans, c["function"], c["variant"], c["keep"], design["class_share_population"])
        (pop.__setitem__(c["function"], t) if c["role"] == "population" else expl.append(t))
    holm = bp.holm({k: t["p"] for k, t in pop.items()})
    return {"pair_family": {"tests": pair, "decision": bp.pair_family(pair, plan)},
            "population_family": {"tests": pop, "holm_rejected": holm,
                                  "note": "classes weighted by their share of active firms among the kept classes (class_weights "
                                          "of each test): a test that keeps 0-9 is in effect about micro firms (about 97% of the "
                                          "weight); a test without 0-9 is about the classes it keeps, not about micro firms; the p "
                                          "values and Holm decisions are diagnostic until the calibration of the whole procedure is "
                                          "checked by simulation"},
            "exploratory": expl, "descriptives": descriptives(items, ans, err, plan) if with_descriptives else None,
            "items": len(items), "items_without_valid_answer": len(items) - len(ans),
            "claims": plan["claims"], "note": "a measurement of agents' behaviour on synthetic profiles, not an estimate of Polish firms' needs"}


def analyze(a) -> None:
    import fcntl
    out = Path(a.out)
    if json.loads((out / "design.json").read_text()).get("parser", "b_k2") != "b_k2":
        sys.exit("not a B-K2 build: analyse it with its own script (e.g. safe_analyze.py)")
    with open(out / ".lock", "a") as lk:
        try:
            fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            sys.exit("a run is in progress in this directory: analyse after it")
    plan = bp.load_plan()
    design = json.loads((out / "design.json").read_text())
    changed = {k for k, v in current_files().items() if design["files_sha256"].get(k) != v} & {"plan", "key", "prompt_script"}
    if changed:
        sys.exit(f"the plan, key or prompt changed since build: {sorted(changed)}; analyse with the files of the build")
    if sha(gzip.open(out / "items.jsonl.gz").read()) != design["items_sha256"]:
        sys.exit("items.jsonl.gz does not match design.json")
    items, ans, err = ratings_of(out)
    res = analyze_core(items, ans, err, design, plan)
    pair, pop, holm = res["pair_family"]["tests"], res["population_family"]["tests"], res["population_family"]["holm_rejected"]
    (out / "analysis.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"items {len(items)}, without a valid answer {len(items) - len(ans)}")
    for k, d in res["pair_family"]["decision"].items():
        print(f"  pair       {k:24s} success {d['success']} restricted {d['success_restricted_to']}")
    for k, t in pop.items():
        print(f"  population {k:24s} {t['variant']:15s} estimate {t['estimate']} p {t['p']:.4f} holm {holm[k]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["build", "run", "analyze", "plan-sizes"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261114)
    ap.add_argument("--pair-per-class", type=int, default=10)
    ap.add_argument("--pop-per-class", type=int, default=40)
    ap.add_argument("--firms-limit", type=int, default=None, help=f"tests only ({TEST_ENV}=1): the first N firms of the sample")
    ap.add_argument("--draws", type=int, default=20, help="plan-sizes: draws of the case set per size")
    ap.add_argument("--grid", type=int, nargs="+", default=[40, 80, 150, 250], help="plan-sizes: case-set sizes per class")
    ap.add_argument("--api", choices=["ollama", "openai", "anthropic", "openai-responses"], default="ollama")
    ap.add_argument("--model")
    ap.add_argument("--base-url", default=None, help="default: Ollama's http://127.0.0.1:11434; for --api anthropic and "
                    "openai-responses the SDK's own")
    ap.add_argument("--effort", choices=EFFORTS["anthropic"] + tuple(e for e in EFFORTS["openai-responses"] if e not in EFFORTS["anthropic"]),
                    help="--api anthropic (output_config.effort: low, medium, high) and openai-responses (reasoning.effort: "
                         "none, low, medium, high, xhigh, max); required there")
    ap.add_argument("--max-usd", type=float, help="--api anthropic and openai-responses: spend cap in USD, cumulative over resumed sessions (required)")
    ap.add_argument("--pricing", default=None, help="price file; default data/reference/anthropic_pricing_2026-11-15.json "
                    "(anthropic) or data/reference/openai_pricing_2026-11-17.json (openai-responses)")
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--num-ctx", type=int, default=32768)
    ap.add_argument("--think", default="false")
    ap.add_argument("--chat-template-kwargs", help="OpenAI-compatible path only, required there: JSON sent with every request")
    ap.add_argument("--stage-a-report", help="a b_k2_input_identity report this run must match (checked before the first call)")
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--max-attempts", type=int, default=2,
                    help="answers of the model per item (an invalid answer is asked again in a fresh conversation; at "
                         "temperature 0 it is usually repeated, so this mainly measures invalid answers)")
    ap.add_argument("--transport-retries", type=int, default=3, help="retries of a call without an answer (server error, timeout)")
    ap.add_argument("--stop-after-transport-errors", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None, help="ask only the first N items of the random order (a check of format, cuts and time)")
    a = ap.parse_args()
    if a.pricing is None:
        a.pricing = str(PRICING.get(a.api, PRICING["anthropic"]))
    if a.command == "run" and a.api in PAID and a.max_usd is None:
        sys.exit(f"--api {a.api} needs --max-usd (a spend cap, cumulative over resumed sessions)")
    if a.command == "run" and not a.model:
        sys.exit("--model is required for run")
    a.ctk = None
    a.base_url_given = a.base_url is not None
    if a.base_url is None:
        a.base_url = "http://127.0.0.1:11434"
    if a.command == "run" and a.api in PAID:
        if a.effort is None:
            sys.exit(f"--api {a.api} needs --effort given explicitly ({', '.join(EFFORTS[a.api])}"
                     + (": thinking is disabled)" if a.api == "anthropic" else ": the reasoning effort)"))
        if a.effort not in EFFORTS[a.api]:
            sys.exit(f"--effort {a.effort} is not an effort of --api {a.api} ({', '.join(EFFORTS[a.api])})")
        if a.chat_template_kwargs is not None:
            sys.exit("--chat-template-kwargs is for the OpenAI-compatible path")
    elif a.command == "run" and a.effort is not None:
        sys.exit("--effort is for --api anthropic and openai-responses")
    if a.command == "run":
        if a.api == "ollama" and a.chat_template_kwargs is not None:
            sys.exit("--chat-template-kwargs is for the OpenAI-compatible path; Ollama takes --think")
        if a.api == "openai":
            if a.chat_template_kwargs is None:
                sys.exit("--api openai needs --chat-template-kwargs given explicitly (e.g. '{\"enable_thinking\": false}'; "
                         "'{}' leaves the server's default on purpose)")
            a.ctk = json.loads(a.chat_template_kwargs)
            t = str(a.think).lower()
            if t in ("false", "true") and a.ctk.get("enable_thinking") is not (t == "true"):
                sys.exit(f"--think {a.think} and --chat-template-kwargs {a.ctk} do not match")
    {"build": build, "run": run, "analyze": analyze, "plan-sizes": plan_sizes}[a.command](a)


if __name__ == "__main__":
    main()
