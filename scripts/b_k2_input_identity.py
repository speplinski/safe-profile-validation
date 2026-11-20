#!/usr/bin/env python3
"""Stage A of the B-K2 serving protocol (data/reference/b_k2_serving_protocol_draft.json): is the INPUT the model gets
the same token sequence on two servers? Weights and the model's behaviour are separate stages.

    python3 scripts/b_k2_input_identity.py ollama --run <B-K2 build dir> \\
        --runner-url auto --out <file.json>     # or http://127.0.0.1:<port of the MLX runner>
    python3 scripts/b_k2_input_identity.py vllm --run <B-K2 build dir> --reference <the Ollama report> \\
        --base-url http://127.0.0.1:8000 --model <served name> [--server-log <its startup log>] [--generation-tie] \\
        --out <file.json>
    python3 scripts/b_k2_input_identity.py compare --run <B-K2 build dir> <a.json> <b.json>

vllm (vLLM 0.33.0 with the vLLM-Metal plugin; read in vLLM's source at the tag v0.33.0rc3, the commit of v0.33.0):
the real chat request (llm_serving_check.openai_chat_body: the B-K2 messages, max_tokens and seed of the build's run,
temperature 0, chat_template_kwargs, default {"enable_thinking": false}) is sent to POST /v1/chat/completions/render.
That endpoint (registered with `vllm serve ... --enable-scale-out`, or by the CPU-only `vllm launch render`) runs
OnlineRenderer.render_chat, the function the chat route itself calls (OpenAIServingChat.render_chat_request), so the
template, the thinking switch and the start of the assistant turn are those of the real request path, and it returns
the token ids prepared for the engine. With --generation-tie (only on a server with the model loaded) the same request
goes to /v1/chat/completions with max_tokens 1 and return_token_ids; its prompt_token_ids (and usage count) must equal
the rendered ids. They are what vLLM's API server handed to the engine for that request, rendered again by the chat
route: not observed inside the plugin's runner. Render and generation tie are kept apart: identical render ids pass
stage A's input check and say nothing about the weights (whether the plugin loads the NVFP4 weights with every scale is
a stage of its own).
Since 0.7 the server is identified from its process, not from arguments: the process listening on --base-url's port
must be `<python> .../vllm serve <model> ... --enable-scale-out`; its interpreter is the environment whose package
versions are read (vllm, vllm-metal, transformers, tokenizers, mlx, mlx-lm, with direct_url and the SHA-256 of each
package's RECORD, which fingerprints an installed wheel); /version must equal the vllm read there; the flags that
shape the rendering (--chat-template, --default-chat-template-kwargs, --tokenizer, --tokenizer-mode, --max-model-len,
--reasoning-parser and others) are recorded from the command line; the tokenizer and template files are hashed in the
model directory of that command line (--model-dir and --python, if given, must be these); the plugin must be seen
active (a vllm_metal file open in the server's process tree, or lines of --server-log); max_model_len from /v1/models
must hold the longest prompt plus max_tokens; the thinking switch must match the run's (false/true). The cards are
exactly the reference report's 20 (which must equal select(build, 20)). HTTP errors stop with a message.

ollama: for each selected card, the real /api/chat request (llm_serving_check.ollama_chat_body, the same body the runner
sends, with the run's think, num_ctx, max_tokens and seed) plus _debug_render_only gives the rendered prompt; the MLX
runner's own POST /v1/tokenize tokenizes that text with the call the generation path makes (Ollama 0.40.15:
Tokenizer.Encode(prompt, Tokenizer.AddBOS()), mlxrunner/pipeline.go:51 and server.go:206; the chat handler and the client
pass the rendered string unchanged; the client posts to /completion, which the runner redirects to /v1/completions). The
runner listens on its own local port, which changes when the model is reloaded (find it on the Mac with
`lsof -nP -iTCP -sTCP:LISTEN | grep -i ollama`; the main server's port 11434 is not the runner). Every card is also sent
as the real request with num_predict 1, and prompt_eval_count must equal the sequence's length: the tie of --runner-url
to the model actually generating (necessary, not a proof by itself). Only an MLX model (format safetensors) is accepted:
a GGUF model takes another code path. The request settings (model, think, num_ctx, max_tokens, seed) come from the
build's run.json when there is one (a different value given on the command line stops the script), and items.jsonl.gz
must match design.json.

Since 0.3 the runner is tied to the server by the operating system, not by a count: before the first card and after the
last, the process listening on --runner-url's port must be `ollama runner ... --model <model> ... --port <port>` and its
parent must be the process listening on --base-url's port (lsof and ps; the same runner process at both checks, so no
reload in between), and it must be the server's only child runner of that model (ps -A). The equal count stays an extra
check: a runner of another server or model with the same count would otherwise pass (a case reproduced by the project
lead). What this proves: the tokenizing process is the only runner of the model under the server before and after the
cards; that generation used it is inferred from Ollama's scheduler and backed by the equal prompt_eval_count, not
observed per request. Load the model with the run's settings first (num_ctx above all: a runner loaded with another
num_ctx is reloaded by the first real request, on another port and pid, and the script stops).

Recorded per card: SHA-256 of the rendered prompt (UTF-8), the full token id sequence, its length and its SHA-256 over
the canonical JSON array (json.dumps(ids, separators=(",", ":")), no spaces); the Ollama version, the model name and
digest (ollama_provenance.api_facts). Cards: deterministic, covering every (role in the system prompt, item kind)
combination of the build, then as many function orders as possible, plus the longest and the shortest prompt.

The output is written also when a check fails, with passed false. compare passes (exit 0) only if each file is
consistent when recomputed from its cards (no repeated card id; every sequence matching its length and SHA-256 and
its card's prompt_eval_count; every role x kind of the build covered; a verified runner process before and after; the
stored flag passed true), both files have the same build and request settings, the same non-empty set of cards, and
every sequence SHA-256 is identical; differences are listed. The stored flags are never trusted on their own. Two
results are kept apart: sequences_identical (the cards both reports hold give the same sequences) and stage_a_passed
(that, and both reports hold exactly the protocol's card set: select(build, 20) recomputed from the build). Exit 0 only
when stage A passed; a smaller or other card set can show identical sequences but never passes the stage. Other servers are added once their code path from rendering to the tokens prepared for generation has been
read, as for Ollama (their own tokenization of their own rendering, never a re-tokenization by us).
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import os
import hashlib
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from urllib.parse import urlparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from llm_serving_check import ollama_chat_body, openai_chat_body  # noqa: E402
from ollama_provenance import api_facts, mask_home  # noqa: E402

VERSION = "0.7"
STAGE_A_CARDS = 20  # the protocol's card count (data/reference/b_k2_serving_protocol_draft.json, stage_A_identity)
ROLE = re.compile(r"^Jesteś osobą z firmy opisanej na karcie firmy: ([^.\n]*)\.")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def seq_sha(ids: list) -> str:
    """The SHA-256 of a token id sequence, always over the same text: a JSON array without spaces."""
    if not all(isinstance(i, int) and not isinstance(i, bool) for i in ids):
        raise ValueError("token ids must be integers")
    return sha(json.dumps(ids, separators=(",", ":")).encode())


def role(it: dict) -> str:
    m = ROLE.match(it["messages"][0]["content"])
    return m.group(1) if m else "?"


def features(it: dict) -> set:
    """What a card can cover: its role x kind (the template variants), its function order, and for a pair item its
    function and version (LL, HL, LH, HH), for a population item its dependence variant."""
    f = {("order", tuple(it["order"]))}
    if it["kind"] == "pair":
        f |= {("function", it.get("function")), ("version", it.get("version"))}
    else:
        f.add(("dependence", it.get("dependence")))
    return f


def select(items: list, n: int) -> list:
    """Deterministic: one card per (role, kind) combination, the longest and the shortest prompt, then greedily the card
    that covers most features not covered yet (ties: the first in sorted item-id order), up to n (at least the coverage
    set). A greedy pick, not the maximum coverage possible. Inside a (role, kind), the first card is also the one
    covering most features."""
    items = sorted(items, key=lambda it: it["item"])
    rank = {it["item"]: i for i, it in enumerate(items)}
    size = lambda it: sum(len(m["content"]) for m in it["messages"])  # noqa: E731
    out, seen, covered = [], set(), set()

    def take(it):
        if it["item"] not in seen:
            out.append(it)
            seen.add(it["item"])
            covered.update(features(it))

    def best(pool):
        return max(pool, key=lambda it: (len(features(it) - covered), -rank[it["item"]]))
    for key in sorted({(role(it), it["kind"]) for it in items}):
        take(best([it for it in items if (role(it), it["kind"]) == key]))
    take(max(items, key=size))
    take(min(items, key=size))
    while len(out) < n and len(out) < len(items):
        take(best([it for it in items if it["item"] not in seen]))
    return out


def listener(port: int) -> int | None:
    """The pid listening on a local TCP port (lsof)."""
    r = subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"], capture_output=True, text=True)
    pids = sorted({int(x) for x in r.stdout.split()})
    return pids[0] if len(pids) == 1 else None


def process(pid: int) -> tuple[int | None, str]:
    """ppid and the whole command line (-ww: never cut to the terminal's width)."""
    r = subprocess.run(["ps", "-ww", "-o", "ppid=", "-o", "command=", "-p", str(pid)], capture_output=True, text=True)
    m = re.match(r"\s*(\d+)\s+(.*)", r.stdout.strip())
    return (int(m.group(1)), m.group(2)) if m else (None, "")


def children(ppid: int) -> list:
    """(pid, command) of every process whose parent is ppid (-A -ww)."""
    r = subprocess.run(["ps", "-A", "-ww", "-o", "pid=", "-o", "ppid=", "-o", "command="], capture_output=True, text=True)
    out = []
    for ln in r.stdout.splitlines():
        m = re.match(r"\s*(\d+)\s+(\d+)\s+(.*)", ln)
        if m and int(m.group(2)) == ppid:
            out.append((int(m.group(1)), m.group(3)))
    return out


def runner_args(cmd: str) -> dict | None:
    """--model and --port of an `ollama runner` command line, None if it is not one."""
    args = cmd.split()
    if not any(re.search(r"(^|/)ollama$", x) and i + 1 < len(args) and args[i + 1] == "runner" for i, x in enumerate(args)):
        return None
    opt = lambda k: args[args.index(k) + 1] if k in args and args.index(k) + 1 < len(args) else None  # noqa: E731
    return {"model": opt("--model"), "port": opt("--port")}


def pid_ok(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x > 1


def facts_ok(f: dict, model: str, port) -> bool:
    """The recorded facts of one runner check, recomputed (used by the run and by compare alike)."""
    ra = runner_args(f.get("runner_command") or "")
    same = [c for c in f.get("runners_of_model_under_server") or []]
    return bool(isinstance(model, str) and model and f.get("local") is True and pid_ok(f.get("runner_pid")) and pid_ok(f.get("server_pid"))
                and f.get("runner_ppid") == f.get("server_pid") and ra and ra["model"] == model and ra["port"] == str(port)
                and str(f.get("runner_port")) == str(port) and same == [f.get("runner_pid")])


def discover_runner(base_url: str, model: str) -> str:
    """--runner-url auto: the address of the server's only child runner of the model (the checks run all the same)."""
    b = urlparse(base_url)
    sp = listener(b.port) if b.port else None
    found = [runner_args(c) for _, c in children(sp)] if sp else []
    ports = sorted({f["port"] for f in found if f and f["model"] == model and f["port"]})
    if len(ports) != 1:
        sys.exit(f"--runner-url auto: {len(ports)} runners of {model!r} under the server on port {b.port} (load the model "
                 f"first with the run's settings)")
    return f"http://127.0.0.1:{ports[0]}"


def runner_tie(base_url: str, runner_url: str, model: str) -> dict:
    """The process on the runner's port is an Ollama runner of this model on this port, a child of the process
    serving base_url (both local), and the server's only child runner of this model."""
    b, r = urlparse(base_url), urlparse(runner_url)
    local = {b.hostname, r.hostname} <= {"127.0.0.1", "localhost", "::1"}
    rp = listener(r.port) if local and r.port else None
    sp = listener(b.port) if local and b.port else None
    ppid, cmd = process(rp) if rp else (None, "")
    same = sorted(pid for pid, c in children(sp) if (runner_args(c) or {}).get("model") == model) if sp else []
    f = {"local": local, "runner_port": r.port, "runner_pid": rp, "runner_ppid": ppid, "server_pid": sp,
         "runner_command": cmd, "runners_of_model_under_server": same}
    return {"ok": facts_ok(f, model, r.port), **f}


def post(url: str, body, timeout: float, text: bool = False):
    data = body.encode() if text else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "text/plain" if text else "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


SETTINGS = {"model": None, "think": "false", "num_ctx": 32768, "max_tokens": 2000, "seed": 20261114}


def settings(a, run: Path) -> dict:
    """The request settings of the build's run (run.json config) or, without one, the command line or the runner's
    defaults; a command-line value that differs from run.json stops the script."""
    cfg = json.loads((run / "run.json").read_text())["config"] if (run / "run.json").exists() else {}
    out = {}
    for k, default in SETTINGS.items():
        given = getattr(a, k)
        if k in cfg and given is not None and given != cfg[k]:
            sys.exit(f"--{k.replace('_', '-')} {given!r} differs from the build's run.json ({cfg[k]!r})")
        out[k] = cfg.get(k, given if given is not None else default)
    if not out["model"]:
        sys.exit("--model is needed (the build has no run.json)")
    return {**out, "from_run_json": bool(cfg)}


def ollama(a) -> None:
    out = Path(a.out)
    if out.exists() and not a.force:
        sys.exit(f"{out} exists: --force to overwrite")
    run = Path(a.run)
    data = (run / "items.jsonl.gz").read_bytes()
    design = json.loads((run / "design.json").read_text())
    if sha(gzip.decompress(data)) != design["items_sha256"]:
        sys.exit("items.jsonl.gz does not match design.json")
    st = settings(a, run)
    items = [json.loads(x) for x in gzip.decompress(data).decode("utf-8").splitlines() if x.strip()]
    unknown = sorted({it["item"] for it in items if role(it) == "?"})[:3]
    if unknown:
        sys.exit(f"system prompts without the expected first sentence (role unknown), e.g. {unknown}")
    chosen = select(items, a.n)
    root = a.base_url.rstrip("/")
    facts = api_facts(root, st["model"])
    if not facts.get("model_digest"):
        sys.exit(f"no model {st['model']!r} at the server: {facts.get('tags_error') or 'not in /api/tags'}")
    fmt = ((facts.get("show") or {}).get("details") or {}).get("format")
    if fmt != "safetensors":
        sys.exit(f"the model's format is {fmt!r}: only an MLX model (safetensors) follows the code path this check relies on")
    if a.runner_url == "auto":
        a.runner_url = discover_runner(root, st["model"])
        print(f"runner: {a.runner_url}", flush=True)
    tie0 = runner_tie(root, a.runner_url, st["model"])
    if not tie0["ok"]:
        sys.exit(f"--runner-url is not the runner of {st['model']!r} under the server at {root}: {tie0}")
    cards = []
    for it in chosen:
        body = ollama_chat_body(st["model"], it["messages"], st["max_tokens"], st["seed"], st["think"], st["num_ctx"])
        rendered = post(root + "/api/chat", {**body, "_debug_render_only": True}, a.timeout).get("_debug_info", {}).get("rendered_template")
        if not isinstance(rendered, str) or not rendered:
            sys.exit(f"{it['item']}: no rendered prompt in the response (is _debug_render_only supported by this version?)")
        try:
            ids = post(a.runner_url.rstrip("/") + "/v1/tokenize", rendered, a.timeout, text=True)
        except OSError as e:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(mask_home({"script": f"b_k2_input_identity {VERSION}", "server": "ollama", "passed": False,
                                                  "stopped_at": it["item"], "error": type(e).__name__, "runner_url": a.runner_url,
                                                  "runner_process": {"before": tie0}, "cards": cards}), ensure_ascii=False, indent=1))
            sys.exit(f"{it['item']}: the runner at {a.runner_url} does not answer ({type(e).__name__}): reloaded? Load the model "
                     f"with the run's settings (num_ctx {st['num_ctx']}) and find the runner's port again; report written with passed false")
        if not isinstance(ids, list):
            sys.exit(f"{it['item']}: the runner's /v1/tokenize did not return a list")
        card = {"item": it["item"], "kind": it["kind"], "role": role(it), "order": it["order"],
                "rendered_sha256": sha(rendered.encode("utf-8")), "rendered_chars": len(rendered),
                "token_ids": ids, "token_count": len(ids), "token_ids_sha256": seq_sha(ids)}
        g = post(root + "/api/chat", {**body, "options": {**body["options"], "num_predict": 1}}, a.timeout)
        card["prompt_eval_count"] = g.get("prompt_eval_count")
        card["prompt_eval_count_matches"] = g.get("prompt_eval_count") == len(ids)
        cards.append(card)
        print(f"{it['item']}: {len(ids)} tokens {card['token_ids_sha256'][:12]} prompt_eval_count {card['prompt_eval_count']}", flush=True)
    cover = {"role_kind": sorted({f"{c['role']} | {c['kind']}" for c in cards}),
             "role_kind_in_build": sorted({f"{role(it)} | {it['kind']}" for it in items}),
             "orders": len({tuple(c["order"]) for c in cards}), "orders_in_build": len({tuple(it["order"]) for it in items})}
    tie1 = runner_tie(root, a.runner_url, st["model"])
    same_runner = tie1["ok"] and tie1["runner_pid"] == tie0["runner_pid"]
    tie = bool(cards) and all(c["prompt_eval_count_matches"] for c in cards)
    covered = cover["role_kind"] == cover["role_kind_in_build"]
    res = {"script": f"b_k2_input_identity {VERSION}", "server": "ollama", "date": dt.date.today().isoformat(),
           "ollama_version": facts.get("ollama_version"), "model": st["model"], "model_digest": facts.get("model_digest"),
           "model_format": fmt, "modelfile_weights_digest": facts.get("modelfile_weights_digest"),
           "template_sha256": (facts.get("show") or {}).get("template_sha256"),
           "request": {k: st[k] for k in ("think", "num_ctx", "max_tokens", "seed")}, "settings_from_run_json": st["from_run_json"],
           "build": run.name, "items_sha256": design["items_sha256"],
           "sequence_hash_format": "SHA-256 of json.dumps(ids, separators=(',', ':')) in UTF-8",
           "runner_process": {"before": tie0, "after": tie1, "same_runner": same_runner,
                              "what_it_proves": "the tokenizing process is the only runner of the model under the server before "
                                                "and after the cards; that generation used it is inferred from Ollama's "
                                                "scheduler and backed by the equal prompt_eval_count, not observed per request"},
           "coverage": cover, "cards": cards, "generation_tie": tie, "role_kind_covered": covered,
           "n": a.n, "protocol_card_set": a.n == STAGE_A_CARDS,
           "passed": tie and covered and same_runner and a.n == STAGE_A_CARDS}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(mask_home(res), ensure_ascii=False, indent=1))
    print(f"{len(cards)} cards, coverage {covered} (role x kind), orders {cover['orders']}/{cover['orders_in_build']}, "
          f"generation tie {tie}; passed {res['passed']}; written {out}")
    if not same_runner:
        sys.exit(f"the runner changed or is no longer tied to the server after the cards: {tie1}")
    if not tie:
        sys.exit("prompt_eval_count differs from the runner's sequence on some card (another model's runner or a stale port?)")
    if not covered:
        sys.exit("the cards do not cover every role x kind of the build")


VERSIONS_CODE = r"""
import hashlib, json, importlib.metadata as m, sys
out = {"_sys_prefix": sys.prefix}
for name in ("vllm", "vllm-metal", "transformers", "tokenizers", "mlx", "mlx-lm", "mlx-metal", "torch"):
    try:
        d = m.distribution(name)
        url = d.read_text("direct_url.json")
        rec = d.read_text("RECORD")
        out[name] = {"version": d.version, "direct_url": json.loads(url) if url else None,
                     "record_sha256": hashlib.sha256(rec.encode()).hexdigest() if rec else None}
    except m.PackageNotFoundError:
        out[name] = None
print(json.dumps(out))
"""
REQUIRED_VERSIONS = ("vllm", "vllm-metal", "transformers", "tokenizers", "mlx", "mlx-lm")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
                   "chat_template.json", "generation_config.json", "config.json", "preprocessor_config.json")
# server flags that shape the rendering or the prompt's room; recorded from the server's own command line
RENDER_FLAGS = ("--chat-template", "--chat-template-content-format", "--default-chat-template-kwargs", "--tokenizer",
                "--tokenizer-mode", "--max-model-len", "--reasoning-parser", "--served-model-name", "--generation-config",
                "--hf-overrides", "--trust-remote-code", "--revision", "--tokenizer-revision", "--enable-scale-out",
                "--speculative-config", "--quantization")


def env_versions(python: str) -> dict:
    r = subprocess.run([python, "-c", VERSIONS_CODE], capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        sys.exit(f"{python}: could not read the package versions: {r.stderr.strip()[:300]}")
    return json.loads(r.stdout)


def get(url: str, timeout: float):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def call(url: str, body, timeout: float, what: str):
    """post() with a message instead of a traceback on an HTTP or connection error."""
    try:
        return post(url, body, timeout)
    except urllib.error.HTTPError as e:
        detail = e.read()[:300].decode("utf-8", "replace")
        hint = " (is the server started with --enable-scale-out?)" if e.code in (404, 405) and "render" in url else ""
        sys.exit(f"{what}: HTTP {e.code} from {url}{hint}: {detail}")
    except OSError as e:
        sys.exit(f"{what}: no answer from {url} ({type(e).__name__})")


def descendants(pid: int) -> list:
    r = subprocess.run(["ps", "-A", "-ww", "-o", "pid=", "-o", "ppid="], capture_output=True, text=True)
    kids = {}
    for ln in r.stdout.splitlines():
        p = ln.split()
        if len(p) == 2 and p[0].isdigit() and p[1].isdigit():
            kids.setdefault(int(p[1]), []).append(int(p[0]))
    out, todo = [], [pid]
    while todo:
        q = todo.pop()
        out.append(q)
        todo.extend(kids.get(q, []))
    return out


def open_files(pids: list) -> list:
    r = subprocess.run(["lsof", "-n", "-P", "-Fn", "-p", ",".join(map(str, pids))], capture_output=True, text=True)
    return sorted({ln[1:] for ln in r.stdout.splitlines() if ln.startswith("n")})


def parse_vllm_command(cmd: str) -> dict:
    """The interpreter, the model path and the render flags of `<python> .../vllm serve <model> ...`."""
    t = cmd.split()
    python = t[0] if t and re.search(r"(^|/)python[0-9.]*$", t[0]) else None
    i = next((k for k, x in enumerate(t) if re.search(r"(^|/)vllm$", x)), None)
    serve = i is not None and i + 1 < len(t) and t[i + 1] == "serve"
    rest = t[i + 2:] if serve else []
    model = rest[0] if rest and not rest[0].startswith("-") else None
    flags = {}
    for k, x in enumerate(rest):
        name = x.split("=", 1)[0]
        if name == "--model" and model is None:
            model = x.split("=", 1)[1] if "=" in x else (rest[k + 1] if k + 1 < len(rest) else None)
        if name in RENDER_FLAGS:
            flags[name] = x.split("=", 1)[1] if "=" in x else (
                rest[k + 1] if k + 1 < len(rest) and not rest[k + 1].startswith("--") else True)
    return {"python": python, "serve": serve, "model_path": model, "flags": flags}


def plugin_evidence(pids: list, log: str | None) -> dict:
    files = [f for f in open_files(pids) if "vllm_metal" in f]
    lines = []
    if log:
        lines = [ln.strip()[:300] for ln in Path(log).read_text(errors="replace").splitlines()
                 if re.search(r"vllm_metal|platform plugin|MetalPlatform|metal platform", ln, re.I)][:20]
    return {"open_files_of_the_server_tree": files[:20], "server_log_lines": lines,
            "server_log_sha256": sha(Path(log).read_bytes()) if log else None, "found": bool(files or lines)}


def vllm_body(model: str, messages: list, st: dict, kwargs: dict) -> dict:
    """The request of a vLLM B-K2 run (run_b_k2 --api openai sends this same body through llm_serving_check.chat)."""
    return openai_chat_body(model, messages, st["max_tokens"], st["seed"], chat_template_kwargs=kwargs)


def thinking_matches(think, kwargs: dict) -> bool:
    t = str(think).lower()
    return (t == "false" and kwargs.get("enable_thinking") is False) or (t == "true" and kwargs.get("enable_thinking") is True)


def vllm(a) -> None:
    out = Path(a.out)
    if out.exists() and not a.force:
        sys.exit(f"{out} exists: --force to overwrite")
    run = Path(a.run)
    items, rk, items_sha = load_build(run)
    ref = json.loads(Path(a.reference).read_text())
    probs = check_report(ref, items, rk)
    if probs or ref.get("items_sha256") != items_sha:
        sys.exit(f"the reference report does not pass on this build: {probs or 'another build'}")
    ids = [c["item"] for c in ref["cards"]]
    if set(ids) != {it["item"] for it in select(list(items.values()), STAGE_A_CARDS)}:
        sys.exit("the reference report's cards are not the protocol's set")
    served = a.model
    if (run / "run.json").exists():
        a.model = None  # the served name may differ from the Ollama tag in run.json
    st = settings(a, run)
    served = served or st["model"]
    kwargs = json.loads(a.chat_template_kwargs)
    if not thinking_matches(st["think"], kwargs):
        sys.exit(f"the run's thinking (think {st['think']!r}) and --chat-template-kwargs {kwargs} do not match")
    root = a.base_url.rstrip("/")
    port = urlparse(root).port
    spid = listener(port) if port else None
    if not spid:
        sys.exit(f"no single process listens on port {port}")
    ppid, cmd = process(spid)
    pc = parse_vllm_command(cmd)
    if not (pc["serve"] and pc["python"]):
        sys.exit(f"the process on port {port} is not `<python> .../vllm serve ...`: {cmd[:300]}")
    if a.python and os.path.abspath(a.python) != os.path.abspath(pc["python"]):
        sys.exit(f"--python {a.python} is not the server's interpreter {pc['python']}")
    if "--enable-scale-out" not in pc["flags"]:
        sys.exit("the server was not started with --enable-scale-out (the render route is not registered)")
    if a.model_dir and os.path.abspath(a.model_dir) != os.path.abspath(pc["model_path"] or ""):
        sys.exit(f"--model-dir {a.model_dir} is not the directory the server serves ({pc['model_path']})")
    versions = env_versions(pc["python"])
    missing = [k for k in REQUIRED_VERSIONS if not versions.get(k)]
    if missing:
        sys.exit(f"{pc['python']}: not installed there: {missing}")
    try:
        server_version = get(root + "/version", a.timeout).get("version")
        models = get(root + "/v1/models", a.timeout).get("data") or []
    except OSError as e:
        sys.exit(f"no vLLM API at {root} ({type(e).__name__})")
    if server_version != versions["vllm"]["version"]:
        sys.exit(f"the server reports vLLM {server_version!r}, its environment holds {versions['vllm']['version']!r}")
    entry = next((m for m in models if m.get("id") == served), None)
    if entry is None:
        sys.exit(f"the server does not serve {served!r}: {[m.get('id') for m in models]}")
    plugin = plugin_evidence(descendants(spid), a.server_log)
    if not plugin["found"]:
        sys.exit("no evidence that the vllm-metal plugin is active in the server (no vllm_metal file open in its process "
                 "tree; pass --server-log with the server's startup log)")
    mdir = Path(pc["model_path"]) if pc["model_path"] and Path(pc["model_path"]).is_dir() else None
    files = {f: (sha((mdir / f).read_bytes()) if (mdir / f).exists() else None) for f in TOKENIZER_FILES} if mdir else {}
    ct = pc["flags"].get("--chat-template")
    if isinstance(ct, str) and Path(ct).is_file():
        files["--chat-template " + Path(ct).name] = sha(Path(ct).read_bytes())
    server = {"pid": spid, "ppid": ppid, "command": cmd, "python": pc["python"], "model_path": pc["model_path"],
              "render_flags": pc["flags"], "max_model_len": entry.get("max_model_len"), "plugin": plugin,
              "files_sha256_from": "the directory in the server's command line" if mdir else None, "files_sha256": files}
    cards = []
    for k in ids:
        it = items[k]
        body = vllm_body(served, it["messages"], st, kwargs)
        r = call(root + "/v1/chat/completions/render", body, a.timeout, k)
        tid = r.get("token_ids") if isinstance(r, dict) else None
        if not isinstance(tid, list) or not tid:
            sys.exit(f"{k}: no token_ids from /v1/chat/completions/render: {str(r)[:300]}")
        card = {"item": k, "kind": it["kind"], "role": role(it), "order": it["order"], "rendered_sha256": None,
                "request_sha256": sha(json.dumps(body, ensure_ascii=False, sort_keys=True).encode()),
                "token_ids": tid, "token_count": len(tid), "token_ids_sha256": seq_sha(tid)}
        if a.generation_tie:
            g = call(root + "/v1/chat/completions", {**body, "max_tokens": 1, "return_token_ids": True}, a.timeout, k)
            gp = g.get("prompt_token_ids")
            card["generation_prompt_token_ids_sha256"] = seq_sha(gp) if isinstance(gp, list) and gp else None
            card["generation_prompt_tokens"] = (g.get("usage") or {}).get("prompt_tokens")
        cards.append(card)
        print(f"{k}: {len(tid)} tokens {card['token_ids_sha256'][:12]}"
              + (f" generation {'same' if card['generation_prompt_token_ids_sha256'] == card['token_ids_sha256'] else 'DIFFERENT'}"
                 if a.generation_tie else ""), flush=True)
    res = {"script": f"b_k2_input_identity {VERSION}", "server": "vllm", "date": dt.date.today().isoformat(),
           "server_version": server_version, "versions": versions, "model": served, "server_process": server,
           "request": {"chat_template_kwargs": kwargs, "max_tokens": st["max_tokens"], "seed": st["seed"], "temperature": 0},
           "run_think": st["think"], "settings_from_run_json": st["from_run_json"], "build": run.name, "items_sha256": items_sha,
           "reference": {"file": Path(a.reference).name, "sha256": sha(Path(a.reference).read_bytes())},
           "sequence_hash_format": "SHA-256 of json.dumps(ids, separators=(',', ':')) in UTF-8",
           "render_path": "POST /v1/chat/completions/render (OnlineRenderer.render_chat, the chat route's own)",
           "generation_tie_run": bool(a.generation_tie),
           "generation_tie_meaning": "prompt_token_ids: the tokens vLLM's API server handed to the engine for that request "
                                     "(rendered again by the chat route), not observed inside the plugin's runner",
           "cards": cards}
    res["passed"] = not check_report(res, items, rk, require_flag=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(mask_home(res), ensure_ascii=False, indent=1))
    print(f"{len(cards)} cards; generation tie {'run' if a.generation_tie else 'not run'}; passed {res['passed']}; written {out}")
    if not res["passed"]:
        sys.exit(f"the report does not pass its own checks: {check_report(res, items, rk, require_flag=False)}")


def count_ok(x, n) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x == n


def check_report(d: dict, build_items: dict | None = None, build_rk: list | None = None, require_flag: bool = True) -> list:
    """Problems of one report, recomputed from its cards (the stored flags are never trusted): repeated card ids, a
    sequence whose length or SHA-256 does not match, a card whose count does not equal its length, the role x kind
    coverage, the runner tie before and after, and the stored flags against all of this."""
    out = []
    cards = d.get("cards") or []
    if not all(isinstance(c, dict) and isinstance(c.get("item"), str) for c in cards):
        return ["a card without an item id"]
    ids = [c.get("item") for c in cards]
    rep_ids = sorted({i for i in ids if ids.count(i) > 1})
    if rep_ids:
        out.append(f"repeated card ids {rep_ids}")
    for c in cards:
        t = c.get("token_ids")
        try:
            h = seq_sha(t) if isinstance(t, list) and t else None
        except ValueError:
            h = None
        if h is None or h != c.get("token_ids_sha256") or not count_ok(c.get("token_count"), len(t)):
            out.append(f"{c.get('item')}: the stored sequence, its length and its SHA-256 disagree (or it is empty)")
        elif d.get("server") == "ollama" and not count_ok(c.get("prompt_eval_count"), len(t)):
            out.append(f"{c.get('item')}: prompt_eval_count {c.get('prompt_eval_count')!r} differs from the sequence's length {len(t)}")
        elif d.get("server") == "vllm" and d.get("generation_tie_run") is True and (
                c.get("generation_prompt_token_ids_sha256") != c.get("token_ids_sha256")
                or not count_ok(c.get("generation_prompt_tokens"), len(t))):
            out.append(f"{c.get('item')}: the prompt_token_ids of the generation differ from the rendered ids (or their count)")
        if build_items is not None:
            it = build_items.get(c.get("item"))
            if it is None:
                out.append(f"{c.get('item')}: no such item in the build")
            elif (role(it), it["kind"], it["order"]) != (c.get("role"), c.get("kind"), c.get("order")):
                out.append(f"{c.get('item')}: role, kind or order differ from the build")
    have = sorted({f"{c.get('role')} | {c.get('kind')}" for c in cards})
    if build_rk is None or have != build_rk:
        out.append("the cards do not cover every role x kind of the build (recomputed from the build)")
    if d.get("server") == "ollama":
        rp = d.get("runner_process") or {}
        b, f = rp.get("before") or {}, rp.get("after") or {}
        port = b.get("runner_port")
        if not (facts_ok(b, d.get("model"), port) and facts_ok(f, d.get("model"), port) and b.get("runner_pid") == f.get("runner_pid")):
            out.append("no verified runner process (recomputed from the recorded facts: before and after, the same pid, the "
                       "server's only runner of the model)")
    elif d.get("server") == "vllm":
        v = d.get("versions") or {}
        sp = d.get("server_process") or {}
        if not all(isinstance(v.get(k), dict) and v[k].get("version") for k in REQUIRED_VERSIONS):
            out.append(f"versions not recorded for all of {list(REQUIRED_VERSIONS)}")
        elif not d.get("server_version") or d.get("server_version") != v["vllm"]["version"]:
            out.append(f"the server's /version {d.get('server_version')!r} is not the vLLM of the recorded environment")
        pc = parse_vllm_command(sp.get("command") or "")
        if not (pid_ok(sp.get("pid")) and pc["serve"] and pc["python"] and pc["python"] == sp.get("python")
                and "--enable-scale-out" in pc["flags"] and pc["flags"] == sp.get("render_flags")):
            out.append("no verified server process (recomputed from its command line: interpreter, vllm serve, --enable-scale-out, flags)")
        if not (sp.get("plugin") or {}).get("open_files_of_the_server_tree") and not (sp.get("plugin") or {}).get("server_log_lines"):
            out.append("no evidence that the vllm-metal plugin is active")
        need = max((c.get("token_count") or 0 for c in cards), default=0) + ((d.get("request") or {}).get("max_tokens") or 0)
        if not isinstance(sp.get("max_model_len"), int) or sp["max_model_len"] < need:
            out.append(f"max_model_len {sp.get('max_model_len')!r} is below the longest prompt plus max_tokens ({need})")
        if not thinking_matches(d.get("run_think"), (d.get("request") or {}).get("chat_template_kwargs") or {}):
            out.append("the request's thinking switch does not match the run's")
        if d.get("render_path") is None:
            out.append("no render path recorded")
    else:
        out.append(f"unknown server {d.get('server')!r}")
    if require_flag and not out and d.get("passed") is not True:
        out.append("the report says it did not pass")
    return out


def load_build(run: Path) -> tuple[dict, list, str]:
    data = gzip.decompress((run / "items.jsonl.gz").read_bytes())
    design = json.loads((run / "design.json").read_text())
    if sha(data) != design["items_sha256"]:
        sys.exit("items.jsonl.gz does not match design.json")
    items = {it["item"]: it for it in (json.loads(x) for x in data.decode("utf-8").splitlines() if x.strip())}
    return items, sorted({f"{role(it)} | {it['kind']}" for it in items.values()}), design["items_sha256"]


def compare(a) -> None:
    x, y = (json.loads(Path(p).read_text()) for p in (a.a, a.b))
    items, rk, items_sha = load_build(Path(a.run))
    diff = []
    for name, d in (("a", x), ("b", y)):
        if d.get("items_sha256") != items_sha:
            diff.append({name: "made on another build than --run"})
        for prob in check_report(d, items, rk):
            diff.append({name: prob})
    if x.get("items_sha256") != y.get("items_sha256"):
        diff.append({"different items_sha256": [x.get("items_sha256"), y.get("items_sha256")]})
    if x.get("server") == y.get("server") and x.get("request") != y.get("request"):
        diff.append({"different request": [x.get("request"), y.get("request")]})
    cx, cy = ({c["item"]: c for c in d.get("cards") or [] if isinstance(c, dict) and isinstance(c.get("item"), str)} for d in (x, y))
    if not set(cx) & set(cy):
        diff.append({"problem": "no card in common"})
    if set(cx) != set(cy):
        diff.append({"cards_only_in_a": sorted(set(cx) - set(cy)), "cards_only_in_b": sorted(set(cy) - set(cx))})
    for k in sorted(set(cx) & set(cy)):
        try:
            stored_ok = all(seq_sha(c["token_ids"]) == c["token_ids_sha256"] for c in (cx[k], cy[k]))
        except (ValueError, TypeError, KeyError):
            stored_ok = False
        if not stored_ok:
            diff.append({"item": k, "problem": "a stored sequence does not match its stored SHA-256"})
        elif cx[k]["token_ids_sha256"] != cy[k]["token_ids_sha256"]:
            p = next((i for i, (u, v) in enumerate(zip(cx[k]["token_ids"], cy[k]["token_ids"])) if u != v),
                     min(len(cx[k]["token_ids"]), len(cy[k]["token_ids"])))
            u, v = cx[k]["token_ids"], cy[k]["token_ids"]
            diff.append({"item": k, "lengths": [len(u), len(v)], "first_difference_at": p,
                         "context_a": {"from": max(0, p - 8), "ids": u[max(0, p - 8):p + 9]},
                         "context_b": {"from": max(0, p - 8), "ids": v[max(0, p - 8):p + 9]}})
    expected = {it["item"] for it in select(list(items.values()), STAGE_A_CARDS)}
    card_set = [{name: "its cards are not the protocol's set of %d (select on the build)" % STAGE_A_CARDS}
                for name, c in (("a", cx), ("b", cy)) if set(c) != expected or len((x if name == "a" else y).get("cards") or []) != STAGE_A_CARDS]
    identical = not diff
    passed = identical and not card_set
    print(json.dumps({"a": [x.get("server"), x.get("ollama_version") or x.get("server_version"), x.get("model_digest") or x.get("model")],
                      "b": [y.get("server"), y.get("ollama_version") or y.get("server_version"), y.get("model_digest") or y.get("model")],
                      "generation_tie": {"a": "runner process and prompt_eval_count" if x.get("server") == "ollama" else
                                         ("prompt_token_ids per request" if x.get("generation_tie_run") else "not run"),
                                         "b": "runner process and prompt_eval_count" if y.get("server") == "ollama" else
                                         ("prompt_token_ids per request" if y.get("generation_tie_run") else "not run")},
                      "cards": len(set(cx) & set(cy)), "sequences_identical": identical, "stage_a_passed": passed,
                      "differences": diff, "card_set": card_set}, ensure_ascii=False, indent=1))
    if not passed:
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("ollama")
    o.add_argument("--run", required=True)
    o.add_argument("--model", help="default: the build's run.json")
    o.add_argument("--base-url", default="http://127.0.0.1:11434")
    o.add_argument("--runner-url", required=True, help="the MLX runner's own local address (not port 11434), or auto")
    o.add_argument("--n", type=int, default=STAGE_A_CARDS, help="another number gives a diagnostic report that never passes stage A")
    o.add_argument("--think", help="default: run.json, else false")
    o.add_argument("--num-ctx", type=int, help="default: run.json, else 32768")
    o.add_argument("--max-tokens", type=int, help="default: run.json, else 2000")
    o.add_argument("--seed", type=int, help="default: run.json, else 20261114")
    o.add_argument("--timeout", type=float, default=600)
    o.add_argument("--out", required=True)
    o.add_argument("--force", action="store_true")
    v = sub.add_parser("vllm")
    v.add_argument("--run", required=True)
    v.add_argument("--reference", required=True, help="the Ollama report whose 20 cards are used")
    v.add_argument("--base-url", default="http://127.0.0.1:8000")
    v.add_argument("--model", help="the served model name (default: the build's run.json model)")
    v.add_argument("--python", help="optional: must be the server's own interpreter (found from its process)")
    v.add_argument("--model-dir", help="optional: must be the directory the server serves (found from its process)")
    v.add_argument("--server-log", help="the server's startup log (evidence of the plugin's activation, recorded)")
    v.add_argument("--chat-template-kwargs", default='{"enable_thinking": false}')
    v.add_argument("--generation-tie", action="store_true", help="also run each card (max_tokens 1, return_token_ids)")
    v.add_argument("--max-tokens", type=int)
    v.add_argument("--seed", type=int)
    v.add_argument("--think", help=argparse.SUPPRESS)
    v.add_argument("--num-ctx", type=int, help=argparse.SUPPRESS)
    v.add_argument("--timeout", type=float, default=600)
    v.add_argument("--out", required=True)
    v.add_argument("--force", action="store_true")
    c = sub.add_parser("compare")
    c.add_argument("--run", required=True, help="the build both reports were made on (coverage and cards recomputed from it)")
    c.add_argument("a")
    c.add_argument("b")
    a = ap.parse_args()
    {"ollama": ollama, "vllm": vllm, "compare": compare}[a.cmd](a)


if __name__ == "__main__":
    main()
