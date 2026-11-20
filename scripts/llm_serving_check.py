#!/usr/bin/env python3
"""Checks a local model server before a full agent run on the k3 scenarios.

Works with any OpenAI-compatible endpoint (Ollama, mlx_lm.server, vllm-metal, llama-server, vLLM,
SGLang). Standard library only, so it runs on a fresh machine. Four checks:

1. context: the same request with the decision key repeated 1-4 times. The prompt token count the
   server reports must grow linearly; a plateau means the server truncates the prompt (Ollama does
   this silently when the prompt exceeds its context length). A code placed at the very start of the
   prompt must come back in the reply.
2. load: real k3 episodes (public working split) at increasing concurrency: requests per second,
   output tokens per second, latency p50 / p95, errors (e.g. HTTP 503 from a full queue).
3. determinism: one episode 8 times one after another and 8 times at once, temperature 0 and a fixed
   seed. Different replies mean results depend on load (batching), which matters for reproducibility.
4. format: share of replies that parse as the JSON object asked for, with a valid action.

Example (Ollama, thinking off through the native API, which honours it for every model):
  python3 scripts/llm_serving_check.py --model qwen3.8:27b --api ollama --think false

Example (Ollama, OpenAI-compatible API):
  OLLAMA_CONTEXT_LENGTH=16384 OLLAMA_NUM_PARALLEL=8 ollama serve
  python3 scripts/llm_serving_check.py --base-url http://localhost:11434/v1 --model qwen3:14b

Results go to data/processed/serving/<server>_<model>_<date>.json (no personal data: the public
scenarios only, replies are not stored beyond the determinism check).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import gzip
import json
import platform
import random
import re
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.5.2"

INSTRUCTIONS = (
    "Session code: {code}.\n"
    "You decide k3 cases: whether a business customer's register data held by the bank is still current. "
    "Use only the decision key and the case record below. Reply with one JSON object and nothing else: "
    '{{"action": <one of the actions>, "handoffs": [<handoff ids>], "interim_measure": <id or null>, '
    '"session_code": <the session code given at the very start>}}.\n\nDecision key:\n{key}'
)


def agent_key(policy: dict) -> dict:
    """The sections of the key the agent may see (scoring.agent_input), without alternative readings."""
    keep = policy.get("scoring", {}).get("agent_sections") or [
        "actions", "action_meaning", "outputs", "review_changes", "data_changes", "handoffs", "handoffs_on_opening",
        "interim_measures", "retry_days", "max_failed_attempts", "periodic_review_days", "risk_classes", "roles",
        "status_labels", "periodic_review_rule"]  # keys before 1.0.1 had no agent_sections
    out = {k: policy[k] for k in keep if k in policy and k != "interpretations"}
    interp = policy.get("interpretations", {})
    def strip(d: dict) -> dict:  # only the chosen reading: no alternative, also not inside "readings"
        out = {k: v for k, v in d.items() if "alternative" not in k}
        if isinstance(out.get("readings"), dict):
            out["readings"] = {k: v for k, v in out["readings"].items() if k == "chosen"}
        return out
    out["interpretations"] = [strip(i) for i in interp] if isinstance(interp, list) else {k: strip(v) for k, v in interp.items()}
    return out


def as_text(content) -> str:
    """Message content as text: OpenAI-style lists of parts are joined, anything else is dumped."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join((p.get("text") or "") if isinstance(p, dict) and p.get("type", "text") in ("text", "output_text")
                       else (p if isinstance(p, str) else "") for p in content)
    return json.dumps(content, ensure_ascii=False) if content is not None else ""


def openai_chat_body(model: str, messages: list, max_tokens: int, seed: int | None, reasoning_effort: str | None = None,
                     chat_template_kwargs: dict | None = None) -> dict:
    """The body of an OpenAI-compatible chat request, built in one place so that a render-only request
    (b_k2_input_identity, vLLM's /v1/chat/completions/render) is the real request itself."""
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0}
    if seed is not None:
        body["seed"] = seed
    if reasoning_effort:  # Ollama /v1 ignores "think"; "none" turns thinking off where the model allows it
        body["reasoning_effort"] = reasoning_effort
    if chat_template_kwargs:  # e.g. vLLM: {"enable_thinking": false} for Qwen's template
        body["chat_template_kwargs"] = chat_template_kwargs
    return body


def chat(base: str, model: str, messages: list, max_tokens: int, seed: int | None, timeout: float, api_key: str,
         reasoning_effort: str | None = None, chat_template_kwargs: dict | None = None) -> dict:
    body = openai_chat_body(model, messages, max_tokens, seed, reasoning_effort, chat_template_kwargs)
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
        usage = data.get("usage") or {}
        msg = (data.get("choices") or [{}])[0].get("message") or {}
        reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
        # some models on Ollama /v1 leave content empty and put everything in the reasoning field
        text = as_text(msg.get("content"))
        return {"ok": True, "latency": time.perf_counter() - t0, "text": text,
                "reasoning_chars": len(as_text(reasoning)), "empty_content": not text.strip(),
                "fallback_text": as_text(reasoning), "fallback_allowed": True,
                "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                "prompt_cached_tokens": openai_cached(usage),
                "finish_reason": (data.get("choices") or [{}])[0].get("finish_reason")}
    except urllib.error.HTTPError as e:
        return {"ok": False, "latency": time.perf_counter() - t0, "error": f"HTTP {e.code}"}
    except Exception as e:  # timeouts, refused connections
        return {"ok": False, "latency": time.perf_counter() - t0, "error": type(e).__name__}


THINK_VALUES = {"false", "true", "low", "medium", "high", "xhigh", "max"}


def _count(x):
    return x if isinstance(x, int) and not isinstance(x, bool) else None


def openai_cached(usage: dict) -> int | None:
    """usage.prompt_tokens_details.cached_tokens of an OpenAI-compatible response, None if absent or malformed."""
    d = usage.get("prompt_tokens_details") if isinstance(usage, dict) else None
    return _count(d.get("cached_tokens")) if isinstance(d, dict) else None


def server_timing(data: dict) -> dict:
    """What Ollama's native response reports about the call itself: prompt tokens taken from the prefix cache
    (prompt_eval_cached_count: a pointer in Ollama, so 0 is sent as 0 and the field is absent only when the runner
    does not report it) and the server's own durations in seconds (loading the model, evaluating the prompt,
    generating, total). Ollama omits a duration that is exactly 0 (omitempty), so a None duration means 0 or not
    reported (load_s is typically absent when the model was already loaded): never read it as missing data."""
    s = lambda k: _count(data.get(k)) / 1e9 if _count(data.get(k)) is not None else None  # noqa: E731
    return {"prompt_cached_tokens": _count(data.get("prompt_eval_cached_count")), "load_s": s("load_duration"),
            "prompt_eval_s": s("prompt_eval_duration"), "eval_s": s("eval_duration"), "total_s": s("total_duration")}


def ollama_chat_body(model: str, messages: list, max_tokens: int, seed: int | None, think: str | None = None,
                     num_ctx: int | None = None) -> dict:
    """The body of Ollama's native /api/chat request, built in one place so that a render-only request
    (b_k2_input_identity) is the real request plus one flag."""
    options = {"temperature": 0, "num_predict": max_tokens, **({"seed": seed} if seed is not None else {}),
               **({"num_ctx": num_ctx} if num_ctx else {})}
    body = {"model": model, "messages": messages, "stream": False, "options": options}
    if think is not None:
        body["think"] = {"false": False, "true": True}.get(think, think)
    return body


def chat_ollama(base: str, model: str, messages: list, max_tokens: int, seed: int | None, timeout: float,
                think: str | None = None, num_ctx: int | None = None) -> dict:
    """Ollama's native API, where think=false is honoured (the /v1 endpoint ignores it for some models).
    num_ctx sets the context length per request, so a server default cannot truncate the prompt."""
    root = re.sub(r"/v1/?$", "", base.rstrip("/"))
    body = ollama_chat_body(model, messages, max_tokens, seed, think, num_ctx)
    req = urllib.request.Request(root + "/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
        msg = data.get("message") or {}
        thinking, text = as_text(msg.get("thinking")), as_text(msg.get("content"))
        # the native API keeps thinking apart from the answer: the thinking is never read as the answer
        return {"ok": True, "latency": time.perf_counter() - t0, "text": text,
                "reasoning_chars": len(thinking), "empty_content": not text.strip(),
                "fallback_text": thinking, "fallback_allowed": False, "prompt_tokens": data.get("prompt_eval_count"),
                "completion_tokens": data.get("eval_count"), "finish_reason": data.get("done_reason"),
                **server_timing(data)}
    except urllib.error.HTTPError as e:
        return {"ok": False, "latency": time.perf_counter() - t0, "error": f"HTTP {e.code}"}
    except Exception as e:
        return {"ok": False, "latency": time.perf_counter() - t0, "error": type(e).__name__}


def parse_reply(r: dict) -> dict | None:
    """The decision in a reply. A reply cut by the token limit is no decision (its JSON may be a draft).
    The thinking text is read only when the content is empty and the model stopped on its own (seen
    with some models on Ollama /v1)."""
    if r.get("finish_reason") == "length":
        return None
    got = parse(r.get("text", ""))
    if got is None and r.get("fallback_allowed") and r.get("empty_content") and r.get("finish_reason") in ("stop", None):
        got = parse(r.get("fallback_text", ""))
        if got is not None:
            r["answer_from_reasoning"] = True
    return got


def parse(text) -> dict | None:
    """The last JSON object in the text with a string "action", outside any <think> block (closed or
    not). Objects are decoded with the JSON parser, so nested objects and braces in strings are fine."""
    if not isinstance(text, str):
        text = json.dumps(text) if isinstance(text, (dict, list)) else str(text or "")
    if "<think>" in text:
        text = re.sub(r"<think>.*?(</think>|$)", "", text, flags=re.S)
    elif "</think>" in text:  # the opening tag was in the prompt template: keep only what follows the reasoning
        text = text.rsplit("</think>", 1)[1]
    dec, found, i = json.JSONDecoder(), None, text.find("{")
    while i != -1:
        try:
            obj, end = dec.raw_decode(text, i)
        except (ValueError, RecursionError):
            i = text.find("{", i + 1)
            continue
        if isinstance(obj, dict) and isinstance(obj.get("action"), str):
            found = obj
        i = text.find("{", end)
    return found


def newest_agent_file() -> Path:
    """The newest k3 agent file by numeric version; a release sorts after its draft (1.0.6 > 1.0.6-draft).
    Two files of the newest version are an error (no quiet choice by name or modification time)."""
    def key(p: Path):
        m = re.match(r"k3_v(\d+)\.(\d+)(?:\.(\d+))?(-draft)?", p.name)
        if not m:
            return ((0, 0, 0), False)
        return (tuple(int(x or 0) for x in m.groups()[:3]), m.group(4) is None)
    files = sorted((ROOT / "data/scenarios").glob("k3_v*.agent.jsonl.gz"), key=key)
    if not files:
        raise ValueError("no k3 agent file in data/scenarios")
    top = [f for f in files if key(f) == key(files[-1])]
    if len(top) > 1:  # two builds of one version (another sample or seed): refuse to pick one quietly
        raise ValueError(f"several k3 agent files of the newest version: {[f.name for f in top]}")
    return files[-1]


def messages_for(key_text: str, episode: dict, code: str, repeat: int = 1) -> list:
    system = INSTRUCTIONS.format(code=code, key="\n".join([key_text] * repeat))
    return [{"role": "system", "content": system},
            {"role": "user", "content": "Case record:\n" + json.dumps(episode, ensure_ascii=False)}]


def pct(v: list, p: float):
    if not v:
        return None
    v = sorted(v)
    return round(v[min(len(v) - 1, int(round(p * (len(v) - 1))))], 3)


def host_info() -> dict:
    info = {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version()}
    for k, cmd in (("chip", ["sysctl", "-n", "machdep.cpu.brand_string"]), ("memory_bytes", ["sysctl", "-n", "hw.memsize"]),
                   ("model", ["sysctl", "-n", "hw.model"])):
        try:
            info[k] = subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout.strip() or None
        except Exception:
            info[k] = None
    return info


def server_info(base: str) -> dict:
    root = re.sub(r"/v1/?$", "", base.rstrip("/"))
    for path in ("/api/version", "/version"):
        try:
            with urllib.request.urlopen(root + path, timeout=5) as r:
                return {"version_endpoint": path, "response": json.loads(r.read())}
        except Exception:
            continue
    return {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--server", default="ollama", help="label for the results file, e.g. ollama, mlx, vllm-metal")
    ap.add_argument("--api-key", default="local", help="dummy for local servers")
    ap.add_argument("--concurrency", default="1,2,4,8,16")
    ap.add_argument("--requests-per-level", type=int, default=0, help="default: max(8, 2 x concurrency)")
    ap.add_argument("--max-tokens", type=int, default=1024, help="raise for reasoning models")
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--reasoning-effort", default=None,
                    help="OpenAI-compatible API: none, low, medium, high (values depend on the server and model)")
    ap.add_argument("--api", default="openai", choices=["openai", "ollama"],
                    help="ollama: Ollama's native /api/chat, where --think false is honoured")
    ap.add_argument("--think", default=None, help="native Ollama API: false, true or a level the model lists "
                    "(ollama show), e.g. low, medium, xhigh; keep it the same for every model you compare")
    ap.add_argument("--num-ctx", type=int, default=32768, help="native Ollama API: context length per request "
                    "(the same as run_k3_agent.py, so the model is not reloaded with another context)")
    ap.add_argument("--seed", type=int, default=7, help="sampling seed sent to the server")
    ap.add_argument("--episode-seed", type=int, default=None,
                    help="which episodes the load check draws; default: new each run, so no run reuses prompts "
                         "a previous run left in the server's cache (written to the results)")
    ap.add_argument("--skip", default="", help="comma list of checks to skip: context,load,determinism")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    skip = set(filter(None, a.skip.split(",")))
    if a.think is not None and a.api != "ollama":
        raise SystemExit("--think is for --api ollama; use --reasoning-effort with the OpenAI-compatible API")
    if a.think is not None:
        if a.think.lower() not in THINK_VALUES:
            raise SystemExit(f"--think must be one of {sorted(THINK_VALUES)}")
        a.think = a.think.lower()

    policy = json.loads((ROOT / "data/reference/k3_policy.json").read_text())
    key_text = json.dumps(agent_key(policy), ensure_ascii=False)
    actions = set(policy["actions"]) if isinstance(policy["actions"], list) else set(policy["actions"])
    path = newest_agent_file()
    episodes = [json.loads(l) for l in gzip.open(path)]
    episodes.sort(key=lambda e: -len(json.dumps(e, ensure_ascii=False)))  # longest first: worst case
    res = {"script": VERSION, "date": dt.datetime.now().isoformat(timespec="seconds"), "base_url": a.base_url,
           "model": a.model, "server": a.server, "server_info": server_info(a.base_url), "host": host_info(),
           "scenarios": path.name, "max_tokens": a.max_tokens, "reasoning_effort": a.reasoning_effort, "api": a.api, "think": a.think, "num_ctx": a.num_ctx if a.api == "ollama" else None, "warnings": []}
    if a.api == "ollama":
        call = lambda msgs: chat_ollama(a.base_url, a.model, msgs, a.max_tokens, a.seed, a.timeout, a.think, a.num_ctx)
    else:
        call = lambda msgs: chat(a.base_url, a.model, msgs, a.max_tokens, a.seed, a.timeout, a.api_key, a.reasoning_effort)

    first = call(messages_for(key_text, episodes[0], "K3-0000"))
    if not first["ok"]:
        raise SystemExit(f"server not answering: {first.get('error')} at {a.base_url}")

    if "context" not in skip:
        rows = []
        for rep in (1, 2, 3, 4):
            code = f"K3-{rep}{rep}{rep}{rep}"
            r = call(messages_for(key_text, episodes[0], code, rep))
            got = parse_reply(r) or {}
            rows.append({"key_copies": rep, "prompt_tokens": r.get("prompt_tokens"), "ok": r["ok"],
                         "code_returned": got.get("session_code") == code, "error": r.get("error")})
            print(f"context x{rep}: prompt_tokens={r.get('prompt_tokens')} code_returned={rows[-1]['code_returned']}")
        pt = [x["prompt_tokens"] for x in rows if x["prompt_tokens"]]
        linear = len(pt) == 4 and all(pt[i + 1] - pt[i] > 0.8 * (pt[1] - pt[0]) for i in range(3))
        res["context"] = {"rows": rows, "grows_linearly": linear,
                          "longest_episode_prompt_tokens": rows[0]["prompt_tokens"]}
        if not linear:
            res["warnings"].append("prompt token count does not grow linearly: the server truncates long prompts "
                                   "(for Ollama raise OLLAMA_CONTEXT_LENGTH)")
        if not rows[0]["code_returned"]:
            res["warnings"].append("the session code from the start of the prompt did not come back for one key copy")

    if "load" not in skip:
        # every level gets its own episodes, so no level reuses a prompt the server has cached; drawn at
        # random so prompt lengths are alike across levels. The shared key prefix stays
        # cacheable, as in a real run.
        # the system prompt (instructions + key) is the same in every request, as in a real agent run, so the
        # server may reuse it from its prompt cache; only the case record differs
        levels, offset = [], 0
        res["episode_seed"] = a.episode_seed if a.episode_seed is not None else random.SystemRandom().randrange(2**31)
        pool = random.Random(res["episode_seed"]).sample(episodes[1:], len(episodes) - 1)
        for c in [int(x) for x in a.concurrency.split(",")]:
            n = a.requests_per_level or max(8, 2 * c)
            batch = [messages_for(key_text, pool[(offset + i) % len(pool)], "K3-0000") for i in range(n)]
            offset += n
            t0 = time.perf_counter()
            with cf.ThreadPoolExecutor(max_workers=c) as ex:
                out = list(ex.map(call, batch))
            wall = time.perf_counter() - t0
            ok = [r for r in out if r["ok"]]
            parsed = [parse_reply(r) for r in ok]
            valid = [p for p in parsed if p and isinstance(p.get("action"), str) and p["action"] in actions]
            toks = sum(r.get("completion_tokens") or 0 for r in ok)
            row = {"concurrency": c, "requests": n, "ok": len(ok), "errors": sorted({r.get("error") for r in out if not r["ok"]}),
                   "wall_s": round(wall, 2), "requests_per_s": round(len(ok) / wall, 3),
                   "output_tokens_per_s": round(toks / wall, 1),
                   "latency_p50_s": pct([r["latency"] for r in ok], 0.5), "latency_p95_s": pct([r["latency"] for r in ok], 0.95),
                   "max_prompt_tokens": max((r.get("prompt_tokens") or 0 for r in ok), default=None),
                   "mean_prompt_tokens": round(statistics.mean([r.get("prompt_tokens") or 0 for r in ok]), 1) if ok else None,
                   "cut_by_max_tokens": sum(r.get("finish_reason") == "length" for r in ok),
                   "empty_content": sum(bool(r.get("empty_content")) for r in ok),
                   "mean_completion_tokens": round(statistics.mean([r.get("completion_tokens") or 0 for r in ok]), 1) if ok else None,
                   "valid_json_action_share": round(len(valid) / len(ok), 3) if ok else None}
            levels.append(row)
            print(f"load c={c}: {row['requests_per_s']} req/s, {row['output_tokens_per_s']} tok/s, "
                  f"p50 {row['latency_p50_s']} s, p95 {row['latency_p95_s']} s, ok {len(ok)}/{n}, "
                  f"valid {row['valid_json_action_share']}, cut {row['cut_by_max_tokens']}, "
                  f"mean reply {row['mean_completion_tokens']} tok")
        res["load"] = levels
        base = levels[0]["requests_per_s"] or 1e-9
        res["speedup_vs_c1"] = {str(r["concurrency"]): round(r["requests_per_s"] / base, 2) for r in levels}
        if levels and (levels[0]["mean_completion_tokens"] or 0) > 300:
            res["warnings"].append(f"replies average {levels[0]['mean_completion_tokens']} tokens: the model is "
                                   "thinking; if you meant it off, try --api ollama --think false")
        if any(r["empty_content"] for r in levels):
            res["warnings"].append("some replies have empty content and the text only in the reasoning field: "
                                   + ("the model did not finish its answer; raise --max-tokens or use --think false"
                                      if a.api == "ollama" else
                                      "seen with Gemma 4 on Ollama /v1; try --reasoning-effort none or --api ollama --think false"))
        if any(r["cut_by_max_tokens"] for r in levels):
            res["warnings"].append("some replies hit max_tokens: raise --max-tokens (reasoning models)")
        if len(levels) > 1 and res["speedup_vs_c1"][str(levels[-1]["concurrency"])] < 1.5:
            res["warnings"].append("throughput hardly grows with concurrency. Either requests run one at a time "
                                   "(OLLAMA_NUM_PARALLEL=1, or Ollama's MLX runner: 'ollama ps' shows RUNNER mlx), or "
                                   "the GPU is already saturated by one request (prompt reading is compute-bound); "
                                   "check GPU use at concurrency 1")

    if "determinism" not in skip:
        msgs = messages_for(key_text, episodes[len(episodes) // 2], "K3-0000")
        seq = [call(msgs) for _ in range(8)]
        with cf.ThreadPoolExecutor(max_workers=8) as ex:
            par = list(ex.map(lambda _: call(msgs), range(8)))
        norm = lambda r: r.get("text", "").strip() if r["ok"] else None
        decision = lambda r: json.dumps({k: (parse_reply(r) or {}).get(k) for k in ("action", "handoffs", "interim_measure")}) if r["ok"] else None
        res["determinism"] = {"distinct_texts_sequential": len({norm(r) for r in seq}),
                              "distinct_texts_concurrent": len({norm(r) for r in par}),
                              "distinct_texts_all": len({norm(r) for r in seq + par}),
                              "distinct_decisions_all": len({decision(r) for r in seq + par})}
        print("determinism:", res["determinism"])
        if res["determinism"]["distinct_decisions_all"] > 1:
            res["warnings"].append("the same case at temperature 0 gives different decisions: results depend on load")

    out = Path(a.out) if a.out else ROOT / "data/processed/serving" / (
        re.sub(r"[^A-Za-z0-9._-]+", "-", f"{a.server}_{a.model}_{dt.datetime.now().strftime('%Y-%m-%d_%H%M%S')}") + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for w in res["warnings"]:
        print("WARNING:", w)
    print("written", out.relative_to(ROOT) if out.is_relative_to(ROOT) else out)


if __name__ == "__main__":
    main()


# --- Anthropic Messages API (B-K2 pilot on Claude Haiku 5.5; the official SDK, its own retries off) ---

def anthropic_chat_body(model: str, messages: list, max_tokens: int, effort: str, thinking: str = "disabled") -> dict:
    """The Messages API body, built in one place: a leading system message becomes the top-level system field; no
    temperature, top_p or top_k (Claude Haiku 5.5 rejects values other than the defaults); thinking and effort explicit."""
    system = messages[0]["content"] if messages and messages[0]["role"] == "system" else None
    rest = messages[1:] if system is not None else messages
    body = {"model": model, "max_tokens": max_tokens, "messages": rest, "thinking": {"type": thinking},
            "output_config": {"effort": effort}}
    if system is not None:
        body["system"] = system
    return body


class RateGate:
    """Shared by the worker threads: after every response the rate-limit headers are read; when the requests or the
    input or output tokens left in the window fall below what the running calls may need, every next call waits
    until that window resets. Waits are counted and logged, never hidden; the limits themselves are never raised."""

    def __init__(self, concurrency: int, input_per_call: int = 10000, output_per_call: int = 2500):
        import threading
        self.lock = threading.Lock()
        self.until = 0.0
        self.need = {"requests": concurrency, "input-tokens": concurrency * input_per_call,
                     "output-tokens": concurrency * output_per_call}
        self.waits, self.waited_s, self.last = 0, 0.0, {}

    @staticmethod
    def _reset(value: str | None) -> float | None:
        if not value:
            return None
        try:
            return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None

    def update(self, headers) -> None:
        snap = {}
        until = 0.0
        for kind, need in self.need.items():
            rem = headers.get(f"anthropic-ratelimit-{kind}-remaining")
            reset = self._reset(headers.get(f"anthropic-ratelimit-{kind}-reset"))
            snap[kind] = {"remaining": rem, "reset": headers.get(f"anthropic-ratelimit-{kind}-reset")}
            if rem is not None and rem.isdigit() and int(rem) < need and reset:
                until = max(until, reset)
        with self.lock:
            self.last = snap
            self.until = max(self.until, until)

    def after(self, seconds: float) -> None:
        with self.lock:
            self.until = max(self.until, time.time() + seconds)

    def wait(self) -> float:
        with self.lock:
            w = self.until - time.time()
        if w > 0:
            with self.lock:
                self.waits += 1
                self.waited_s += w
            print(f"  rate limit: waiting {w:.0f} s for the window to reset", flush=True)
            time.sleep(w)
            return w
        return 0.0


def chat_anthropic(client, body: dict, timeout: float, gate: RateGate | None = None) -> dict:
    """One call through client.messages.with_raw_response (the SDK's own retries must be off: max_retries=0 on the
    client; the caller retries transport errors with one visible count). Returns the same keys as chat_ollama, plus
    stop_reason, the refusal's category, the four usage counts, the request id, the model that served it and the
    rate-limit headers; on an HTTP error ok false, retryable (429, 5xx, connection, timeout) or fatal, and the
    server's retry-after."""
    import anthropic
    t0 = time.perf_counter()
    try:
        raw = client.messages.with_raw_response.create(**body, timeout=timeout)
        msg = raw.parse()
        if gate is not None:
            gate.update(raw.headers)
        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        u = msg.usage
        sd = getattr(msg, "stop_details", None)
        finish = {"end_turn": "stop", "max_tokens": "length", "model_context_window_exceeded": "length",
                  "refusal": "refusal"}.get(msg.stop_reason, msg.stop_reason)
        return {"ok": True, "latency": time.perf_counter() - t0, "text": text, "empty_content": not text.strip(),
                "prompt_tokens": u.input_tokens, "completion_tokens": u.output_tokens,
                "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", None),
                "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", None),
                "finish_reason": finish, "stop_reason": msg.stop_reason,
                "refusal_category": getattr(sd, "category", None) if sd is not None else None,
                "request_id": raw.headers.get("request-id"), "served_model": msg.model}
    except anthropic.RateLimitError as e:
        h = e.response.headers
        ra, ms = h.get("retry-after"), h.get("retry-after-ms")
        wait = float(ms) / 1000 if ms and ms.replace(".", "", 1).isdigit() else (
            float(ra) if ra and ra.replace(".", "", 1).isdigit() else None)
        return {"ok": False, "latency": time.perf_counter() - t0, "error": "HTTP 429", "retryable": True, "retry_after": wait}
    except anthropic.APIStatusError as e:
        return {"ok": False, "latency": time.perf_counter() - t0, "error": f"HTTP {e.status_code}",
                "retryable": e.status_code >= 500 or e.status_code in (408, 409), "fatal": e.status_code < 500 and e.status_code not in (408, 409),
                "detail": "" if e.status_code in (401, 403) else _redact(str(getattr(e, "message", "")))[:200]}
    except (anthropic.APITimeoutError, anthropic.APIConnectionError) as e:
        return {"ok": False, "latency": time.perf_counter() - t0, "error": type(e).__name__, "retryable": True}
    except Exception as e:  # anything else (a body that is not JSON, a response without usage): a transport error, never a crash
        return {"ok": False, "latency": time.perf_counter() - t0, "error": f"unexpected {type(e).__name__}", "retryable": True}


def anthropic_cost(row: dict, price: dict, by_kind: bool = False):
    """USD of one call from its usage counts and a model's price rows (data/reference/anthropic_pricing_*.json):
    each request on its own, above 100,000 input tokens (cache included) the higher row; cache writes at the 5-minute
    price. 0 for a row without usage (a transport error is not billed)."""
    if not row.get("ok") or row.get("prompt_tokens") is None:
        return {"usd": 0.0, "by_kind": {}, "band": None} if by_kind else 0.0
    u = {"input": row.get("prompt_tokens") or 0, "cache_write": row.get("cache_creation_input_tokens") or 0,
         "cache_read": row.get("cache_read_input_tokens") or 0, "output": row.get("completion_tokens") or 0}
    over = u["input"] + u["cache_write"] + u["cache_read"] > 100000
    band = price["prompts_over_100000_tokens"] if over else price["prompts_up_to_100000_tokens"]
    rate = {"input": band["input"], "cache_write": band["cache_write_5m"], "cache_read": band["cache_read"], "output": band["output"]}
    parts = {k: u[k] * rate[k] / 1e6 for k in u}
    total = sum(parts.values())
    if by_kind:
        return {"usd": total, "by_kind": parts, "band": "over_100000" if over else "up_to_100000"}
    return total


# --- OpenAI Responses API (B-K2 pilot on GPT-6 Luna; the official SDK, its own retries off) ---

def openai_responses_body(model: str, messages: list, max_tokens: int, effort: str) -> dict:
    """The Responses API body, built in one place: the conversation's messages as they are (the system message stays a
    system message in input; nothing is moved or rewritten), reasoning effort explicit, no sampling parameters, no
    response format (the same plain-text answer as for every other model), store false (the response is not kept on
    the server), service_tier 'default' (Standard processing and price)."""
    return {"model": model, "input": [{"role": m["role"], "content": m["content"]} for m in messages],
            "max_output_tokens": max_tokens, "reasoning": {"effort": effort}, "store": False, "service_tier": "default"}


def _duration_s(value: str | None) -> float | None:
    """An OpenAI rate-limit reset ('1s', '6m0s', '20ms', '1h2m3.5s') in seconds; None if absent or not readable."""
    if not value:
        return None
    m = re.fullmatch(r"(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m(?!s))?(?:(\d+(?:\.\d+)?)s)?(?:(\d+(?:\.\d+)?)ms)?", value.strip())
    if not m or not any(m.groups()):
        return None
    h, mi, s, ms = (float(g) if g else 0.0 for g in m.groups())
    return h * 3600 + mi * 60 + s + ms / 1000


class OpenAIRateGate(RateGate):
    """RateGate for the OpenAI headers: x-ratelimit-remaining-{requests,tokens} and their resets as durations. The
    tokens limit counts input and output together. The limits themselves are never raised."""

    def __init__(self, concurrency: int, input_per_call: int = 10000, output_per_call: int = 2500):
        super().__init__(concurrency, input_per_call, output_per_call)
        self.need = {"requests": concurrency, "tokens": concurrency * (input_per_call + output_per_call)}

    def update(self, headers) -> None:
        snap, until, now = {}, 0.0, time.time()
        for kind, need in self.need.items():
            rem = headers.get(f"x-ratelimit-remaining-{kind}")
            reset = headers.get(f"x-ratelimit-reset-{kind}")
            snap[kind] = {"remaining": rem, "reset": reset}
            d = _duration_s(reset)
            if rem is not None and rem.isdigit() and int(rem) < need and d is not None:
                until = max(until, now + d)
        with self.lock:
            self.last = snap
            self.until = max(self.until, until)


def _redact(text: str) -> str:
    """An API error message as it may be stored: anything that looks like a key (sk-..., masked or not) removed."""
    return re.sub(r"sk-[^\s'\"]*", "sk-<redacted>", text or "")


def _retry_after(headers) -> float | None:
    ra, ms = headers.get("retry-after"), headers.get("retry-after-ms")
    if ms and ms.replace(".", "", 1).isdigit():
        return float(ms) / 1000
    if ra and ra.replace(".", "", 1).isdigit():
        return float(ra)
    return None


def chat_openai_responses(client, body: dict, timeout: float, gate: OpenAIRateGate | None = None) -> dict:
    """One call through client.responses.with_raw_response (the client's max_retries must be 0; the caller retries
    transport errors with one visible count). Returns the keys of chat_anthropic: finish_reason 'stop' (status
    completed), 'length' (incomplete: max_output_tokens), 'refusal' (a refusal item in the output, or incomplete:
    content_filter, its category then 'content_filter'); the usage counts (input including cache reads and writes,
    cached, cache written with its retention, output including reasoning); the request id, the response id, the model
    and service tier that served it. A 429 for an exhausted quota (insufficient_quota) is fatal, any other 429 is
    retried; 5xx, 408, 409, timeouts and connection errors are retried; any other 4xx is fatal (its message is not kept
    for 401 and 403, and anything that looks like a key is removed from any other). Any status other than completed
    (incomplete for another reason, cancelled, none) is finish_reason 'incomplete', never valid; a response without
    usage is a transport error (retried), as on the Messages API path."""
    import openai
    t0 = time.perf_counter()
    try:
        raw = client.responses.with_raw_response.create(**body, timeout=timeout)
        resp = raw.parse()
        if gate is not None:
            gate.update(raw.headers)
        texts, refusals = [], []
        for item in resp.output or []:
            if getattr(item, "type", None) != "message":
                continue
            for c in item.content or []:
                if getattr(c, "type", None) == "output_text":
                    texts.append(c.text)
                elif getattr(c, "type", None) == "refusal":
                    refusals.append(c.refusal)
        text = "".join(texts)
        status = resp.status
        reason = getattr(resp.incomplete_details, "reason", None) if resp.incomplete_details is not None else None
        if status == "failed":
            err = resp.error
            return {"ok": False, "latency": time.perf_counter() - t0, "error": f"response failed: {getattr(err, 'code', None)}",
                    "retryable": True, "detail": _redact(str(getattr(err, "message", "")))[:200]}
        u = resp.usage
        if u is None or getattr(u, "input_tokens", None) is None:  # as the Messages API path: never a free valid answer
            return {"ok": False, "latency": time.perf_counter() - t0, "error": "response without usage", "retryable": True}
        if refusals or reason == "content_filter":
            finish, category = "refusal", ("content_filter" if reason == "content_filter" else None)
        elif status == "incomplete" and reason == "max_output_tokens":
            finish, category = "length", None
        elif status == "completed":
            finish, category = "stop", None
        else:  # incomplete for another reason (max_messages, steered, none given), cancelled, no status: never valid
            finish, category = "incomplete", None
        itd = getattr(u, "input_tokens_details", None) if u is not None else None
        cw = getattr(itd, "cache_write_tokens_details", None) if itd is not None else None
        otd = getattr(u, "output_tokens_details", None) if u is not None else None
        return {"ok": True, "latency": time.perf_counter() - t0, "text": text, "empty_content": not text.strip(),
                "prompt_tokens": getattr(u, "input_tokens", None), "completion_tokens": getattr(u, "output_tokens", None),
                "cached_tokens": getattr(itd, "cached_tokens", None), "cache_write_tokens": getattr(itd, "cache_write_tokens", None),
                "cache_write_30m_tokens": getattr(cw, "ttl_30m_tokens", None), "cache_write_12h_tokens": getattr(cw, "ttl_12h_tokens", None),
                "reasoning_tokens": getattr(otd, "reasoning_tokens", None),
                "finish_reason": finish, "stop_reason": status, "incomplete_reason": reason, "refusal_category": category,
                "refusal_text": (" ".join(refusals))[:300] if refusals else None,
                "request_id": raw.headers.get("x-request-id"), "response_id": resp.id, "served_model": resp.model,
                "service_tier": resp.service_tier}
    except openai.RateLimitError as e:
        body = getattr(e, "body", None)
        code = (body.get("code") if isinstance(body, dict) else None) or getattr(e, "code", None)
        typ = (body.get("type") if isinstance(body, dict) else None) or getattr(e, "type", None)
        if "insufficient_quota" in (code, typ):
            return {"ok": False, "latency": time.perf_counter() - t0, "error": "HTTP 429 insufficient_quota", "retryable": False,
                    "fatal": True, "detail": "the account's quota or credit is exhausted"}
        return {"ok": False, "latency": time.perf_counter() - t0, "error": "HTTP 429", "retryable": True,
                "retry_after": _retry_after(e.response.headers)}
    except openai.APIStatusError as e:
        return {"ok": False, "latency": time.perf_counter() - t0, "error": f"HTTP {e.status_code}",
                "retryable": e.status_code >= 500 or e.status_code in (408, 409),
                "fatal": e.status_code < 500 and e.status_code not in (408, 409),
                "retry_after": _retry_after(e.response.headers),
                "detail": "" if e.status_code in (401, 403) else _redact(str(getattr(e, "message", "")))[:200]}
    except (openai.APITimeoutError, openai.APIConnectionError) as e:
        return {"ok": False, "latency": time.perf_counter() - t0, "error": type(e).__name__, "retryable": True}
    except Exception as e:  # anything else (a body that is not JSON, a response without usage): a transport error, never a crash
        return {"ok": False, "latency": time.perf_counter() - t0, "error": f"unexpected {type(e).__name__}", "retryable": True}


def openai_cost(row: dict, price: dict, by_kind: bool = False):
    """USD of one call from its usage counts and a model's price rows (data/reference/openai_pricing_*.json): input
    tokens neither read from nor written to the cache at the input price, cache reads at the cached-input price, cache
    writes at their retention's price (30 minutes when the split is not reported), output (reasoning included) at the
    output price; above 272,000 input tokens the long-context row for the whole request. 0 for a row without usage."""
    if not row.get("ok") or row.get("prompt_tokens") is None:
        return {"usd": 0.0, "by_kind": {}, "band": None} if by_kind else 0.0
    total_in = row.get("prompt_tokens") or 0
    read, write = row.get("cached_tokens") or 0, row.get("cache_write_tokens") or 0
    w12 = row.get("cache_write_12h_tokens") or 0
    w30 = row.get("cache_write_30m_tokens")
    w30 = (write - w12) if w30 is None else w30
    u = {"input": max(0, total_in - read - write), "cache_write": w30, "cache_write_12h": w12, "cache_read": read,
         "output": row.get("completion_tokens") or 0}
    long = total_in > 272000
    band = price["long_context"] if long else price["short_context"]
    rate = {"input": band["input"], "cache_write": band["cache_write_30m"], "cache_write_12h": band["cache_write_12h"],
            "cache_read": band["cached_input"], "output": band["output"]}
    parts = {k: u[k] * rate[k] / 1e6 for k in u}
    total = sum(parts.values())
    if by_kind:
        return {"usd": total, "by_kind": parts, "band": "long_context" if long else "short_context"}
    return total
