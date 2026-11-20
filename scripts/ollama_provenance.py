#!/usr/bin/env python3
"""What actually ran in a local Ollama model run: model digest, runner, KV cache precision, context.

Sources, each recorded with where it came from:
- the Ollama API: server version (/api/version), the model's full digest, format and quantization
  (/api/tags, /api/show), the model loaded during the run with its context length (/api/ps);
- the model's manifest on disk: the digest of the weights layer (the blob the runner loads);
- the server log (~/.ollama/logs/server.log): the runner actually started for this model ("llama-server"
  with its command line, or the MLX runner), its flags (--cache-type-k/-v, --flash-attn, -c, -np, ...),
  and the server configuration (OLLAMA_KV_CACHE_TYPE, OLLAMA_NUM_PARALLEL, ...) in force when it started.
A setting the log does not show for a runner (e.g. KV cache type for MLX) is recorded as not shown.

Used by run_k3_agent.py; on its own it writes provenance.json for an earlier run directory:
  python3 scripts/ollama_provenance.py --model qwen3.8:27b --run-dir data/processed/agent_runs/<dir>
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shlex
import urllib.request
from pathlib import Path

VERSION = "0.3.1"
DEFAULT_LOG = Path(os.environ.get("OLLAMA_SERVER_LOG") or Path.home() / ".ollama/logs/server.log")
SERVER_KEYS = ("OLLAMA_KV_CACHE_TYPE", "OLLAMA_FLASH_ATTENTION", "OLLAMA_NUM_PARALLEL", "OLLAMA_CONTEXT_LENGTH",
               "OLLAMA_MAX_LOADED_MODELS", "OLLAMA_KEEP_ALIVE", "OLLAMA_MODELS", "OLLAMA_LLM_LIBRARY")
LLAMA_FLAGS = {"--cache-type-k": "kv_cache_type_k", "--cache-type-v": "kv_cache_type_v", "--flash-attn": "flash_attn",
               "-c": "context_total", "-np": "parallel_slots", "-b": "batch", "-ub": "ubatch",
               "--chat-template": "chat_template", "--keep": "keep"}
LLAMA_SWITCHES = {"--context-shift": "context_shift", "--no-jinja": "no_jinja", "--jinja": "jinja"}


def mask_home(x):
    """The user's home directory written as '~' in every string of a record (no user names in what goes to git)."""
    home = os.path.expanduser("~").rstrip("/")
    if isinstance(x, str):
        return x.replace(home + "/", "~/") if home and home != "~" else x
    if isinstance(x, dict):
        return {k: mask_home(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [mask_home(v) for v in x]
    return x


def _root(base_url: str) -> str:
    return re.sub(r"/v1/?$", "", base_url.rstrip("/"))


def _get(url: str, body: dict | None = None, timeout: float = 10):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _full_name(model: str) -> str:
    return model if ":" in model else model + ":latest"


def api_facts(base_url: str, model: str) -> dict:
    """Server version, the model's tag entry (full digest, size, details) and /api/show without the long texts."""
    root, out = _root(base_url), {}
    try:
        out["ollama_version"] = _get(root + "/api/version").get("version")
    except Exception as e:
        out["ollama_version_error"] = type(e).__name__
    try:
        tags = _get(root + "/api/tags").get("models", [])
        entry = next((m for m in tags if _full_name(m.get("name", "")) == _full_name(model)), None)
        out["model_entry"] = entry
        out["model_digest"] = (entry or {}).get("digest")
    except Exception as e:
        out["tags_error"] = type(e).__name__
    try:
        show = _get(root + "/api/show", {"model": model})
        frm = re.search(r"^FROM\s+\S*?(sha256[-:][0-9a-f]{64})", show.get("modelfile") or "", flags=re.M)
        out["modelfile_weights_digest"] = frm.group(1).replace("sha256-", "sha256:") if frm else None
        out["show"] = {"details": show.get("details"), "parameters": show.get("parameters"),
                       "capabilities": show.get("capabilities"), "modified_at": show.get("modified_at"),
                       "template_sha256": hashlib.sha256((show.get("template") or "").encode()).hexdigest(),
                       "model_info_keys": len(show.get("model_info") or {})}
    except Exception as e:
        out["show_error"] = type(e).__name__
    return out


def loaded(base_url: str, model: str) -> dict | None:
    """The model as /api/ps shows it now (size in VRAM, context length when the server reports it)."""
    try:
        ps = _get(_root(base_url) + "/api/ps").get("models", [])
        return next((m for m in ps if _full_name(m.get("name", "")) == _full_name(model)), None)
    except Exception as e:
        return {"error": type(e).__name__}


def manifest(model: str, models_dir: Path | None = None) -> dict:
    """The model's manifest on disk and the digest of its weights layer."""
    models_dir = Path(models_dir or os.environ.get("OLLAMA_MODELS") or Path.home() / ".ollama/models")
    name, tag = _full_name(model).split(":", 1)
    parts = name.split("/")
    path = models_dir / "manifests" / ("registry.ollama.ai" if len(parts) < 3 else parts[0]) / \
        ("library" if len(parts) == 1 else parts[-2]) / parts[-1] / tag
    if not path.exists():
        return {"path": str(path), "found": False}
    raw = path.read_bytes()
    m = json.loads(raw)
    layers = [{"mediaType": x.get("mediaType"), "digest": x.get("digest"), "size": x.get("size")} for x in m.get("layers", [])]
    weights = next((x["digest"] for x in layers if (x["mediaType"] or "").endswith(".model")), None)
    return {"path": str(path), "found": True, "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "weights_digest": weights, "layers": layers}


def _time(line: str):
    m = re.match(r"time=(\S+)", line)
    try:
        return dt.datetime.fromisoformat(m.group(1)) if m else None
    except ValueError:
        return None


def _env(line: str) -> dict:
    """The OLLAMA_* settings of a "server config" line; values may contain spaces (e.g. a models path)."""
    m = re.search(r'env="map\[(.*)\]"', line)
    if not m:
        return {}
    body = m.group(1)
    pairs = re.findall(r"(?:^|\s)([A-Za-z_]+):(.*?)(?=\s[A-Za-z_]+:|$)", body)
    return {k: v.strip() for k, v in pairs if k in SERVER_KEYS}


def _llama_flags(cmd: str) -> dict:
    try:
        argv = shlex.split(cmd)
    except ValueError:
        argv = cmd.split()
    out = {}
    for i, a in enumerate(argv):
        if a in LLAMA_FLAGS and i + 1 < len(argv):
            out[LLAMA_FLAGS[a]] = argv[i + 1]
        elif a in LLAMA_SWITCHES:
            out[LLAMA_SWITCHES[a]] = True
    m = re.search(r"--model\s+\"?[^\n]*?sha256-([0-9a-f]{64})", cmd)
    if m:
        out["weights_blob"] = "sha256:" + m.group(1)
    if "context_total" in out and "parallel_slots" in out:
        try:
            out["context_per_slot"] = int(out["context_total"]) // int(out["parallel_slots"])
        except ValueError:
            pass
    return out


def log_files(log: Path) -> list[Path]:
    """The server log and its rotated copies (server-1.log, ...), oldest first."""
    log = Path(log)
    files = [p for p in log.parent.glob(log.stem + "*" + log.suffix) if p.is_file()] if log.parent.exists() else []
    return sorted(files, key=lambda p: p.stat().st_mtime)


def runner_starts(log: Path, model: str, weights_digest: str | None) -> list[dict]:
    """Every start of a runner for this model in the server logs, with the server config in force then.
    A runner start of a kind this parser does not know is recorded as unknown, never dropped."""
    lines = []
    for f in log_files(log):
        lines += [ln for ln in f.read_text(errors="replace").splitlines() if ln.startswith("time=")]
    lines = list(dict.fromkeys(lines))  # a copied log is read once
    lines.sort(key=lambda ln: _time(ln) or dt.datetime.min.replace(tzinfo=dt.timezone.utc))
    starts, config = [], {}
    for line in lines:
        t = _time(line)
        if 'msg="server config"' in line:  # the server (re)started: every runner before it is gone
            config = _env(line)
            starts.append({"time": t.isoformat() if t else None, "event": "server_start"})
        elif 'msg="starting llama-server"' in line:
            cmd = re.search(r'cmd="((?:[^"\\]|\\.)*)"', line)
            flags = _llama_flags(cmd.group(1)) if cmd else {}
            if weights_digest and flags.get("weights_blob") == weights_digest:
                starts.append({"time": t.isoformat() if t else None, "runner": "llama.cpp (llama-server)", "flags": flags,
                               "server_config": dict(config)})
        elif 'msg="stopping mlx runner subprocess"' in line or 'msg="stopping llama-server"' in line:
            starts.append({"time": t.isoformat() if t else None, "event": "runner_stop", "line": line[:300]})
        elif 'msg="starting mlx runner subprocess"' in line:
            m = re.search(r"model=(\S+)", line)
            if m and _full_name(m.group(1)) == _full_name(model):
                starts.append({"time": t.isoformat() if t else None, "runner": "mlx", "flags": {},
                               "server_config": dict(config)})
        elif re.search(r'msg="starting [^"]*(runner|server)[^"]*"', line) and (
                _full_name(model) in line or (weights_digest and weights_digest.split(":")[-1] in line)):
            flags = {}
            for flag, key in (("--kv-cache-type", "kv_cache_type_k"), ("--ctx-size", "context_total"), ("--parallel", "parallel_slots")):
                m = re.search(re.escape(flag) + r"[ =](\S+)", line)
                if m:
                    flags[key] = m.group(1).strip('"')
            starts.append({"time": t.isoformat() if t else None, "runner": "unknown", "line": line[:500],
                           "flags": flags, "server_config": dict(config)})
    return starts


def _local(base_url: str) -> bool:
    return re.match(r"https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?(/|$)", base_url) is not None


def capture(base_url: str, model: str, start: dt.datetime | None = None, end: dt.datetime | None = None,
            log: Path = DEFAULT_LOG, loaded_at_start: bool | None = None) -> dict:
    """Everything above for the window [start, end]: the runner still running when the window opened (its
    last start at or before start, unless the server restarted or the runner stopped after it, or
    loaded_at_start says the model was not loaded), every start inside the window, and the runner active at its end."""
    api = api_facts(base_url, model)
    man = manifest(model)
    weights = man.get("weights_digest") or api.get("modelfile_weights_digest")
    events = runner_starts(Path(log), model, weights)
    at = lambda s: dt.datetime.fromisoformat(s["time"]) if s.get("time") else None  # noqa: E731
    starts = [s for s in events if "runner" in s]
    timed = [s for s in starts if at(s)]
    at_start = [s for s in timed if start and at(s) <= start]
    if at_start:
        t0 = at(at_start[-1])
        ended = any(e.get("event") in ("server_start", "runner_stop") and at(e) and t0 < at(e) <= start for e in events)
        if loaded_at_start is False or (ended and loaded_at_start is None):
            at_start = []
    in_window = [s for s in timed if start and at(s) > start and (not end or at(s) <= end)]
    before_end = [s for s in timed if not end or at(s) <= end]
    notes = []
    if not _local(base_url):
        notes.append("the server is not local: the log and manifest read here may not be the server's")
    if man.get("found") and api.get("model_digest") and man.get("manifest_sha256") != api["model_digest"]:
        notes.append("the local manifest does not match the server's model digest: another models directory?")
    if not before_end:
        notes.append("no runner start for this model found in the logs (other log directory, or not a local Ollama)")
    return {"script": VERSION, "captured_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "window": {"start": start.isoformat() if start else None, "end": end.isoformat() if end else None},
            "model": model, "api": api, "manifest": man, "weights_digest_used": weights, "loaded_now": loaded(base_url, model),
            "loaded_at_start": loaded_at_start,
            "logs": [str(p) for p in log_files(log)], "runner_at_start": at_start[-1] if at_start else None,
            "runner_starts_in_window": in_window, "runner_active": before_end[-1] if before_end else None,
            "runner_starts_total": len(starts), "notes": notes}


def _runtime_key(r: dict) -> dict:
    flags, cfg = r.get("flags") or {}, r.get("server_config") or {}
    mlx = r.get("runner") == "mlx"
    return {"runner": r.get("runner"),
            "kv_cache_type": (f"{flags['kv_cache_type_k']}/{flags.get('kv_cache_type_v')}" if "kv_cache_type_k" in flags
                              else ("not applied (MLX runner)" if mlx else None)),
            "flash_attn": flags.get("flash_attn") if not mlx else "not applied (MLX runner)",
            "context_per_slot": flags.get("context_per_slot"), "parallel_slots": flags.get("parallel_slots"),
            "server_num_parallel": cfg.get("OLLAMA_NUM_PARALLEL"), "server_context_length": cfg.get("OLLAMA_CONTEXT_LENGTH"),
            "server_kv_cache_type": cfg.get("OLLAMA_KV_CACHE_TYPE") if not mlx else "not applied (MLX runner)"}


def runtime_from(provs: list[dict], pinned: dict | None = None) -> dict:
    """The runtime of a run from all its provenance captures (one per invocation that decided episodes):
    every runner that served it (active at a window's start, started in a window); runtime_mixed when they
    differ, or when the model digest or Ollama version differ between captures or from run.json."""
    provs = [p for p in provs if p and not p.get("pending")]
    if not provs:
        return {}
    runners = []
    for p in provs:
        for r in ([p.get("runner_at_start")] if p.get("runner_at_start") else []) + list(p.get("runner_starts_in_window") or []):
            runners.append(r)
        if not p.get("runner_at_start") and not p.get("runner_starts_in_window") and p.get("runner_active"):
            runners.append(p["runner_active"])
    keys = [_runtime_key(r) for r in runners]
    distinct = [dict(t) for t in dict.fromkeys(tuple(sorted(k.items())) for k in keys)]
    all_digests = {(p.get("api") or {}).get("model_digest") for p in provs}
    all_versions = {(p.get("api") or {}).get("ollama_version") for p in provs}
    digests, versions = all_digests - {None}, all_versions - {None}
    pinned = pinned or {}
    mixed = len(distinct) > 1 or len(digests) > 1 or len(versions) > 1 \
        or (bool(digests) and pinned.get("model_digest") not in (None, *digests)) \
        or (bool(versions) and pinned.get("ollama_version") not in (None, *versions))
    unknown = None in all_digests or None in all_versions or any(not (p.get("runner_at_start") or p.get("runner_starts_in_window")
                                                                     or p.get("runner_active")) for p in provs)
    one = distinct[0] if len(distinct) == 1 else {}
    restarts = sum(len(p.get("runner_starts_in_window") or []) - (0 if p.get("runner_at_start") else 1)
                   for p in provs if p.get("runner_starts_in_window"))
    return {**one, "runtime_mixed": mixed, "runtime_unknown": unknown, "runtimes": distinct, "runner_restarts": max(0, restarts),
            "model_digest": next(iter(digests)) if len(digests) == 1 else sorted(digests) or None,
            "ollama_version": next(iter(versions)) if len(versions) == 1 else sorted(versions) or None,
            "captures": len(provs), "notes": sorted({n for p in provs for n in (p.get("notes") or [])})}


def _window_of(run: Path) -> tuple[dt.datetime | None, dt.datetime | None, str]:
    """The run window of an earlier run: the directory name gives the start, summary.date the end;
    file times only when those are missing (copies and checkouts change file times)."""
    def local(t: dt.datetime) -> dt.datetime:  # the local offset of that date (DST), unless already aware
        return t if t.tzinfo else t.astimezone()
    tz = dt.datetime.now().astimezone().tzinfo
    start = end = None
    m = re.search(r"_(\d{4}-\d{2}-\d{2})_(\d{2})(\d{2})(\d{2})?$", run.name)
    if m:
        start = local(dt.datetime.fromisoformat(f"{m.group(1)}T{m.group(2)}:{m.group(3)}:{m.group(4) or '00'}"))
    summ = run / "summary.json"
    if summ.exists():
        d = json.loads(summ.read_text(encoding="utf-8")).get("date")
        end = local(dt.datetime.fromisoformat(d)) if d else None
    if start and end:
        return start, end, "directory name and summary.date (local time)"
    dec = run / "decisions.jsonl"
    st = dec.stat() if dec.exists() else None
    if st:
        start = start or dt.datetime.fromtimestamp(getattr(st, "st_birthtime", st.st_mtime), tz)
        end = end or dt.datetime.fromtimestamp(st.st_mtime, tz)
    return start, end, "file times (unreliable after a copy or checkout)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--run-dir", required=True, help="an earlier run directory")
    ap.add_argument("--base-url", default="http://localhost:11434")
    ap.add_argument("--log", default=str(DEFAULT_LOG))
    ap.add_argument("--force", action="store_true", help="add a capture although provenance.json already has captures")
    a = ap.parse_args()
    run = Path(a.run_dir)
    if (run / "provenance.json").exists() and not a.force:
        raise SystemExit(f"{run} already has provenance.json (captures or a marker of the run); use --force to add a coarser capture")
    meta = {}
    for f in ("run.json", "summary.json"):
        if (run / f).exists():
            meta = meta or json.loads((run / f).read_text(encoding="utf-8"))
    if meta.get("model") and meta["model"] != a.model:
        raise SystemExit(f"{run} was run with {meta['model']}, not {a.model}")
    start, end, source = _window_of(run)
    prov = capture(a.base_url, a.model, start, end, Path(a.log))
    prov["window_source"] = source
    prov["note_after_the_run"] = "captured after the run: the model on disk and the logs must not have changed since"
    pf = run / "provenance.json"
    old = json.loads(pf.read_text(encoding="utf-8")) if pf.exists() else []
    provs = (old if isinstance(old, list) else [old]) + [prov]
    tmp = pf.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(provs, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, pf)
    rt = runtime_from(provs, json.loads((run / "run.json").read_text(encoding="utf-8")) if (run / "run.json").exists() else None)
    sf = run / "summary.json"
    if sf.exists():
        summ = json.loads(sf.read_text(encoding="utf-8"))
        summ["runtime"] = rt
        sf.write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: rt.get(k) for k in ("runner", "kv_cache_type", "context_per_slot", "parallel_slots",
                                             "server_num_parallel", "runtime_mixed", "runner_restarts", "model_digest",
                                             "ollama_version", "notes")}, ensure_ascii=False), "->", pf)


if __name__ == "__main__":
    main()
