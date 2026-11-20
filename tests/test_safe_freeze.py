"""Freezing after the third pilot (the project lead's decision of 19.11.2026): the H1-family qualification recomputes
every bound of the calibration report and keeps its flag; the frozen-settings candidate takes the runs' settings, the
reported model identifiers and K from the pilot's reports and refuses inconsistent inputs."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P1 = ROOT / "data/processed/safe_dev_pilot3_v0.1.json"
P2 = ROOT / "data/processed/safe_dev_pilot3_calibration_v0.1.json"


def run(script, *x):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / script), *map(str, x)], capture_output=True, text=True, timeout=120)


def test_h1_qualification_and_frozen_candidate(tmp_path):
    q = tmp_path / "q.json"
    p = run("safe_h1_qualification.py", "--calibration", P2, "--out", q)
    assert p.returncode == 0, p.stderr
    r = json.loads(q.read_text())
    assert r["h1_family_qualified"] and r["h1_bounds"] == r["h1_bounds_within_levels"] == 18
    assert r["calibration_ready_for_freezing_as_written"] is False and len(r["h3"]["bounds_over_their_level"]) == 4
    assert "not a calibration report" in run("safe_h1_qualification.py", "--calibration", P1, "--out", tmp_path / "x.json").stderr
    f = tmp_path / "frozen.json"
    p = run("safe_make_frozen.py", "--first", P1, "--calibration", P2, "--qualification", q, "--test-seed", 20261121, "--out", f)
    assert p.returncode == 0, p.stderr
    fr = json.loads(f.read_text())
    assert {n: m["K"] for n, m in fr["models"].items()} == {"qwen": 1, "haiku": 1, "luna": 2}
    assert fr["models"]["qwen"]["model_digest"] and fr["models"]["haiku"]["served_model"] == "claude-haiku-5-5"
    assert fr["build"]["elicitation"] == "counts_nozero" and fr["build"]["draws_of_full_and_segment"] == 2
    assert fr["status"].startswith("incomplete") and fr["test_build"] is None  # no test build yet
    assert "not a first report" in run("safe_make_frozen.py", "--first", P2, "--calibration", P2, "--qualification", q,
                                       "--test-seed", 1, "--out", tmp_path / "y.json").stderr
    bad = json.loads(q.read_text())
    bad["h1_family_qualified"] = False
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    assert "not qualified" in run("safe_make_frozen.py", "--first", P1, "--calibration", P2, "--qualification", tmp_path / "bad.json",
                                  "--test-seed", 1, "--out", tmp_path / "z.json").stderr


def test_qualification_refuses_a_changed_grid(tmp_path):
    """The project lead's counterexamples (19.11.2026): a correction over 18 bounds, three copies of the 0.05 level for H1,
    a summary count that differs from the details - each refused."""
    base = json.loads(P2.read_text())

    def refused(mut, msg):
        r = json.loads(json.dumps(base))
        mut(r)
        p_ = tmp_path / "c.json"
        p_.write_text(json.dumps(r))
        out = run("safe_h1_qualification.py", "--calibration", p_, "--out", tmp_path / "o.json")
        assert out.returncode != 0 and msg in out.stderr, out.stderr

    def eighteen(r):
        lc = r["level_check"]
        lc["checks"] = [c for c in lc["checks"] if "/H1/" in c["check"]]
        lc["number_of_bounds"] = lc["expected_bounds"] = 18
        lc["bound_quantile_each"] = 1 - 0.05 / 18
    refused(eighteen, "declared grid")

    def same_level(r):
        for c in r["level_check"]["checks"]:
            if "/H1/" in c["check"]:
                c["level"] = 0.05
    refused(same_level, "twice")

    def count(r):
        r["level_check"]["checks"][0]["claim_tail_rejections"] += 1
    refused(count, "differ from the details")

    def conf(r):
        r["level_check"]["simultaneous_confidence"] = 0.9
    refused(conf, "0.95")


def test_frozen_binds_the_test_build(tmp_path):
    """--test-build writes the test set's sha256 into the frozen file; a synthetic or partial set is refused."""
    import gzip
    import hashlib
    q = tmp_path / "q.json"
    assert run("safe_h1_qualification.py", "--calibration", P2, "--out", q).returncode == 0
    first = json.loads(P1.read_text())
    tb = tmp_path / "tb"
    tb.mkdir()
    items = b'{"item": "x", "kind": "safe", "messages": []}\n'
    (tb / "items.jsonl.gz").write_bytes(gzip.compress(items))
    (tb / "key.json").write_text("{}")
    sys.path.insert(0, str(ROOT / "scripts"))
    import safe_prompt as P
    design = {"split": "test", "elicitation": "counts_nozero", "parser": "safe_counts", "seed": 20261121, "repeats": 1,
              "sample_per_size": None, "sample_by_size": None, "only_repeat_firms_of_an_earlier_build": None,
              "card_key_sha256": hashlib.sha256(P.KEY_PATH.read_bytes()).hexdigest(),
              "prompt_sha256": hashlib.sha256(Path(P.__file__).read_bytes()).hexdigest(),
              "source_sha256": first["source_sha256"], "records": 692, "repeat_firms": 692, "items": 1,
              "items_sha256": hashlib.sha256(items).hexdigest(), "key_json_sha256": hashlib.sha256(b"{}").hexdigest(),
              "target_answers_in_split": {"1-10": 692, "99": 30}, "excluded_firms_from_earlier_builds": 0}
    (tb / "design.json").write_text(json.dumps(design))
    mk = lambda out, b: run("safe_make_frozen.py", "--first", P1, "--calibration", P2, "--qualification", q,  # noqa: E731
                            "--test-seed", 20261121, "--test-build", b, "--out", out)
    p = mk(tmp_path / "f.json", tb)
    assert p.returncode == 0, p.stderr
    fr = json.loads((tmp_path / "f.json").read_text())
    assert fr["test_build"]["items_sha256"] == design["items_sha256"] and "no serving stack" in fr["status"]
    assert all(m["config"]["script"] == "0.9.5" for m in fr["models"].values()) and "0.9.4" in fr["runner_note"]
    assert {"scripts/llm_serving_check.py", "scripts/ollama_provenance.py", "data/reference/safe_validation_plan.json"} <= set(fr["materials_sha256"])
    assert json.loads(run("safe_make_frozen.py", "--first", P1, "--calibration", P2, "--qualification", q, "--test-seed",
                          20261121, "--out", tmp_path / "g.json").stdout or "{}") is not None
    assert json.loads((tmp_path / "g.json").read_text())["status"].startswith("incomplete")
    p = run("safe_make_frozen.py", "--first", P1, "--calibration", P2, "--qualification", q, "--test-seed", 20261121,
            "--test-build", tb, "--approved-on", "2026-11-19", "--out", tmp_path / "a.json")
    assert p.returncode != 0 and "not complete" in p.stderr  # no pilot runs: no approval
    for change in ({"source_sha256": "0" * 64}, {"records": 4}, {"sample_per_size": 2}, {"repeat_firms": 40},
                   {"excluded_firms_from_earlier_builds": 3}, {"target_answers_in_split": {"1-10": 700}}):
        (tb / "design.json").write_text(json.dumps({**design, **change}))
        p = mk(tmp_path / "h.json", tb)
        assert p.returncode != 0 and "--test-build" in p.stderr, (change, p.stderr)


def test_pilot_server_and_the_runner_lists(tmp_path, monkeypatch):
    """--pilot-run: the serving stack of a pilot run is taken only from the run of the report, all sessions alike; the
    runner's and the analysis' lists of the runner modules are the same."""
    import re
    sys.path.insert(0, str(ROOT / "scripts"))
    import safe_analyze as A
    import safe_make_frozen as M
    rb = (ROOT / "scripts/run_b_k2.py").read_text()
    assert eval(re.search(r"^RUN_MATERIALS = (\(.*?\))", rb, re.M).group(1)) == A.RUN_MATERIALS
    d = tmp_path / "run"
    d.mkdir()
    (d / "answers.jsonl").write_text("{}\n")
    srv = {"anthropic_sdk": "1.0", "anthropic_version": "2023-06-01", "model": {"id": "m", "created_at": "2026-01-01"}}
    (d / "run.json").write_text(json.dumps({"config": {"items_sha256": "abc"}, "sessions": [{"server": srv}, {"server": srv}]}))
    import hashlib
    ri = {"items_sha256": "abc", "answers_sha256": hashlib.sha256(b"{}\n").hexdigest()}
    got = M.pilot_server(d, ri)
    assert got["anthropic_sdk"] == "1.0" and got["model_created"] == "2026-01-01"
    import pytest
    with pytest.raises(SystemExit):
        M.pilot_server(d, {**ri, "answers_sha256": "0" * 64})
    (d / "run.json").write_text(json.dumps({"config": {"items_sha256": "abc"},
                                            "sessions": [{"server": srv}, {"server": {**srv, "anthropic_sdk": "2.0"}}]}))
    with pytest.raises(SystemExit):
        M.pilot_server(d, ri)


def test_frozen_arguments_and_preflight(tmp_path):
    """--pilot-run with a name outside the family and --approved-on without a date are refused (code review 12); the
    preflight refuses a frozen file without a test build and a run off the frozen settings or stack."""
    q = tmp_path / "q.json"
    assert run("safe_h1_qualification.py", "--calibration", P2, "--out", q).returncode == 0
    base = ["--first", P1, "--calibration", P2, "--qualification", q, "--test-seed", 20261121]
    assert "NAME one of" in run("safe_make_frozen.py", *base, "--pilot-run", "qwn=x", "--out", tmp_path / "a.json").stderr
    assert "YYYY-MM-DD" in run("safe_make_frozen.py", *base, "--approved-on", "jutro", "--out", tmp_path / "b.json").stderr
    fz = tmp_path / "f.json"
    assert run("safe_make_frozen.py", *base, "--out", fz).returncode == 0
    assert "no test_build" in run("safe_preflight.py", "--frozen", fz, "--run", f"haiku={tmp_path}").stderr
    sys.path.insert(0, str(ROOT / "scripts"))
    import safe_analyze as A
    fr = json.loads(fz.read_text())
    fr["test_build"] = {"items_sha256": "abc"}
    fr["models"]["haiku"]["server"] = A.server_identity({"anthropic_sdk": "1.0"})
    fz.write_text(json.dumps(fr))
    d = tmp_path / "pre"
    d.mkdir()
    cfg = {**fr["models"]["haiku"]["config"], "items_sha256": "abc"}
    mats = {f_: fr["materials_sha256"][f_] for f_ in A.RUN_MATERIALS}
    sess = {"materials_sha256": mats, "server": {"anthropic_sdk": "1.0"}}
    (d / "answers.jsonl").write_text(json.dumps({"item": "i", "served_model": "claude-haiku-5-5"}) + "\n")
    (d / "run.json").write_text(json.dumps({"config": cfg, "sessions": [sess]}))
    p = run("safe_preflight.py", "--frozen", fz, "--run", f"haiku={d}")
    assert p.returncode == 0 and "match the frozen ones" in p.stdout, p.stderr
    (d / "run.json").write_text(json.dumps({"config": cfg, "sessions": [{**sess, "server": {"anthropic_sdk": "2.0"}}]}))
    assert "another serving stack" in run("safe_preflight.py", "--frozen", fz, "--run", f"haiku={d}").stderr
    (d / "run.json").write_text(json.dumps({"config": {**cfg, "max_tokens": 1}, "sessions": [sess]}))
    assert "settings differ" in run("safe_preflight.py", "--frozen", fz, "--run", f"haiku={d}").stderr
