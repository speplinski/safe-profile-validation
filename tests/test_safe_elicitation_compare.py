"""The elicitation comparison A/B/C (scripts/safe_elicitation_compare.py, plan elicitation_comparison): the build holds
every firm x card x arm with the second pilot's cards, the arm's prompt and parser; the runner reads the parser per
item; the selection follows the rule fixed in the plan (S with equal weights, errors as the marginal, eligibility at
5%, ties broken by the smallest change); runs off the second pilot's settings are refused."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))
from test_safe_analyze import _csv, _run  # noqa: E402
import safe_prompt as P  # noqa: E402


def _builds(tmp_path):
    src = _csv(tmp_path / "s.csv.gz")
    run = lambda *x: subprocess.run([sys.executable, str(ROOT / "scripts/safe_build.py"), "--safe", str(src), *map(str, x)],  # noqa: E731
                                    capture_output=True, text=True)
    p = run("--split", "dev", "--sample", "10", "--repeats", "2", "--repeat-firms", "40", "--out", tmp_path / "ref")
    assert p.returncode == 0, p.stderr
    assert "needs --only-repeat-firms-of" in run("--split", "dev", "--arms", "A", "B", "C", "--variants", "full",
                                                  "--out", tmp_path / "x").stderr
    p = run("--split", "dev", "--only-repeat-firms-of", tmp_path / "ref/key.json", "--seed", "20261118", "--arms", "A", "B", "C",
            "--variants", "full", "segment", "--out", tmp_path / "abc")
    assert p.returncode == 0, p.stderr
    return src, run


def _cmp(tmp_path, src, runs, refs, out="c.json"):
    x = [sys.executable, str(ROOT / "scripts/safe_elicitation_compare.py"), "--build", str(tmp_path / "abc"),
         "--reference-build", str(tmp_path / "ref"), "--safe", str(src), "--out", str(tmp_path / out)]
    for n, d in runs.items():
        x += ["--run", f"{n}={d}"]
    for n, d in refs.items():
        x += ["--reference-run", f"{n}={d}"]
    return subprocess.run(x, capture_output=True, text=True, timeout=600)


def test_reference_with_partial_repeats(tmp_path):
    """--only-repeat-firms-of keeps the reference's repeated firms only: a reference with repeats for part of its firms
    is accepted (the firm check compares with the repeated firms, not all of them)."""
    src = _csv(tmp_path / "s.csv.gz")
    run = lambda *x: subprocess.run([sys.executable, str(ROOT / "scripts/safe_build.py"), "--safe", str(src), *map(str, x)],  # noqa: E731
                                    capture_output=True, text=True)
    assert run("--split", "dev", "--sample", "10", "--repeats", "1", "--repeat-firms", "20", "--out", tmp_path / "ref").returncode == 0
    p = run("--split", "dev", "--only-repeat-firms-of", tmp_path / "ref/key.json", "--seed", "20261118", "--arms", "A", "B", "C",
            "--variants", "full", "segment", "--out", tmp_path / "abc")
    assert p.returncode == 0, p.stderr
    design, key = json.loads((tmp_path / "abc/design.json").read_text()), json.loads((tmp_path / "abc/key.json").read_text())
    rd, rk = json.loads((tmp_path / "ref/design.json").read_text()), json.loads((tmp_path / "ref/key.json").read_text())
    assert design["records"] == 20 and len({(k["wave"], k["permid"]) for k in rk.values()}) == 40
    runs, refs = {}, {}
    for n in ("m1", "m2", "m3"):
        _run(tmp_path / f"ref_{n}", rd, rk, "flat", model=n)
        _run(tmp_path / f"abc_{n}", design, key, "flat", model=n, good_arms={"A", "B", "C"})
        runs[n], refs[n] = tmp_path / f"abc_{n}", tmp_path / f"ref_{n}"
    p = _cmp(tmp_path, src, runs, refs)
    assert p.returncode == 0, p.stderr
    assert json.loads((tmp_path / "c.json").read_text())["firms"] == 20


def test_build_prompts_and_selection(tmp_path):
    src, run = _builds(tmp_path)
    design = json.loads((tmp_path / "abc/design.json").read_text())
    key = json.loads((tmp_path / "abc/key.json").read_text())
    assert design["parser"] == "safe_by_item" and design["items"] == 40 * 2 * 3
    a = P.system_prompt(P.load_key(), "counts")
    b = P.system_prompt(P.load_key(), "counts_nozero")
    assert a.replace(P.ZERO_RULE, P.ZERO_RULE_PILOT1) == b and a != b  # B differs from A in the zero rule only
    rd, rk = json.loads((tmp_path / "ref/design.json").read_text()), json.loads((tmp_path / "ref/key.json").read_text())
    refs, runs = {}, {}
    for n in ("m1", "m2", "m3"):
        _run(tmp_path / f"ref_{n}", rd, rk, "flat", model=n)
        refs[n] = tmp_path / f"ref_{n}"
        _run(tmp_path / f"abc_{n}", design, key, "flat", model=n, good_arms={"C"})
        runs[n] = tmp_path / f"abc_{n}"
    p = _cmp(tmp_path, src, runs, refs)
    assert p.returncode == 0, p.stderr
    sel = json.loads((tmp_path / "c.json").read_text())["selection"]
    assert sel["chosen"] == "C" and sel["remaining"] == ["C"] and set(sel["excluded_by_a_demonstrated_advantage"]) == {"A", "B"}
    assert sel["pairwise"]["A-C"]["demonstrated_advantage"] == "C" and sel["pairwise"]["A-B"]["demonstrated_advantage"] is None
    # equal arms: no demonstrated difference, the conservative choice is the smallest change (A)
    for n in runs:
        (tmp_path / f"abc_{n}").rename(tmp_path / f"old_{n}")
        _run(tmp_path / f"abc_{n}", design, key, "flat", model=n, good_arms={"A", "B", "C"})
    p = _cmp(tmp_path, src, runs, refs, "c2.json")
    assert p.returncode == 0, p.stderr
    sel = json.loads((tmp_path / "c2.json").read_text())["selection"]
    assert sel["chosen"] == "A" and sel["remaining"] == ["A", "B", "C"] and not sel["excluded_by_a_demonstrated_advantage"]
    # an arm with too many technical errors is not eligible, even when it is the best on its valid answers
    for n in runs:
        (tmp_path / f"abc_{n}").rename(tmp_path / f"old2_{n}")
        _run(tmp_path / f"abc_{n}", design, key, "flat", model=n, good_arms={"B"}, fail_arm="B")
    p = _cmp(tmp_path, src, runs, refs, "c3.json")
    assert p.returncode == 0, p.stderr
    rep = json.loads((tmp_path / "c3.json").read_text())
    assert "B" not in rep["selection"]["eligible"] and rep["selection"]["chosen"] == "A"
    assert rep["selection"]["pairwise"]["A-B"]["demonstrated_advantage"] == "B" and not rep["selection"]["excluded_by_a_demonstrated_advantage"]
    assert rep["models"]["m1"]["cards"]["full"]["B"]["failed_model_answers_by_class"] == {"all counts 0": rep["models"]["m1"]["cards"]["full"]["B"]["failed_model_answers_by_class"]["all counts 0"]}
    # every arm over the error limit: "no eligible arm", no automatic choice (project lead, 18.11.2026)
    for n in runs:
        (tmp_path / f"abc_{n}").rename(tmp_path / f"old3_{n}")
        _run(tmp_path / f"abc_{n}", design, key, "flat", model=n, fail_arm="*")
    p = _cmp(tmp_path, src, runs, refs, "c6.json")
    assert p.returncode == 0, p.stderr
    sel = json.loads((tmp_path / "c6.json").read_text())["selection"]
    assert sel["eligible"] == [] and sel["chosen"] is None and sel["result"] == "no eligible arm"
    # one arm with only errors in one cell: the report is written, the empty cell is named (code review 8)
    for n in runs:
        (tmp_path / f"abc_{n}").rename(tmp_path / f"old4_{n}")
        _run(tmp_path / f"abc_{n}", design, key, "flat", model=n, good_arms={"A", "B", "C"})
    ans = tmp_path / "abc_m1/answers.jsonl"
    rows = [json.loads(x) for x in ans.read_text().splitlines()]
    bad_items = {i for i, k in key.items() if k["arm"] == "C" and k["variant"] == "full"}
    out_rows = []
    for r in rows:
        if r["item"] in bad_items:
            out_rows += [{"item": r["item"], "ok": True, "valid": False, "outcome": "invalid_format", "error": "x", "text": ""}] * 2
        else:
            out_rows.append(r)
    ans.write_text("\n".join(json.dumps(r) for r in out_rows) + "\n")
    p = _cmp(tmp_path, src, runs, refs, "c7.json")
    assert p.returncode == 0, p.stderr
    cc = json.loads((tmp_path / "c7.json").read_text())["selection"]
    assert cc["S_complete_cases_common_to_the_three_arms"]["cells_without_common_firms"] == ["m1/full"] and "C" not in cc["eligible"]
    # the same run three times, or the same directory: refused
    p = _cmp(tmp_path, src, {n: runs["m2"] for n in runs}, refs, "c8.json")
    assert p.returncode != 0 and "same directory" in p.stderr
    # a run off the second pilot's settings is refused
    rj = json.loads((tmp_path / "abc_m1/run.json").read_text())
    rj["config"]["max_tokens"] = 7
    (tmp_path / "abc_m1/run.json").write_text(json.dumps(rj))
    p = _cmp(tmp_path, src, runs, refs, "c4.json")
    assert p.returncode != 0 and "settings differ" in p.stderr
    # two models only: not the family
    p = _cmp(tmp_path, src, {k: v for k, v in runs.items() if k != "m3"}, {k: v for k, v in refs.items() if k != "m3"}, "c5.json")
    assert p.returncode != 0 and "models of the family" in p.stderr
    # another true answer, or another SAFE extract, than the reference build: refused (project lead, 19.11.2026)
    import hashlib as _h0
    kb = tmp_path / "abc/key.json"
    good_kb, good_db = kb.read_text(), (tmp_path / "abc/design.json").read_text()
    k2 = json.loads(good_kb)
    f0 = next(iter(k2.values()))
    for r_ in k2.values():
        if (r_["wave"], r_["permid"]) == (f0["wave"], f0["permid"]):
            r_["answer"] = "1" if r_["answer"] != "1" else "2"
    kb.write_text(json.dumps(k2))
    (tmp_path / "abc/design.json").write_text(json.dumps({**json.loads(good_db), "key_json_sha256": _h0.sha256(kb.read_bytes()).hexdigest()}))
    p = _cmp(tmp_path, src, runs, refs, "c10.json")
    assert p.returncode != 0 and "another true answer" in p.stderr
    kb.write_text(good_kb)
    (tmp_path / "abc/design.json").write_text(json.dumps({**json.loads(good_db), "source_sha256": "0" * 64}))
    p = _cmp(tmp_path, src, runs, refs, "c11.json")
    assert p.returncode != 0 and "same SAFE extract" in p.stderr
    (tmp_path / "abc/design.json").write_text(good_db)
    # a reference build with another system prompt than arm A's is refused
    import gzip as _gz
    import hashlib as _h
    rows = [json.loads(x) for x in _gz.decompress((tmp_path / "ref/items.jsonl.gz").read_bytes()).decode().splitlines()]
    rows[0]["messages"][0]["content"] += " "
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()
    (tmp_path / "ref/items.jsonl.gz").write_bytes(_gz.compress(text))
    rd2 = {**rd, "items_sha256": _h.sha256(text).hexdigest()}
    (tmp_path / "ref/design.json").write_text(json.dumps(rd2))
    p = _cmp(tmp_path, src, runs, refs, "c9.json")
    assert p.returncode != 0 and "not arm A's" in p.stderr
    # the main analysis refuses the comparison build
    q = subprocess.run([sys.executable, str(ROOT / "scripts/safe_analyze.py"), "--build", str(tmp_path / "abc"), "--run",
                        f"m1={runs['m2']}", "--safe", str(src), "--out", str(tmp_path / "x.json")], capture_output=True, text=True)
    assert q.returncode != 0 and "safe_elicitation_compare" in q.stderr
