"""SAFE validation analysis (scripts/safe_analyze.py): the floor, band scores, the sign-flip test, Holm, and an end-to-end
score of a build with two mock runs: a model that knows the answer under the full card wins, a flat one does not;
technical errors leave the common denominator and are counted; a run of another build is refused."""
import csv
import gzip
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import safe_analyze as A  # noqa: E402
import safe_signal_check as SC  # noqa: E402


def test_units():
    p = np.zeros((1, 10))
    p[0, 0] = 1.0
    s = A.scores(p, np.array([0]))
    assert s["band"][0] == pytest.approx(np.log(0.99 + 0.003))
    assert s["ten"][0] == pytest.approx(np.log(0.99 + 0.001))
    assert np.isfinite(A.scores(p, np.array([9]))["band"][0])
    assert A.flip_test(np.zeros(20)) == 1.0
    assert A.flip_test(np.ones(30)) < 0.001
    assert A.holm({"a": 0.01, "b": 0.02, "c": 0.04}) == {"a": True, "b": True, "c": True}
    assert A.holm({"a": 0.01, "b": 0.04, "c": 0.03}) == {"a": True, "b": False, "c": False}
    nan = float("nan")
    assert A.holm({"b": nan, "a": 0.001, "c": 0.01}) == A.holm({"a": 0.001, "c": 0.01, "b": nan}) == {"a": True, "c": True, "b": False}
    assert A.holm({"a": None, "b": 0.001, "c": 0.5}) == {"b": True, "c": False, "a": False}
    assert A.flip_test(np.array([])) is None and A.boot_ci(np.array([])) is None
    assert A.worst_band(np.array([0, 4]))[0] == pytest.approx(np.log(0.003))
    rng = np.random.default_rng(3)
    assert A.boot_t_test(rng.normal(0, 1, 200), 2000) > 0.01 and A.boot_t_test(rng.normal(0.5, 1, 200), 2000) < 0.001
    assert A.boot_t_test(np.zeros(5)) is None and A.boot_t_test(np.array([1.0])) is None
    assert A.boot_t_test(np.array([0.3, 0.3])) is None  # constant d: no test (code review round 4)
    assert A.boot_t_test(np.array([0.3, 0.5]), 2000) > 0.05  # two values: resamples with se* = 0 count as extreme
    ci = A.boot_t_ci(rng.normal(0.5, 1, 200), 2000)
    assert ci[0] < 0.5 < ci[1]
    # graph: alpha/3 on H1s, H1_i -> H3_i, H3_i -> other H1s by 1/2
    names = ["a", "b", "c"]
    W = {f"H1:{n}": 1 / 3 for n in names} | {f"H3:{n}": 0.0 for n in names}
    G = {(f"H1:{n}", f"H3:{n}"): 1.0 for n in names}
    G |= {(f"H3:{n}", f"H1:{m}"): 0.5 for n in names for m in names if m != n}
    P = {"H1:a": 0.01, "H1:b": 0.03, "H1:c": 0.5, "H3:a": 0.01, "H3:b": 1.0, "H3:c": 1.0}
    r = A.graph_procedure(P, W, G)
    # H1:a at 0.0167 -> H3:a gets 0.0167 -> rejected at 0.01 -> passes 1/2 each to b and c: b at 0.0167 + 0.0083 = 0.025
    assert r["H1:a"] and r["H3:a"] and not r["H1:b"] and not r["H1:c"]
    P["H1:b"] = 0.024
    assert A.graph_procedure(P, W, G)["H1:b"]
    P2 = {k: 1.0 for k in P} | {"H1:a": 0.02}
    assert not any(A.graph_procedure(P2, W, G).values())  # 0.02 > 0.05/3
    assert not A.graph_procedure({**P2, "H1:a": None}, W, G)["H1:a"]
    assert A.error_class("count of 3 is not an integer in [0, 100]: 2.5") == "a count not an integer in 0-100"
    assert A.error_class("not one JSON object: Extra data: line 1") == "not one JSON object" and A.error_class("x") == "other"
    tf = A.text_features('Oto odpowiedź: {"firms_per_100": {}} dziękuję')
    assert tf["leading_text"] and tf["trailing_text"] and tf["has_firms_per_100"] and not tf["starts_with_brace"]
    ds = A.dist_summary(np.array([-5.0] + [0.1] * 39))
    assert ds["worst_5pct_firms"] == 2 and ds["sum_of_worst_5pct"] == -4.9 and ds["share_below_minus_4"] == 0.025
    lc = A.calibration_check(rng.normal(0.3, 1, 300), 300, reps=60, inner=200)
    assert set(lc["claim_tail"]) == {"0.0167", "0.0250", "0.0500"} and 0 <= lc["two_sided_rate"]["0.0500"] <= 0.2
    # the criterion: simultaneous one-sided Clopper-Pearson bounds of the claim tail within every local level
    ok_ = {"reps": 4000, "inner": 10000, "claim_tail": {"0.0167": 30, "0.0250": 45, "0.0500": 95}}
    bad = {**ok_, "claim_tail": {"0.0167": 70, "0.0250": 80, "0.0500": 150}}
    full = lambda r3=ok_, ms=("a", "b", "c"): [(f"{m}/{c}", r3 if (m, c) == ("c", "H3/all_eligible") else ok_)  # noqa: E731
                                               for m in ms for c in A.LEVEL_CHECKS]
    K3 = {"a": 1, "b": 2, "c": 1}
    c1 = A.level_criterion(full(), ["a", "b", "c"], k_by_model=K3)
    assert c1["ready_for_freezing"] and c1["status"] == "calibration" and c1["number_of_bounds"] == c1["expected_bounds"] == 36
    assert c1["h1_family"]["qualified"] and c1["h1_family"]["bounds"] == 18
    assert not c1["warnings_alpha_over_3"] and c1["K_by_model"] == K3
    c2 = A.level_criterion(full(bad), ["a", "b", "c"], k_by_model=K3)
    assert not c2["ready_for_freezing"] and not c2["bounds_within_levels"] and c2["status"] == "calibration"
    assert c2["h1_family"]["qualified"]  # only an H3 bound over (c/H3/all_eligible): the H1 family qualifies, the flag stays false
    assert c2["warnings_alpha_over_3"] == ["c/H3/all_eligible"]
    c0 = A.level_criterion(full(), ["a", "b", "c"])  # no fixed K: bounds fine, not ready
    assert c0["bounds_within_levels"] and not c0["ready_for_freezing"] and "no fixed K" in c0["not_ready_because"][0]
    assert "not development" in A.level_criterion(full(), ["a", "b", "c"], k_by_model=K3, split="test")["not_ready_because"][0]
    one_inner = {**ok_, "reps": 400, "inner": 1, "claim_tail": {"0.0167": 0, "0.0250": 0, "0.0500": 0}}
    c8 = A.level_criterion([(lab, one_inner) for lab, _ in full()], ["a", "b", "c"], k_by_model=K3)
    assert c8["bounds_within_levels"] and not c8["ready_for_freezing"]  # the project lead's counterexample 1
    c9 = A.level_criterion([("a/H1/all_eligible", ok_)], ["a", "b", "c"], k_by_model=K3)
    assert c9["bounds_within_levels"] and not c9["ready_for_freezing"]  # counterexample 2: 3 bounds of 36
    # a quick technical check, or an incomplete one, is never "met" (project lead's review, 18.11.2026)
    quick = {**ok_, "reps": 20, "inner": 200, "claim_tail": {"0.0167": 0, "0.0250": 0, "0.0500": 0}}
    c3 = A.level_criterion([(lab, quick) for lab, _ in full()], ["a", "b", "c"], k_by_model=K3)
    assert not c3["ready_for_freezing"] and c3["status"] == "diagnostic" and "fewer than 4000" in c3["not_ready_because"][0]
    c4 = A.level_criterion(full(ms=("a", "b")), ["a", "b"], k_by_model=K3)
    assert not c4["ready_for_freezing"] and "models of the family missing" in c4["not_ready_because"][0]
    c5 = A.level_criterion([x for x in full() if x[0] != "b/H3/all_eligible"], ["a", "b", "c"], k_by_model=K3)
    assert not c5["ready_for_freezing"] and "b/H3/all_eligible" in c5["not_ready_because"][0]
    c6 = A.level_criterion([(lab, None if lab == "a/H1/all_eligible" else r_) for lab, r_ in full()], ["a", "b", "c"], k_by_model=K3)
    assert not c6["ready_for_freezing"] and "a/H1/all_eligible" in c6["not_ready_because"][0]
    from scipy.stats import beta
    assert c1["checks"][0]["simultaneous_upper"] == pytest.approx(beta.ppf(1 - 0.05 / 36, 31, 3970), abs=1e-5)
    c7 = A.level_criterion([(lab, {**r_, "replications_without_p": 4000}) for lab, r_ in full()], ["a", "b", "c"], k_by_model=K3)
    assert not c7["ready_for_freezing"] and "without a p value" in c7["not_ready_because"][0]
    assert A.calibration_check(np.full(40, 0.2), 50, reps=5, inner=50)["replications_without_p"] == 5  # constant d


def _csv(path):
    cols = sorted(set(SC.CARD) | {"permid", "wave", "q0b_3", "d1_rec", "d3_rec"})
    buf = io.StringIO()
    w = csv.DictWriter(buf, cols)
    w.writeheader()
    rng = np.random.default_rng(1)
    n = 0
    for wave in ("32", "36", "40"):
        for _ in range(120):
            n += 1
            r = {c: str(rng.choice(["1", "2", "7"])) for c in SC.CARD if c.startswith("q4")}
            r.update({c: str(rng.choice(["1", "3", ""])) for c in SC.CARD if c.startswith("q7a")})
            r.update({c: "" for c in SC.CARD if c.startswith("q7b")})
            r.update({"permid": str(n), "wave": wave, "d1_rec": str(rng.integers(1, 5)), "d3_rec": str(rng.integers(1, 5)),
                      "d4": "5", "d5_rec": "1", "d6": "5", "d7_rec": "1", "q0b_3": str(rng.integers(1, 11))})
            w.writerow(r)
    path.write_bytes(gzip.compress(buf.getvalue().encode()))
    return path


RUN_MATERIALS = ("scripts/run_b_k2.py", "scripts/llm_serving_check.py", "scripts/ollama_provenance.py", "scripts/safe_prompt.py")


def _run(dirp, design, key, mode, drop=0, finished=True, skip=0, model=None, bad_first_full=(), repeat_full_sum=None,
         served=None, good_arms=None, fail_arm=None):
    dirp.mkdir()
    sess = {"started_at": "t", "materials_sha256": {f: A.sha((ROOT / f).read_bytes()) for f in RUN_MATERIALS}}
    if finished:
        sess["finished_at"] = "t2"
    (dirp / "run.json").write_text(json.dumps({"config": {"items_sha256": design["items_sha256"], "model": model or mode, "api": "mock",
                                                          "parser": design["parser"], "max_attempts": 2,
                                                          "model_digest": "sha256:mock"},
                                               "sessions": [sess]}))
    lines, k = [], 0
    for iid, row in key.items():
        y = int(row["answer"])
        good = {str(i): (0.6 if i == y else 0.4 / 9) for i in range(1, 11)}
        worse = {str(i): (0.4 / 9 if i == y else 0.6 / 9 + 0.4 / 81) for i in range(1, 11)}
        p = {"oracle": good if row["variant"] == "full" else None, "harmful": worse if row["variant"] == "full" else None}.get(mode) \
            or {str(i): 0.1 for i in range(1, 11)}
        k += 1
        if skip and k % skip == 0:
            continue
        if drop and row["variant"] == "segment" and k % drop == 0:
            lines += [json.dumps({"item": iid, "ok": True, "valid": False, "outcome": "invalid_format"})] * 2
            continue
        if fail_arm and row["variant"] == "full" and ((fail_arm == "*" and int(row["permid"]) % 4 == 0)
                                                      or (row.get("arm") == fail_arm and k % 4 == 0)):
            lines += [json.dumps({"item": iid, "ok": True, "valid": False, "outcome": "invalid_format", "error": "all counts are 0",
                                  "text": '{"firms_per_100": {}}'})] * 2
            continue
        if good_arms is not None:
            p = good if (row.get("arm") in good_arms and row["variant"] == "full") else {str(i): 0.1 for i in range(1, 11)}
        if row["variant"] == "full" and row["repeat"] == 0 and row["permid"] in bad_first_full:
            lines += [json.dumps({"item": iid, "ok": True, "valid": False, "outcome": "invalid_format"})] * 2
            continue
        if mode == "broken":
            lines += [json.dumps({"item": iid, "ok": True, "valid": False, "outcome": "invalid_format"})] * 2
            continue
        sm = 1000 if (mode == "flat" and k % 5 == 0) else 100  # a few degenerate sums for the F4 sensitivity
        if repeat_full_sum is not None and row["variant"] == "full" and row["repeat"] > 0:
            sm = repeat_full_sum
        row_ = {"item": iid, "ok": True, "valid": True, "outcome": "valid", "answer": {"probabilities": p, "sum": sm}}
        if served:
            row_["served_model"] = served(k)
        lines.append(json.dumps(row_))
    (dirp / "answers.jsonl").write_text("\n".join(lines) + "\n")


def test_end_to_end(tmp_path):
    src = _csv(tmp_path / "s.csv.gz")
    b = tmp_path / "b"
    p = subprocess.run([sys.executable, str(ROOT / "scripts/safe_build.py"), "--safe", str(src), "--split", "test",
                        "--sample", "10", "--repeats", "1", "--repeat-firms", "5", "--out", str(b)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    design = json.loads((b / "design.json").read_text())
    key = json.loads((b / "key.json").read_text())
    fz = tmp_path / "frozen.json"  # the test split runs only under frozen settings (since 0.7)
    frozen = {"models": {n: {"config": {k: None for k in A.FROZEN_CONFIG} | {"api": "mock", "model": m, "max_attempts": 2,
                                                                         "parser": design["parser"]},
                             "model_digest": "sha256:mock", "K": 1}
                         for n, m in {"oracle": "oracle", "flat": "flat", "harmful": "harmful", "b2": "broken",
                                      "u": "flat", "s": "flat", "x": "flat", "y": "flat"}.items()}}
    frozen["build"] = {k: design.get(k) for k in A.FROZEN_BUILD}
    frozen["materials_sha256"] = {f: A.sha((ROOT / f).read_bytes()) for f in ("scripts/safe_analyze.py", "data/reference/safe_card_key.json",
                                                                         *RUN_MATERIALS)}
    frozen["test_build"] = {k: design[k] for k in ("items_sha256", "key_json_sha256", "source_sha256", "records", "items")}
    for m_ in frozen["models"].values():
        m_["server"] = A.server_identity({})
    frozen["status"] = "frozen (approved by the project lead on 2026-11-19)"
    fz.write_text(json.dumps(frozen))
    _run(tmp_path / "oracle", design, key, "oracle", drop=7)
    _run(tmp_path / "flat", design, key, "flat")
    rep = tmp_path / "r.json"
    _run(tmp_path / "harmful", design, key, "harmful")
    analyze = lambda *runs: subprocess.run([sys.executable, str(ROOT / "scripts/safe_analyze.py"), "--build", str(b), "--frozen", str(fz),  # noqa: E731
                                            *[x for n in runs for x in ("--run", f"{n}={tmp_path / n}")],
                                            "--safe", str(src), "--out", str(rep)], capture_output=True, text=True, timeout=600)
    p = analyze("oracle", "flat", "harmful")
    assert p.returncode == 0, p.stderr
    fz_ok = fz.read_text()
    bad = [("model_digest", "sha256:other", "weights digest"), (("config", "max_attempts"), 3, "settings differ"),
           (("config", "seed"), 1, "settings differ"), (("config", "effort"), "DROP", "lacks the settings"),
           ("config", "DROP", "has no config")]
    for path, val, msg in bad:  # the test split refuses a run off the frozen settings (code review rounds 7)
        f2 = json.loads(fz_ok)
        e = f2["models"]["oracle"]
        tgt, key_ = (e[path[0]], path[1]) if isinstance(path, tuple) else (e, path)
        if val == "DROP":
            del tgt[key_]
        else:
            tgt[key_] = val
        fz.write_text(json.dumps(f2))
        q = analyze("oracle")
        assert q.returncode != 0 and msg in q.stderr and "Traceback" not in q.stderr, q.stderr
    fz.write_text(fz_ok)
    for part, change, msg in (("build", {"elicitation": "counts_nozero"}, "not the frozen one"),
                              ("materials_sha256", {"scripts/safe_analyze.py": "0" * 64}, "materials changed"),
                              ("test_build", {"items_sha256": "0" * 64}, "not the frozen test set"),
                              ("test_build", {"records": 4}, "not the frozen test set")):
        f3 = json.loads(fz_ok)
        f3[part] = {**f3[part], **change}
        fz.write_text(json.dumps(f3))
        q = analyze("oracle")
        assert q.returncode != 0 and msg in q.stderr, q.stderr  # since 0.8: the frozen build and materials
    fz.write_text(fz_ok)
    rj0 = (tmp_path / "oracle/run.json").read_text()  # a session that ran with another runner module (code review 10)
    rj1 = json.loads(rj0)
    rj1["sessions"][0]["materials_sha256"]["scripts/llm_serving_check.py"] = "0" * 64
    (tmp_path / "oracle/run.json").write_text(json.dumps(rj1))
    q = analyze("oracle")
    assert q.returncode != 0 and "materials other than the frozen ones" in q.stderr
    rj1["sessions"][0].pop("materials_sha256")
    (tmp_path / "oracle/run.json").write_text(json.dumps(rj1))
    assert "recorded no materials" in analyze("oracle").stderr
    rj2 = json.loads(rj0)  # a partial record of the modules (code review 11, D1)
    rj2["sessions"][0]["materials_sha256"].pop("scripts/safe_prompt.py")
    (tmp_path / "oracle/run.json").write_text(json.dumps(rj2))
    assert "not the runner's modules" in analyze("oracle").stderr
    rj3 = json.loads(rj0)  # another SDK than the frozen serving stack (code review 11, D2)
    rj3["sessions"][0]["server"] = {"anthropic_sdk": "9.9.9"}
    (tmp_path / "oracle/run.json").write_text(json.dumps(rj3))
    assert "another serving stack" in analyze("oracle").stderr
    (tmp_path / "oracle/run.json").write_text(rj0)
    f5 = json.loads(fz_ok)
    f5["status"] = "candidate"
    fz.write_text(json.dumps(f5))
    assert "not approved" in analyze("oracle").stderr
    f4 = json.loads(fz_ok)
    f4.pop("test_build")
    fz.write_text(json.dumps(f4))
    q = analyze("oracle")
    assert q.returncode != 0 and "no test_build" in q.stderr
    fz.write_text(fz_ok)
    q = subprocess.run([sys.executable, str(ROOT / "scripts/safe_analyze.py"), "--build", str(b), "--frozen", str(fz), "--run",
                        f"oracle={tmp_path / 'oracle'}", "--safe", str(src), "--out", str(rep), "--level-check"],
                       capture_output=True, text=True, timeout=600)
    assert q.returncode != 0 and "development builds only" in q.stderr  # refused before any test data is read
    q = subprocess.run([sys.executable, str(ROOT / "scripts/safe_analyze.py"), "--build", str(b), "--frozen", str(fz), "--run",
                        f"oracle={tmp_path / 'oracle'}", "--safe", str(src), "--out", str(rep), "--diagnose"],
                       capture_output=True, text=True, timeout=600)
    assert q.returncode != 0 and "development builds only" in q.stderr
    p = analyze("oracle", "flat", "harmful")
    r = json.loads(rep.read_text())
    o, f, h = r["models"]["oracle"], r["models"]["flat"], r["models"]["harmful"]
    sup = r["primary_family"]["supports_claim"]
    assert o["primary"]["mean_d"] > 0.3 and sup["oracle"]
    assert f["primary"]["mean_d"] == 0 and not sup["flat"]
    assert h["primary"]["mean_d"] < 0 and not sup["harmful"]  # directional: a significant harm is no support
    assert r["primary_family"]["p_directional_H1"]["harmful"] == 1.0
    assert r["primary_family"]["holm_rejected"] == sup and "Holm over the three H1" in r["primary_family"]["procedure"]
    assert "graph_rejected" not in r["primary_family"] and set(r["primary_family"]["h3_exploratory"]["p_directional_shown_without_claim"]) == {"oracle", "flat", "harmful"}
    assert o["h3_full_vs_logit_segment"]["mean"] > 0 and o["primary"]["ci95"][0] > 0
    assert o["raw_sums"]["full"]["share_exactly_100"] == 1
    fs = f["primary"]["sensitivity"]["without_sums_outside_range"]
    assert f["raw_sums"]["full"]["max"] == 1000 and 0 < fs["share_of_answers_outside"] and fs["firms"] < 40
    assert o["technical_errors"]["segment"] > 0 and o["primary"]["firms"] == 40 - o["technical_errors"]["segment"]
    assert o["primary"]["sensitivity"]["errors_as_marginal"]["firms"] == 40
    sv = o["primary"]["sensitivity"]
    assert sv["errors_worst_symmetric"]["mean_d"] >= sv["errors_as_marginal"]["mean_d"]  # only segment answers fail here
    assert sv["errors_against_the_claim"]["mean_d"] == sv["errors_as_marginal"]["mean_d"]  # no failed full card
    assert {"floor_0.001", "floor_0.05"} <= set(sv)
    assert o["technical_error_flag"]["over_threshold"] and o["technical_error_flag"]["full_minus_segment"] < 0
    assert "secondary_gated_full_vs_shuffled" not in r and "test" not in o["full_minus_shuffled"]
    assert o["primary"]["test"] == "two-sided bootstrap-t" and o["primary"]["p_sign_flip"] < 0.01 and o["primary"]["p_t"] < 0.01
    assert o["technical_error_outcomes"] == {"invalid_format": o["technical_errors"]["segment"]}
    assert r["fits"]["failed"] == 0 and r["statistical"]["training_records"] == 240
    st = o["stability"]
    assert st["full"]["pairs"] == 5 and st["segment"]["pairs"] >= 4 and st["d"]["firms"] >= 4
    assert st["d"]["noise_variance_of_one_draw_of_d"] == 0  # the mock answers the same every time
    assert set(o["references"]["full"]) >= {"minus_marginal", "minus_rule", "minus_logit", "minus_logit_segment"}
    assert o["precision"]["mde_80_alpha_0.05_over_3"] > 0
    p = analyze("oracle", "flat")  # a family of two: the third model enters Holm with p = 1
    r = json.loads(rep.read_text())
    assert r["primary_family"]["models_missing_from_this_report"] == 1
    _run(tmp_path / "b2", design, key, "broken")
    p = analyze("b2")
    assert p.returncode == 0, p.stderr
    assert json.loads(rep.read_text())["models"]["b2"]["primary"]["firms"] == 0
    _run(tmp_path / "u", design, key, "flat", finished=False)
    assert "did not finish" in analyze("u").stderr
    _run(tmp_path / "s", design, key, "flat", skip=9)
    assert "neither valid nor" in analyze("s").stderr
    assert "same directory" in subprocess.run([sys.executable, str(ROOT / "scripts/safe_analyze.py"), "--build", str(b), "--frozen", str(fz), "--run",
                                               f"x={tmp_path / 'flat'}", "--run", f"y={tmp_path / 'flat'}", "--safe", str(src),
                                               "--out", str(rep)], capture_output=True, text=True).stderr
    assert "unique" in subprocess.run([sys.executable, str(ROOT / "scripts/safe_analyze.py"), "--build", str(b), "--frozen", str(fz), "--run",
                                       f"x={tmp_path / 'flat'}", "--run", f"x={tmp_path / 'flat'}", "--safe", str(src), "--out", str(rep)],
                                      capture_output=True, text=True).stderr
    other = tmp_path / "x"
    _run(other, {**design, "items_sha256": "0" * 64}, key, "flat")
    p = subprocess.run([sys.executable, str(ROOT / "scripts/safe_analyze.py"), "--build", str(b), "--frozen", str(fz), "--run", f"x={other}",
                        "--safe", str(src), "--out", str(rep)], capture_output=True, text=True)
    assert p.returncode != 0 and "another build" in p.stderr


def test_format_comparison_on_the_same_records(tmp_path):
    src = _csv(tmp_path / "s.csv.gz")
    run = lambda *x: subprocess.run([sys.executable, str(ROOT / "scripts/safe_build.py"), "--safe", str(src), *map(str, x)],  # noqa: E731
                                    capture_output=True, text=True)
    p = run("--split", "dev", "--sample", "10", "--repeats", "1", "--repeat-firms", "6", "--elicitation", "probabilities",
            "--repeat-variants", "full", "--out", tmp_path / "p1")
    assert p.returncode == 0, p.stderr
    p1 = json.loads((tmp_path / "p1/key.json").read_text())
    # the first pilot's key had no donor_wave: such a key needs the earlier build's seed (code review round 5, user review 3)
    (tmp_path / "old").mkdir()
    (tmp_path / "old/key.json").write_text(json.dumps({i: {k: v for k, v in r.items() if k != "donor_wave"} for i, r in p1.items()}))
    (tmp_path / "old/design.json").write_text((tmp_path / "p1/design.json").read_text())
    p = run("--split", "dev", "--only-repeat-firms-of", tmp_path / "old/key.json", "--seed", "7", "--out", tmp_path / "bad")
    assert p.returncode != 0 and "earlier build's --seed" in p.stderr
    for fmt in ("probabilities", "counts"):
        p = run("--split", "dev", "--only-repeat-firms-of", tmp_path / "old/key.json", "--elicitation", fmt, "--out", tmp_path / fmt)
        assert p.returncode == 0, p.stderr
    p = run("--split", "dev", "--only-repeat-firms-of", tmp_path / "p1/key.json", "--seed", "7", "--out", tmp_path / "counts7")
    assert p.returncode == 0, p.stderr  # with donor_wave the seed does not matter for the cards
    rep_firms = {(k["wave"], k["permid"]) for k in p1.values() if k["repeat"] > 0}
    donors = lambda key: {(k["wave"], k["permid"]): (k["donor_permid"], k["donor_wave"]) for k in key.values() if k["variant"] == "shuffled"}  # noqa: E731
    for b in ("probabilities", "counts", "counts7"):
        key = json.loads((tmp_path / b / "key.json").read_text())
        assert {(k["wave"], k["permid"]) for k in key.values()} == rep_firms
        assert donors(key) == {f: d for f, d in donors(p1).items() if f in rep_firms}  # the earlier donors, so the same cards
    for b in ("probabilities", "counts", "counts7"):
        _run(tmp_path / f"r_{b}", json.loads((tmp_path / b / "design.json").read_text()),
             json.loads((tmp_path / b / "key.json").read_text()), "oracle" if b != "probabilities" else "flat", model="m")
    rep = tmp_path / "fc.json"
    fc = lambda o, n: subprocess.run([sys.executable, str(ROOT / "scripts/safe_format_compare.py"),  # noqa: E731
                                      "--old-build", str(tmp_path / o), "--old-run", str(tmp_path / f"r_{o}"),
                                      "--new-build", str(tmp_path / n), "--new-run", str(tmp_path / f"r_{n}"), "--out", str(rep)],
                                     capture_output=True, text=True)
    p = fc("probabilities", "counts")
    assert p.returncode == 0, p.stderr
    r = json.loads(rep.read_text())
    assert r["cards_checked_identical"] == 3 * len(rep_firms)
    assert r["variants"]["full"]["band_log_score"]["counts_minus_probabilities"] > 0
    assert fc("probabilities", "counts7").returncode == 0
    # runs must be finished and complete, with the same settings (project lead's review, 18.11.2026)
    rj = tmp_path / "r_counts7/run.json"
    good_rj = rj.read_text()
    c = json.loads(good_rj)
    c["config"]["max_tokens"] = 99
    rj.write_text(json.dumps(c))
    assert "differ in their settings" in fc("probabilities", "counts7").stderr
    c = json.loads(good_rj)
    del c["sessions"][-1]["finished_at"]
    rj.write_text(json.dumps(c))
    assert "did not finish" in fc("probabilities", "counts7").stderr
    rj.write_text(good_rj)
    ans_f = tmp_path / "r_counts7/answers.jsonl"
    good_ans = ans_f.read_text()
    ans_f.write_text("\n".join(good_ans.splitlines()[:1]) + "\n")  # an interrupted run with one answer
    assert "neither valid nor" in fc("probabilities", "counts7").stderr
    ans_f.write_text(good_ans)
    # the same cards with another survey answer are refused (user review, 18.11.2026)
    k7 = json.loads((tmp_path / "counts7/key.json").read_text())
    f0 = next(iter(k7.values()))
    for r_ in k7.values():
        if (r_["wave"], r_["permid"]) == (f0["wave"], f0["permid"]):
            r_["answer"] = "1" if r_["answer"] != "1" else "2"
    (tmp_path / "counts7/key.json").write_text(json.dumps(k7))
    d7 = json.loads((tmp_path / "counts7/design.json").read_text())
    assert "key.json does not match" in fc("probabilities", "counts7").stderr
    d7["key_json_sha256"] = A.sha((tmp_path / "counts7/key.json").read_bytes())
    (tmp_path / "counts7/design.json").write_text(json.dumps(d7))
    p = fc("probabilities", "counts7")
    assert p.returncode != 0 and "different answers" in p.stderr
    p = fc("counts", "counts7")  # two counts runs are not a comparison of formats
    assert p.returncode != 0 and "not a probabilities build" in p.stderr
    # a changed card is refused
    items = [json.loads(x) for x in gzip.decompress((tmp_path / "counts/items.jsonl.gz").read_bytes()).decode().splitlines()]
    items[0]["messages"][1]["content"] += " "
    text = "".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items).encode()
    (tmp_path / "counts/items.jsonl.gz").write_bytes(gzip.compress(text))
    d = json.loads((tmp_path / "counts/design.json").read_text())
    d["items_sha256"] = A.sha(text)
    (tmp_path / "counts/design.json").write_text(json.dumps(d))
    c = json.loads((tmp_path / "r_counts/run.json").read_text())
    c["config"]["items_sha256"] = d["items_sha256"]
    (tmp_path / "r_counts/run.json").write_text(json.dumps(c))
    p = fc("probabilities", "counts")
    assert p.returncode != 0 and "cards differ" in p.stderr


def test_sample_by_size_and_arm_b_build(tmp_path):
    """safe_build 0.8: --sample-by-size (the third pilot's 30/30/30/11) names every class once; --elicitation
    counts_nozero builds arm B's prompt with the counts parser."""
    src = _csv(tmp_path / "s.csv.gz")
    run = lambda *x: subprocess.run([sys.executable, str(ROOT / "scripts/safe_build.py"), "--safe", str(src), "--split", "dev",  # noqa: E731
                                     *map(str, x)], capture_output=True, text=True)
    p = run("--sample-by-size", "1=5", "2=5", "3=5", "4=2", "--elicitation", "counts_nozero", "--repeats", "2",
            "--repeat-firms", "17", "--out", tmp_path / "b")
    assert p.returncode == 0, p.stderr
    d = json.loads((tmp_path / "b/design.json").read_text())
    assert d["records_by_size"] == {"1": 5, "2": 5, "3": 5, "4": 2} and d["parser"] == "safe_counts" and d["elicitation"] == "counts_nozero"
    import safe_prompt as P
    sysp = {json.loads(x)["messages"][0]["content"] for x in gzip.decompress((tmp_path / "b/items.jsonl.gz").read_bytes()).decode().splitlines()}
    assert sysp == {P.system_prompt(P.load_key(), "counts_nozero")}
    assert "exactly the size classes" in run("--sample-by-size", "1=5", "2=5", "3=5", "--out", tmp_path / "x").stderr
    assert "excludes --sample" in run("--sample", "5", "--sample-by-size", "1=5", "2=5", "3=5", "4=2", "--out", tmp_path / "y").stderr
    assert "records, 999 asked" in run("--sample-by-size", "1=5", "2=5", "3=5", "4=999", "--out", tmp_path / "z").stderr


def test_build_refuses_silent_shrinking(tmp_path):
    src = _csv(tmp_path / "s.csv.gz")
    run = lambda *x: subprocess.run([sys.executable, str(ROOT / "scripts/safe_build.py"), "--safe", str(src), *map(str, x)],  # noqa: E731
                                    capture_output=True, text=True)
    rows = list(csv.DictReader(gzip.decompress(src.read_bytes()).decode().splitlines()))
    ex = {str(i): {"permid": r["permid"]} for i, r in enumerate(rows) if r["d1_rec"] == "4"}
    (tmp_path / "ex.json").write_text(json.dumps(ex))
    p = run("--split", "dev", "--sample", "2", "--exclude-key", tmp_path / "ex.json", "--out", tmp_path / "b")
    assert p.returncode != 0 and "size class 4: 0 records" in p.stderr  # an emptied class stops the build (user review 4)
    p = run("--split", "dev", "--exclude-key", tmp_path / "ex.json", "--out", tmp_path / "b0")
    assert p.returncode != 0 and "size class 4: 0 records left" in p.stderr  # also without --sample (code review round 6)
    p = run("--split", "dev", "--sample", "10", "--repeats", "1", "--repeat-firms", "4", "--out", tmp_path / "p1")
    assert p.returncode == 0, p.stderr
    p = run("--split", "dev", "--only-repeat-firms-of", tmp_path / "p1/key.json", "--exclude-key", tmp_path / "p1/key.json",
            "--out", tmp_path / "b2")
    assert p.returncode != 0 and "both kept and excluded" in p.stderr


def test_average_draws_one_denominator(tmp_path):
    """--average-draws: an invalid first draw makes the firm-card an error everywhere, the sums rule sees every draw,
    and the report has no NaN (user review 2, code review round 5)."""
    src = _csv(tmp_path / "s.csv.gz")
    b = tmp_path / "b"
    p = subprocess.run([sys.executable, str(ROOT / "scripts/safe_build.py"), "--safe", str(src), "--split", "dev",
                        "--sample", "10", "--repeats", "1", "--repeat-firms", "40", "--out", str(b)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    design, key = json.loads((b / "design.json").read_text()), json.loads((b / "key.json").read_text())
    rep_ids = sorted({k["permid"] for k in key.values() if k["repeat"] > 0})
    _run(tmp_path / "m", design, key, "oracle", bad_first_full=set(rep_ids[:5]), repeat_full_sum=1000)
    out = {}
    for flag in ([], ["--average-draws"]):
        rep = tmp_path / f"r{len(flag)}.json"
        p = subprocess.run([sys.executable, str(ROOT / "scripts/safe_analyze.py"), "--build", str(b), "--run", f"m={tmp_path / 'm'}",
                            "--safe", str(src), "--out", str(rep), *flag], capture_output=True, text=True, timeout=600)
        assert p.returncode == 0, p.stderr
        out[len(flag)] = json.loads(rep.read_text())["models"]["m"]
    first, avg = out[0], out[1]
    assert first["technical_errors"]["full"] == avg["technical_errors"]["full"] == 5
    assert avg["primary"]["firms"] == first["primary"]["firms"]  # an invalid first draw is an error in both modes
    assert first["raw_sums"]["full"]["share_outside_50_200"] == 0  # the first draws all sum to 100
    assert avg["raw_sums"]["full"]["share_outside_50_200"] > 0.2  # every repeat of the full card sums to 1000
    assert avg["primary"]["sensitivity"]["without_sums_outside_range"]["firms"] == first["primary"]["firms"] - (len(rep_ids) - 5)
    assert all(v is not None for v in avg["ten_logloss"].values()) and all(v is not None for v in avg["rps"].values())
    assert avg["stability"]["k_rule"]["firms"] == len(rep_ids) - 5
    assert A.boot_t_ci(np.ones(10) * 0.1) is None  # below MIN_FIRMS: no interval


def test_k_rule_fallback_below_two_firms(tmp_path):
    """One firm with valid repeats of both cards: K = 3, not no recommendation (user review, 18.11.2026)."""
    src = _csv(tmp_path / "s.csv.gz")
    b = tmp_path / "b"
    p = subprocess.run([sys.executable, str(ROOT / "scripts/safe_build.py"), "--safe", str(src), "--split", "dev",
                        "--sample", "10", "--repeats", "1", "--repeat-firms", "1", "--out", str(b)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    _run(tmp_path / "m", json.loads((b / "design.json").read_text()), json.loads((b / "key.json").read_text()), "flat")
    rep = tmp_path / "r.json"
    p = subprocess.run([sys.executable, str(ROOT / "scripts/safe_analyze.py"), "--build", str(b), "--run", f"m={tmp_path / 'm'}",
                        "--safe", str(src), "--out", str(rep)], capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stderr
    k = json.loads(rep.read_text())["models"]["m"]["stability"]["k_rule"]
    assert k["recommended_K"] == 3 and k["firms"] <= 1


def test_frozen_settings_k_and_coverage(tmp_path):
    """Since 0.7 (project lead's review, 18.11.2026): one served model per run; a fixed K needs K draws of both cards
    for every firm; coverage, the estimand and the error diagnostics are reported; a duplicated draw is refused."""
    src = _csv(tmp_path / "s.csv.gz")
    build = lambda out, *x: subprocess.run([sys.executable, str(ROOT / "scripts/safe_build.py"), "--safe", str(src), "--split", "dev",  # noqa: E731
                                            "--sample", "10", "--out", str(out), *map(str, x)], capture_output=True, text=True)
    assert "named twice" in build(tmp_path / "bd", "--repeats", "1", "--repeat-firms", "5", "--repeat-variants", "full", "full").stderr
    b = tmp_path / "b"
    p = build(b, "--repeats", "1", "--repeat-firms", "40")  # every firm has two draws of both cards
    assert p.returncode == 0, p.stderr
    design, key = json.loads((b / "design.json").read_text()), json.loads((b / "key.json").read_text())
    an = lambda *x: subprocess.run([sys.executable, str(ROOT / "scripts/safe_analyze.py"), "--build", str(b),  # noqa: E731
                                    "--safe", str(src), "--out", str(tmp_path / "r.json"), *map(str, x)],
                                   capture_output=True, text=True, timeout=600)
    _run(tmp_path / "mix", design, key, "oracle", served=lambda k: "m-a" if k % 2 else "m-b")
    (tmp_path / "mix/run.json").write_text((tmp_path / "mix/run.json").read_text().replace('"api": "mock"', '"api": "anthropic"'))
    p = an("--run", f"m={tmp_path / 'mix'}")
    assert p.returncode != 0 and "more than one model" in p.stderr
    _run(tmp_path / "one", design, key, "oracle", served=lambda k: "m-a", bad_first_full=set())
    p = an("--run", f"m={tmp_path / 'one'}", "--k", "m=2")
    assert p.returncode == 0, p.stderr
    m = json.loads((tmp_path / "r.json").read_text())["models"]["m"]
    assert m["draws"] == "fixed K = 2" and m["served_model"] == "m-a"
    assert m["coverage"]["share"] == 1.0 and "complete cases" in m["primary"]["estimand"]
    assert m["references"]["full"]["minus_marginal_ci95"] is not None
    pr = m["precision"]
    assert pr["variance_between_classes"] >= 0 and pr["sd_d_total_at_test_sizes"] ** 2 >= pr["variance_within_classes"] - 1e-3
    assert an("--run", f"m={tmp_path / 'one'}", "--k", "m=3").returncode != 0  # the build has two draws
    p = an("--run", f"m={tmp_path / 'one'}", "--level-check", "--level-reps", "20", "--level-inner", "200")
    assert p.returncode == 0, p.stderr
    lc = json.loads((tmp_path / "r.json").read_text())["level_check"]
    assert lc["number_of_bounds"] == 12 and {c["check"].split("/")[1] for c in lc["checks"]} == {"H1", "H3"}
    assert not lc["ready_for_freezing"] and lc["status"] == "diagnostic"  # one model, 20 replications: diagnostic only
    assert any("not taken from a first report" in x for x in lc["not_ready_because"])
    # the two-step procedure (project lead, 18.11.2026): K from the first report, never re-chosen
    an2 = lambda out, *x: subprocess.run([sys.executable, str(ROOT / "scripts/safe_analyze.py"), "--build", str(b), "--safe", str(src),  # noqa: E731
                                          "--run", f"m={tmp_path / 'one'}", "--out", str(out), *map(str, x)],
                                         capture_output=True, text=True, timeout=600)
    first = tmp_path / "first.json"
    assert an2(first).returncode == 0
    kr = json.loads(first.read_text())["models"]["m"]["stability"]["k_rule"]["recommended_K"]
    p = an2(tmp_path / "second.json", "--k-from", first, "--level-check", "--level-reps", "20", "--level-inner", "200")
    assert p.returncode == 0, p.stderr
    sec = json.loads((tmp_path / "second.json").read_text())
    assert sec["draws_by_model"] == {"m": kr} and sec["k_source"]["first_report_sha256"] == A.sha(first.read_bytes())
    assert not any("not taken from a first report" in x for x in sec["level_check"]["not_ready_because"])
    assert "--level-check and --k" in an2(tmp_path / "x.json", "--k-from", tmp_path / "second.json").stderr  # not a first report
    assert "excludes" in an2(tmp_path / "x.json", "--k-from", first, "--k", "m=1").stderr
    # --k-from binds the first report to exactly the same run (project lead's review, 18.11.2026)
    import shutil
    for what in ("answers", "max_tokens", "digest"):
        alt = tmp_path / f"alt_{what}"
        shutil.copytree(tmp_path / "one", alt)
        rj_ = json.loads((alt / "run.json").read_text())
        if what == "answers":
            (alt / "answers.jsonl").write_text((alt / "answers.jsonl").read_text() + "\n")
        elif what == "max_tokens":
            rj_["config"]["max_tokens"] = 123
        else:
            rj_["config"]["model_digest"] = "sha256:other"
        (alt / "run.json").write_text(json.dumps(rj_))
        q = subprocess.run([sys.executable, str(ROOT / "scripts/safe_analyze.py"), "--build", str(b), "--safe", str(src), "--run",
                            f"m={alt}", "--out", str(tmp_path / "x.json"), "--k-from", str(first)], capture_output=True, text=True, timeout=600)
        assert q.returncode != 0 and "not the run of the first report" in q.stderr, (what, q.stderr)
    # --settings-like (the third pilot): the same settings as an earlier report, the runner version aside
    assert "same build" in an2(tmp_path / "s1.json", "--settings-like", first).stderr  # code review 9, D1
    other = json.loads(first.read_text())
    other["build"]["items_sha256"] = "0" * 64  # stands for the second pilot's report (another build)
    first_other = tmp_path / "first_other.json"
    first_other.write_text(json.dumps(other))
    assert an2(tmp_path / "s1.json", "--settings-like", first_other).returncode == 0
    rj_ = json.loads((tmp_path / "one/run.json").read_text())
    shutil.copytree(tmp_path / "one", tmp_path / "newer")
    (tmp_path / "newer/run.json").write_text(json.dumps({**rj_, "config": {**rj_["config"], "script": "9.9.9"}}))
    q = subprocess.run([sys.executable, str(ROOT / "scripts/safe_analyze.py"), "--build", str(b), "--safe", str(src), "--run",
                        f"m={tmp_path / 'newer'}", "--out", str(tmp_path / "s2.json"), "--settings-like", str(first_other)],
                       capture_output=True, text=True, timeout=600)
    assert q.returncode == 0, q.stderr
    assert json.loads((tmp_path / "s2.json").read_text())["settings_source"]["report_sha256"] == A.sha(first_other.read_bytes())
    for what in ("max_tokens", "digest"):
        q = subprocess.run([sys.executable, str(ROOT / "scripts/safe_analyze.py"), "--build", str(b), "--safe", str(src), "--run",
                            f"m={tmp_path / f'alt_{what}'}", "--out", str(tmp_path / "s3.json"), "--settings-like", str(first_other)],
                           capture_output=True, text=True, timeout=600)
        assert q.returncode != 0 and "--settings-like" in q.stderr, (what, q.stderr)
    assert "exclude each other" in an("--run", f"m={tmp_path / 'one'}", "--k", "m=2", "--average-draws").stderr
    assert "without a run" in an("--run", f"m={tmp_path / 'one'}", "--k", "z=1").stderr
    # a run of only errors keeps the error diagnostics
    _run(tmp_path / "br", design, key, "broken")
    assert an("--run", f"m={tmp_path / 'br'}").returncode == 0
    m = json.loads((tmp_path / "r.json").read_text())["models"]["m"]
    assert m["primary"]["firms"] == 0 and m["technical_error_flag"]["over_threshold"] and m["coverage"]["share"] == 0
    assert m["primary"]["sensitivity"]["errors_against_the_claim"]["firms"] == m["coverage"]["eligible_firms"]
    # coverage by size class carried to the test sizes (project lead's review): n = sum_k 692 share_k cov_k
    _run(tmp_path / "part", design, key, "oracle", drop=3)
    assert an("--run", f"m={tmp_path / 'part'}").returncode == 0
    pr = json.loads((tmp_path / "r.json").read_text())["models"]["m"]["precision"]
    sh = json.loads((ROOT / "data/processed/safe_validation_audit_v0.1.json").read_text())["waves"]["2026q3"]["size"]
    code = {"1-9": "1", "10-49": "2", "50-249": "3", "250+": "4"}
    tot = sum(sh.values())
    exp = sum(692 * v / tot * pr["coverage_by_size"].get(code[k], pr["coverage_of_this_run"]) for k, v in sh.items())
    assert abs(pr["test_firms_expected_in_the_comparison"] - exp) <= 1 and len(set(pr["coverage_by_size"].values())) > 1
    # --diagnose: a fixed descriptive scope; the other numbers stay as they are (project lead, 18.11.2026)
    base = json.loads((tmp_path / "r.json").read_text())
    p = an("--run", f"m={tmp_path / 'part'}", "--diagnose")
    assert p.returncode == 0, p.stderr
    dgr = json.loads((tmp_path / "r.json").read_text())
    dg = dgr["models"]["m"].pop("diagnosis")
    assert dgr["models"] == base["models"]
    assert dg["d_H1"]["n"] == base["models"]["m"]["primary"]["firms"] and "quantiles" in dg["d3_H3"]
    e = dg["technical_errors"]["segment"]
    assert e["firm_cards_failed_on_the_first_draw"] == base["models"]["m"]["technical_errors"]["segment"]
    assert e["classes"] == {"other": e["failed_model_answers"]} and all(isinstance(v, int) for v in e["text_features"].values())
    assert set(dg["concentration_first_draw"]) == {"full", "segment", "shuffled"}
    # a duplicated draw in key.json is refused
    k2 = dict(key)
    first = next(iter(k2.values()))
    k2["dup000000000"] = dict(first)
    (b / "key.json").write_text(json.dumps(k2))
    d2 = {**design, "key_json_sha256": A.sha((b / "key.json").read_bytes())}
    (b / "design.json").write_text(json.dumps(d2))
    assert "more than once" in an("--run", f"m={tmp_path / 'one'}").stderr


def test_interval_matches_test():
    """The symmetric bootstrap-t interval excludes 0 exactly when the two-sided test rejects at 0.05 (user review 1)."""
    rng = np.random.default_rng(11)
    for _ in range(40):
        d = rng.lognormal(0, 1.2, 50) - np.exp(0.72) + rng.normal(0.25, 0.3)
        p, ci = A.boot_t_test(d, 4000), A.boot_t_ci(d, 4000)
        assert (ci[0] > 0 or ci[1] < 0) == (p < 0.05) or abs(p - 0.05) < 0.002
