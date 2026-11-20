"""SAFE validation: the card (variants, labels, omitted lines, no source or identifiers), the parser of explicit
probabilities (strict JSON, keys, ranges, sums, duplicates), the build (split, analysis population, donors, repeats,
checksums) and a run of run_b_k2.py 0.9 with the safe_probabilities parser against a local mock server."""
import csv
import gzip
import io
import hashlib
import json
import re
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import safe_prompt as P  # noqa: E402

KEY = P.load_key()
REC = {"permid": "7", "wave": "36", "d1_rec": "2", "d3_rec": "4", "d4": "6.0", "d5_rec": "1", "d6": "5.0", "d7_rec": "1.0",
       "q4_a_rec": "1.0", "q4_c_rec": "2.0", "q4_d_rec": "7.0", "q4_r_rec": "", "q7a_a_rec": "", "q7a_d_rec": "1.0",
       "q7b_d_rec": "4.0", "q0b_3": "8.0"}
DONOR = {**REC, "permid": "8", "d4": "2", "q4_c_rec": "1", "q7a_d_rec": "3", "q7b_d_rec": ""}


def good(**over):
    p = {str(i): 0.1 for i in range(1, 11)}
    p.update(over)
    return json.dumps({"probabilities": p, "reason": "krótko"})


def test_card_variants_and_labels():
    full = P.card(REC, "full", KEY)
    seg = P.card(REC, "segment", KEY)
    shuf = P.card(REC, "shuffled", KEY, DONOR)
    assert "od 10 do 49" in full and "usługi" in full and "ponad 500 tys. do 1 mln euro" in full
    assert "linia kredytowa, kredyt w rachunku albo zadłużenie na karcie kredytowej: wchodzi w grę; w poprzednim i bieżącym kwartale: nie" in full
    assert "kredyt bankowy (bez dotowanych, bez kredytu w rachunku i linii kredytowych): nie wchodzi w grę" in full
    assert "faktoring" not in full  # not asked: the line is omitted
    assert "firma złożyła wniosek; wynik: wniosek odrzucono" in full
    assert "o żaden wniosek nie pytano" in P.card({**REC, "q7a_d_rec": "", "q7b_d_rec": ""}, "full", KEY)
    assert seg.count("\n- ") == 3 and "nie podano na tej karcie" in seg and "Roczny obrót" not in seg
    assert seg.splitlines()[:3] == shuf.splitlines()[:3]  # own size and sector
    assert "ponad 2 mln do 10 mln euro" in shuf and "nie złożyła wniosku, bo miała wystarczające środki własne" in shuf
    for t in (full, seg, shuf, P.system_prompt(KEY)):
        assert not re.search(r"safe|ecb|ebc|permid|\b20\d\d\b", t, re.I)
    with pytest.raises(ValueError):
        P.card({**REC, "d4": "8"}, "full", KEY)
    with pytest.raises(ValueError):
        P.card(REC, "shuffled", KEY)


def test_parser():
    a = P.parse(good())
    assert a["probabilities"]["1"] == pytest.approx(0.1) and a["sum"] == pytest.approx(1.0)
    b = P.parse(good(**{"1": 0.11}))  # sum 1.01: normalised
    assert sum(b["probabilities"].values()) == pytest.approx(1.0)
    assert P.parse(good(**{"1": 0.0, "2": 0.2}))["probabilities"]["1"] == 0.0  # zero allowed; scoring floors it
    assert P.parse(json.dumps({"probabilities": {str(i): 0.1 for i in range(1, 11)}, "reason": ""}))["reason"] == ""
    bad = [good(**{"1": 0.2}),  # sum 1.1
           good(**{"1": -0.1}), good(**{"1": True}), good(**{"1": "0.1"}),
           "```json\n" + good() + "\n```", "x" + good(),
           json.dumps({"probabilities": {str(i): 0.1 for i in range(1, 10)}, "reason": "r"}),
           json.dumps({"probabilities": {str(i): 0.1 for i in range(1, 11)}, "reason": 3}), "[" * 20000,
           json.dumps({"probabilities": {str(i): 0.1 for i in range(1, 11)}, "reason": "r", "extra": 1}),
           good().replace('"1": 0.1', '"1": 0.1, "1": 0.1'), "[]", ""]
    for t in bad:
        with pytest.raises(ValueError):
            P.parse(t)


def _safe_csv(path, drop=()):
    import safe_signal_check as SC
    cols = sorted((set(REC) | {"q0b_3"} | set(SC.CARD)) - set(drop))
    buf = io.StringIO()
    w = csv.DictWriter(buf, cols)
    w.writeheader()
    n = 0
    for wave in ("32", "36", "40"):
        for size in "1234":
            for sector in ("1", "4"):
                for k in range(3):
                    n += 1
                    w.writerow({**{c: "" for c in cols}, **{k: v for k, v in REC.items() if k in cols}, "permid": str(n), "wave": wave, "d1_rec": size, "d3_rec": sector,
                                "q0b_3": "99" if k == 2 and wave == "40" else str(1 + n % 10)})
    w.writerow({**{c: "" for c in cols}, **{k: v for k, v in REC.items() if k in cols}, "permid": "2", "wave": "40",
                "d1_rec": "1", "d3_rec": "1"})  # firm 2 is also in the test wave
    path.write_bytes(gzip.compress(buf.getvalue().encode()))
    return path


def test_build_dev_and_test(tmp_path):
    src = _safe_csv(tmp_path / "s.csv.gz")
    run = lambda *a: subprocess.run([sys.executable, str(ROOT / "scripts/safe_build.py"), "--safe", str(src), *a],  # noqa: E731
                                    capture_output=True, text=True)
    p = run("--split", "dev", "--sample", "2", "--repeats", "2", "--repeat-firms", "3", "--out", str(tmp_path / "d"))
    assert p.returncode == 0, p.stderr
    d = json.loads((tmp_path / "d/design.json").read_text())
    key = json.loads((tmp_path / "d/key.json").read_text())
    items = [json.loads(x) for x in gzip.open(tmp_path / "d/items.jsonl.gz", "rt")]
    assert d["parser"] == "safe_counts" and d["records"] == 8 and d["items"] == 8 * 3 + 3 * 2 * 2  # repeats of full and segment
    assert {k["wave"] for k in key.values()} <= {"2024q3", "2025q3"} and "2" not in {k["permid"] for k in key.values()}
    raw = gzip.open(tmp_path / "d/items.jsonl.gz").read()
    import hashlib
    assert hashlib.sha256(raw).hexdigest() == d["items_sha256"]
    for k in key.values():
        if k["variant"] == "shuffled":
            assert k["donor_permid"] != k["permid"]
    assert all(set(it) == {"item", "kind", "messages"} for it in items)
    p = run("--split", "test", "--out", str(tmp_path / "t"))
    assert p.returncode == 0, p.stderr
    t = json.loads((tmp_path / "t/design.json").read_text())
    assert t["target_answers_in_split"] == {"1-10": 17, "99": 8} and t["records"] == 17  # no distribution of the test wave
    p = run("--split", "test", "--out", str(tmp_path / "t"))
    assert p.returncode != 0 and "not empty" in p.stderr
    src2 = _safe_csv(tmp_path / "s2.csv.gz", drop=("q7a_c_rec",))
    p = subprocess.run([sys.executable, str(ROOT / "scripts/safe_build.py"), "--safe", str(src2), "--split", "dev",
                        "--out", str(tmp_path / "d2")], capture_output=True, text=True)
    assert p.returncode != 0 and "lacks card columns" in p.stderr and "q7a_c_rec" in p.stderr
    p = run("--split", "test", "--out", str(ROOT / "data/processed/safe_should_not_exist"))
    assert p.returncode != 0 and "data/local" in p.stderr and not (ROOT / "data/processed/safe_should_not_exist").exists()


class Mock(BaseHTTPRequestHandler):
    calls = 0

    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        out = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_GET(self):
        if self.path == "/api/version":
            return self._send({"version": "mock"})
        if self.path == "/api/tags":
            return self._send({"models": [{"name": "mock:latest", "digest": "sha256:" + "0" * 64}]})
        self._send({}, 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/api/show":
            return self._send({"modelfile": "", "details": {}})
        Mock.calls += 1
        sysmsg = body["messages"][0]["content"] if "messages" in body else ""
        ok = (json.dumps({"firms_per_100": {str(i): 10 for i in range(1, 11)}, "reason": "r"}) if "firms_per_100" in sysmsg
              else good())
        text = "nie JSON" if Mock.calls % 5 == 0 else ok
        self._send({"message": {"content": text}, "done_reason": "stop", "prompt_eval_count": 1, "eval_count": 1,
                    "load_duration": 1, "prompt_eval_duration": 1, "eval_duration": 1, "total_duration": 1})


def test_runner_with_the_safe_parser(tmp_path):
    src = _safe_csv(tmp_path / "s.csv.gz")
    out = tmp_path / "b"
    p = subprocess.run([sys.executable, str(ROOT / "scripts/safe_build.py"), "--safe", str(src), "--split", "dev",
                        "--sample", "1", "--out", str(out)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    srv = HTTPServer(("127.0.0.1", 0), Mock)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"
    try:
        p = subprocess.run([sys.executable, str(ROOT / "scripts/run_b_k2.py"), "run", "--out", str(out), "--model", "mock",
                            "--base-url", base, "--max-attempts", "2"], capture_output=True, text=True, timeout=300)
        d = json.loads((out / "design.json").read_text())
        d.pop("parser")  # a resume must not fall back to the B-K2 parser silently
        (out / "design.json").write_text(json.dumps(d))
        p2 = subprocess.run([sys.executable, str(ROOT / "scripts/run_b_k2.py"), "run", "--out", str(out), "--model", "mock",
                             "--base-url", base, "--max-attempts", "2"], capture_output=True, text=True, timeout=300)
        d["parser"] = "safe_counts"
        (out / "design.json").write_text(json.dumps(d))
    finally:
        srv.shutdown()
    assert p2.returncode != 0 and "parser" in p2.stderr
    rr = subprocess.run([sys.executable, str(ROOT / "scripts/b_k2_run_report.py"), "--run", str(out), "--out", str(tmp_path / "rr.json")],
                        capture_output=True, text=True, timeout=120)
    assert rr.returncode == 0, rr.stderr  # the run report reads a SAFE build
    assert p.returncode == 0, p.stderr
    rows = [json.loads(x) for x in (out / "answers.jsonl").read_text().splitlines()]
    valid = [r for r in rows if r.get("valid")]
    items = {json.loads(x)["item"] for x in gzip.open(out / "items.jsonl.gz", "rt")}
    assert {r["item"] for r in valid} == items  # every item valid, an invalid first attempt asked again
    assert any(r.get("outcome") == "invalid_format" for r in rows)
    assert all(abs(sum(r["answer"]["probabilities"].values()) - 1) < 1e-9 for r in valid)
    cfg = json.loads((out / "run.json").read_text())["config"]
    assert cfg["parser"] == "safe_counts" and cfg["script"] == "0.9.5"
    sess = json.loads((out / "run.json").read_text())["sessions"]
    assert all(set(s_["materials_sha256"]) == {"scripts/run_b_k2.py", "scripts/llm_serving_check.py", "scripts/ollama_provenance.py", "scripts/safe_prompt.py"} for s_ in sess)
    p = subprocess.run([sys.executable, str(ROOT / "scripts/run_b_k2.py"), "analyze", "--out", str(out)],
                       capture_output=True, text=True)
    assert p.returncode != 0 and "not a B-K2 build" in p.stderr


def test_runner_refuses_items_changed_after_the_build(tmp_path):
    """A card edited after the build (design.json not updated) is refused before the first call and on a resume
    (user review, 18.11.2026): nothing is sent, nothing is written."""
    src = _safe_csv(tmp_path / "s.csv.gz")
    out = tmp_path / "b"
    p = subprocess.run([sys.executable, str(ROOT / "scripts/safe_build.py"), "--safe", str(src), "--split", "dev",
                        "--sample", "1", "--out", str(out)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    good_bytes = (out / "items.jsonl.gz").read_bytes()
    rows = [json.loads(x) for x in gzip.decompress(good_bytes).decode().splitlines()]
    rows[0]["messages"][1]["content"] += "\n- Dopisana cecha: tak"
    with open(out / "items.jsonl.gz", "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as fh:
        fh.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode())
    srv = HTTPServer(("127.0.0.1", 0), Mock)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"
    cmd = [sys.executable, str(ROOT / "scripts/run_b_k2.py"), "run", "--out", str(out), "--model", "mock", "--base-url", base,
           "--max-attempts", "2"]
    try:
        before = Mock.calls
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        assert p.returncode != 0 and "items_sha256" in p.stderr and Mock.calls == before
        assert not (out / "answers.jsonl").exists() and not (out / "run.json").exists()
        (out / "items.jsonl.gz").write_bytes(good_bytes)  # the original items: the run goes through
        p = subprocess.run(cmd + ["--limit", "1"], capture_output=True, text=True, timeout=300)
        assert p.returncode == 0, p.stderr
        (out / "items.jsonl.gz").write_bytes(gzip.compress("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()))
        n = len((out / "answers.jsonl").read_text().splitlines())
        before = Mock.calls
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # a resume with edited items
        assert p.returncode != 0 and "items_sha256" in p.stderr and Mock.calls == before
        assert len((out / "answers.jsonl").read_text().splitlines()) == n
    finally:
        srv.shutdown()


def test_runner_reads_the_parser_per_item(tmp_path):
    """safe_by_item (run_b_k2 0.9.4, the elicitation comparison): arms A and B are parsed as counts, C as probabilities;
    a build whose items lack a known parser is refused before any call."""
    src = _safe_csv(tmp_path / "s.csv.gz")
    sb = lambda *x: subprocess.run([sys.executable, str(ROOT / "scripts/safe_build.py"), "--safe", str(src), "--split", "dev",  # noqa: E731
                                    *map(str, x)], capture_output=True, text=True)
    p = sb("--sample", "1", "--repeats", "1", "--repeat-firms", "10", "--out", tmp_path / "ref")
    assert p.returncode == 0, p.stderr
    out = tmp_path / "abc"
    p = sb("--only-repeat-firms-of", tmp_path / "ref/key.json", "--seed", "20261118", "--arms", "A", "B", "C",
           "--variants", "full", "segment", "--out", out)
    assert p.returncode == 0, p.stderr
    srv = HTTPServer(("127.0.0.1", 0), Mock)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    cmd = [sys.executable, str(ROOT / "scripts/run_b_k2.py"), "run", "--out", str(out), "--model", "mock", "--base-url",
           f"http://127.0.0.1:{srv.server_port}", "--max-attempts", "2"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    finally:
        srv.shutdown()
    assert "Traceback" not in p.stderr, p.stderr  # the mock answers every fifth call with no JSON: an item may fail twice
    key = json.loads((out / "key.json").read_text())
    rows = [json.loads(x) for x in (out / "answers.jsonl").read_text().splitlines()]
    valid = {r["item"]: r for r in rows if r.get("valid")}
    assert set(valid) <= set(key) and len(valid) >= len(key) - 2 and {key[i]["arm"] for i in valid} == {"A", "B", "C"}
    assert all((valid[i]["answer"]["sum"] > 2) == (key[i]["arm"] in ("A", "B")) for i in valid)  # counts: sum 100, probabilities 1
    assert json.loads((out / "run.json").read_text())["config"]["parser"] == "safe_by_item"
    # an item without a parser: refused before any call
    out2 = tmp_path / "abc2"
    p = sb("--only-repeat-firms-of", tmp_path / "ref/key.json", "--seed", "20261118", "--arms", "A", "C",
           "--variants", "full", "--out", out2)
    assert p.returncode == 0, p.stderr
    rows2 = [json.loads(x) for x in gzip.decompress((out2 / "items.jsonl.gz").read_bytes()).decode().splitlines()]
    rows2[0].pop("parser")
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows2).encode()
    (out2 / "items.jsonl.gz").write_bytes(gzip.compress(text))
    d = json.loads((out2 / "design.json").read_text())
    d["items_sha256"] = hashlib.sha256(text).hexdigest()
    (out2 / "design.json").write_text(json.dumps(d))
    before = Mock.calls
    p = subprocess.run(cmd[:4] + [str(out2)] + cmd[5:], capture_output=True, text=True, timeout=300)
    assert p.returncode != 0 and "without a known parser" in p.stderr and Mock.calls == before


def test_safe_scripts_do_not_touch_record_level_registers():
    """SAFE terms: no record-level linkage of SAFE with other firm data; the safe_* scripts read no data/raw/ file and no
    population or register file of the project."""
    for p in (ROOT / "scripts").glob("safe_*.py"):
        t = p.read_text(encoding="utf-8").lower()
        if p.name == "safe_assumption_check.py":  # compares aggregates of the synthetic population with SAFE aggregates
            assert "data/local/safe" not in t and "data/raw" not in t  # it reads SAFE only through the audit's aggregates
            continue
        for bad in ("data/raw", "data/population", "krs", "ceidg", "bir_", "wl_"):
            assert bad not in t, (p.name, bad)


def test_counts_parser_and_prompt():
    c = lambda **o: json.dumps({"firms_per_100": {str(i): o.get(str(i), 10) for i in range(1, 11)}, "reason": "r"})  # noqa: E731
    a = P.parse_counts(c(**{"1": 15}))  # sum 105: turned into shares
    assert a["sum"] == 105 and sum(a["probabilities"].values()) == pytest.approx(1) and a["probabilities"]["1"] == pytest.approx(15 / 105)
    assert P.parse_counts(c(**{"2": 0}))["probabilities"]["2"] == 0
    for bad in (c(**{"1": 10.5}), c(**{"1": -1}), c(**{"1": 101}), c(**{"1": True}),
                json.dumps({"firms_per_100": {str(i): 0 for i in range(1, 11)}, "reason": "r"}),
                json.dumps({"probabilities": {str(i): 0.1 for i in range(1, 11)}, "reason": "r"}), "[" * 20000):
        with pytest.raises(ValueError):
            P.parse_counts(bad)
    sp = P.system_prompt(KEY)
    assert "100 firm wybranych losowo" in sp and "mniej niż 1 na 100" in sp and "firms_per_100" in sp and "probabilities" not in sp and "nie dawaj zera" not in sp
    assert "probabilities" in P.system_prompt(KEY, "probabilities")


def test_a_record_without_any_donor_stops_the_build(tmp_path):
    import safe_signal_check as SC
    cols = sorted(set(REC) | {"q0b_3"} | set(SC.CARD))
    buf = io.StringIO()
    w = csv.DictWriter(buf, cols)
    w.writeheader()
    for n, size in enumerate(("1", "2"), 1):  # one firm per size class: no donor anywhere in its class
        w.writerow({**{c: "" for c in cols}, **{k: v for k, v in REC.items() if k in cols}, "permid": str(n), "wave": "32",
                    "d1_rec": size, "q0b_3": "3"})
    src = tmp_path / "s.csv.gz"
    src.write_bytes(gzip.compress(buf.getvalue().encode()))
    p = subprocess.run([sys.executable, str(ROOT / "scripts/safe_build.py"), "--safe", str(src), "--split", "dev",
                        "--out", str(tmp_path / "b")], capture_output=True, text=True)
    assert p.returncode != 0 and "no donor" in p.stderr and "Traceback" not in p.stderr
