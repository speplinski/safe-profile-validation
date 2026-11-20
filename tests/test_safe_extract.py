"""safe_extract: the Polish extract from a synthetic ECB-like archive (no SAFE microdata)."""
import gzip
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "safe_extract.py"

CSV = ("permid,wave,d0,wgtcommon,q0b_3,q4_a_rec\n"
       "1,40,PL,0.5,3,\n"
       "2,40,DE,1.2,5,1\n"
       "3,36,PL,0.7,,2\n"
       "4,36,FR,0.9,8,7\n")


def make_zip(path: Path, member: str = "safepanel_allrounds.csv") -> Path:
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(member, CSV)
    return path


def run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True)


def test_extract_polish_records_deterministic(tmp_path):
    z = make_zip(tmp_path / "a.zip")
    r1 = run("--zip", z, "--out", tmp_path / "o1" / "pl.csv.gz")
    r2 = run("--zip", z, "--out", tmp_path / "o2" / "pl.csv.gz")
    assert r1.returncode == 0, r1.stderr
    a, b = json.loads(r1.stdout), json.loads(r2.stdout)
    assert a["records"] == 2 and a["waves"] == {"36": 1, "40": 1}
    assert a["extract_sha256"] == b["extract_sha256"]  # same archive, same bytes
    rows = gzip.decompress((tmp_path / "o1" / "pl.csv.gz").read_bytes()).decode().splitlines()
    assert rows[0] == "permid,wave,d0,wgtcommon,q0b_3,q4_a_rec" and len(rows) == 3
    assert all(",PL," in x for x in rows[1:])
    assert a["csv_sha256"] == hashlib.sha256(("\n".join(rows) + "\n").encode()).hexdigest()
    assert a["same_as_the_validation"]["extract_sha256"] is False  # a synthetic archive is not the ECB's


def test_refuses_record_level_output_in_git_and_overwrite(tmp_path):
    z = make_zip(tmp_path / "a.zip")
    r = run("--zip", z, "--out", ROOT / "docs" / "pl.csv.gz")
    assert r.returncode != 0 and "data/local" in r.stderr
    out = tmp_path / "pl.csv.gz"
    out.write_bytes(b"x")
    r = run("--zip", z, "--out", out)
    assert r.returncode != 0 and "not overwritten" in r.stderr


def test_refuses_archive_without_the_panel_file(tmp_path):
    z = make_zip(tmp_path / "a.zip", member="other.csv")
    r = run("--zip", z, "--out", tmp_path / "pl.csv.gz")
    assert r.returncode != 0 and "safepanel_allrounds.csv" in r.stderr
