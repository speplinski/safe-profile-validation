#!/usr/bin/env python3
"""Build the release directory of the SAFE validation (code, plan, frozen settings, aggregate reports, documentation and
the paper's sources) for a separate repository and its Zenodo archive, in the manner of the WorkTrace archive
(10.5281/zenodo.23932329). No model call, no network.

Only the files listed here are copied. NOT included, by the SAFE terms: the SAFE microdata, any card, build, key or run
(record-level, under data/local/), and the model answers by item (the project lead's decision of 19.11.2026). Also not
included: the saved third-party web pages of the terms (their manifest with URLs and sha256 is). Every number of the
paper's tables can be read from the aggregate reports with scripts/safe_paper_numbers.py; a new replication needs SAFE
access from the ECB and new model runs (the README's three levels of reproduction).

    python3 scripts/make_safe_release.py --out <empty directory outside the repository> --version 1.0
    python3 scripts/make_safe_release.py --check <release directory> [--staged]
    python3 scripts/make_safe_release.py --refresh <release directory>

--refresh: copies the current version of every package file (FILES, the population modules, the compiled paper when
present) from the project into an existing release, keeps the release's own README.md, requirements.txt,
CITATION.cff, .zenodo.json and .gitignore as they are (edit them by hand), updates the "Built from commit" line of its
README when there is one, and rewrites MANIFEST.sha256.json; then --check passes on it.

--check (before publishing): the release directory must hold exactly the files of its MANIFEST.sha256.json, each with
that sha256, and the manifest may name only files this package defines (FILES, the population modules, the generated
README.md, requirements.txt, CITATION.cff, .zenodo.json and the compiled paper) and nothing that looks record-level.
With --staged the files staged for commit under that directory (git index) are checked instead of the working tree:
every staged file must be in the manifest with the same sha256, and no manifest file may be missing from the index.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.1"
PAPER_TITLE = ("Does a Firm Profile Help an AI Agent Forecast Real Survey Answers? A Prospectively Specified Validation on "
               "the EC/ECB Survey on the Access to Finance of Enterprises for Polish Firms")

FILES = [
    # code: the SAFE pipeline, the runner and what the runner imports
    "scripts/safe_prompt.py", "scripts/safe_build.py", "scripts/safe_analyze.py", "scripts/safe_signal_check.py",
    "scripts/safe_validation_audit.py", "scripts/safe_assumption_check.py", "scripts/safe_test_level_sim.py",
    "scripts/safe_format_compare.py", "scripts/safe_elicitation_compare.py", "scripts/safe_h1_qualification.py",
    "scripts/safe_make_frozen.py", "scripts/safe_preflight.py", "scripts/safe_paper_numbers.py", "scripts/make_safe_release.py",
    "scripts/run_b_k2.py", "scripts/llm_serving_check.py", "scripts/ollama_provenance.py",
    "scripts/b_k2_plan.py", "scripts/b_k2_prompt.py", "scripts/profile_check.py", "scripts/b_k2_input_identity.py",
    "scripts/b_k2_run_report.py", "scripts/safe_extract.py",
    "src/dnba/__init__.py",
    # tests that run without the microdata (synthetic extracts and the aggregate reports)
    "tests/test_safe_prompt.py", "tests/test_safe_analyze.py", "tests/test_safe_elicitation_compare.py",
    "tests/test_safe_freeze.py", "tests/test_safe_validation_audit.py", "tests/test_safe_extract.py",
    # plan, card key, frozen settings, prices used for the cost caps, the terms' manifest
    "data/reference/safe_validation_plan.json", "data/reference/safe_card_key.json", "data/reference/safe_frozen.json",
    "data/reference/anthropic_pricing_2026-11-15.json", "data/reference/openai_pricing_2026-11-17.json",
    "data/reference/sources/2026-11-18-safe-terms/manifest.json",
    "data/reference/sources/2026-11-12-safe/manifest.json",  # the download record of the ECB archive
    # aggregate reports
    "data/processed/safe_validation_audit_v0.1.json", "data/processed/safe_assumption_check_v0.1.json",
    "data/processed/safe_signal_check_v0.3.json", "data/processed/safe_test_level_sim_v0.2.json",
    "data/processed/safe_dev_pilot_v0.1.json", "data/processed/safe_dev_pilot2_v0.1.json",
    "data/processed/safe_dev_pilot2_calibration_v0.1.json", "data/processed/safe_dev_pilot2_diagnosis_v0.1.json",
    "data/processed/safe_elicitation_compare_v0.1.json", "data/processed/safe_dev_pilot3_v0.1.json",
    "data/processed/safe_dev_pilot3_calibration_v0.1.json", "data/processed/safe_h1_family_qualification_v0.1.json",
    "data/processed/safe_test_v0.1.json",
    # the paper's sources (the validation log and the paper checklist are in Polish and stay in the project repository)
    "docs/paper/safe-validation/main.tex",
    "LICENSE",
]
DIRS = ["src/dnba/population"]  # the population modules the runner imports (code only)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


README = """# SAFE profile validation: does a firm profile help AI agents forecast real survey answers?

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24396236.svg)](https://doi.org/10.5281/zenodo.24396236)

Code, frozen analysis plan, aggregate results and paper sources for the prospectively specified validation described in
*Does a Firm Profile Help an AI Agent Forecast Real Survey Answers? A Prospectively Specified Validation on the EC/ECB
Survey on the Access to Finance of Enterprises for Polish Firms* (Szymon P. Pepliński, Generatywnie, 2026;
`docs/paper/safe-validation/`).

**Data.** This paper uses data from the EC/ECB Survey on the access to finance of enterprises. Data source:
EC/ECB Survey on the access to finance of enterprises (SAFE). The SAFE microdata are available from the European
Central Bank (https://www.ecb.europa.eu/stats/ecb_surveys/safe/html/data.en.html) under its user agreement
(https://www.ecb.europa.eu/services/data-protection/privacy-statements/html/SAFE_microdata_disclaimer_and_user_agreement.en.html)
and are **not** included: no SAFE record, no card, build, key or model answer by item is in this archive. No
record-level SAFE data were combined with any other firm data. The tables in `data/processed/` are aggregates compiled
from these data; the same source applies to them.

**Reproduction: three different things.**

1. *Reading the tables (no API, no SAFE microdata).*

   ```bash
   python3 scripts/safe_paper_numbers.py
   ```

   prints every number of the paper's tables from the aggregate reports in `data/processed/` and checks that the
   confirmatory report was made with the frozen settings, the plan and the analysis script in this archive (sha256).
   It reads the stored estimates and intervals; it does not recompute them.

2. *Recomputing the estimates from the historical model answers.* The item-level answers of the confirmatory runs are
   derived from SAFE records and are not published. Recomputing needs them from the author, SAFE access and
   `scripts/safe_analyze.py --frozen data/reference/safe_frozen.json`.

3. *A new replication with new model calls.* Download the ECB archive `ecb.SAFE_microdata.zip` under the SAFE
   microdata user agreement and make the Polish extract:

   ```bash
   python3 scripts/safe_extract.py --zip data/local/safe/source/ecb.SAFE_microdata.zip --out data/local/safe/safe_poland_all_waves.csv.gz
   ```

   With the archive version used here (SHA-256 `64a118f1…`, downloaded on 12.11.2026, see
   `data/reference/sources/2026-11-12-safe/manifest.json`) and pandas 3.0.5 the extract is byte-identical to the one
   of the study (SHA-256 `15fef143…`, `source_sha256` in `data/reference/safe_frozen.json`); the script also prints the
   SHA-256 of the uncompressed CSV, to tell a different compression from different records. The ECB revises the
   archive with each wave, so a later download may give a different extract. `scripts/safe_build.py` with the command in
   `safe_frozen.json` then rebuilds the test items (their sha256 is in the frozen file); new runs use
   `scripts/run_b_k2.py` with the frozen settings. Models served behind API aliases may have changed and need not give
   the same answers, and `safe_analyze.py --frozen` refuses runs on a serving stack other than the frozen one. A new
   replication should be analysed under its own frozen settings, not reported as the original run.

The confirmatory report did not record the versions of the numerical libraries. The tests of this archive pass with
Python 3.13, NumPy 2.5.3, SciPy 1.18.1 and pandas 3.0.5.

**Chronology.** The order of freezing and confirmatory runs is documented by `data/reference/safe_frozen.json` (with the
approval date entered by the project lead) and the checksums that bind the analysis to it; the project's validation
log (in Polish) and commit history are private and available from the author on request. The frozen code and the card key
mention `docs/walidacja-safe.md`: that is this log, not included here; frozen files are not edited after freezing. The plan was not registered with an independent registry.

**Tests** (synthetic extracts and the aggregate reports): `python3 -m pytest tests`.

**Contents.**

| Path | What |
| --- | --- |
| `scripts/safe_*.py`, `scripts/run_b_k2.py`, `scripts/llm_serving_check.py` | extract, cards, elicitation, runner, analysis, calibration, freezing |
| `data/reference/safe_validation_plan.json` | the analysis plan, frozen before confirmatory model evaluation, with its development amendments |
| `data/reference/safe_frozen.json` | the frozen settings of the confirmatory run (test set, models, K, materials' sha256) |
| `data/processed/safe_*.json` | aggregate reports of the audit, three development pilots, the elicitation comparison, the calibration and the test |
| `docs/paper/safe-validation/` | the paper's LaTeX sources |

**Licences.** Code: Apache-2.0 (`LICENSE`). Texts, the plan, the reports and the paper: CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/). The SAFE data remain under the ECB's terms.

Version {version}. Built from commit {commit} of the project repository.
"""


GENERATED = ("README.md", "requirements.txt", "CITATION.cff", ".zenodo.json", ".gitignore")
GITIGNORE = ("# local data never published: SAFE extracts, cards, builds, keys, runs\n**/data/local/\n"
             "__pycache__/\n.pytest_cache/\n.DS_Store\n"
             "# LaTeX build files\n*.aux\n*.log\n*.out\n*.toc\n*.synctex.gz\n*.fls\n*.fdb_latexmk\n")
OPTIONAL = ("docs/paper/safe-validation/main.pdf",)
RECORD_LEVEL = ("data/local", "safe_runs", ".csv.gz", "items.jsonl", "answers.jsonl", "/key.json", "run.json")
IGNORED = {".git", ".DS_Store", "MANIFEST.sha256.json", "__pycache__", ".pytest_cache"}  # never published (gitignored)


def allowed_files() -> set:
    return (set(FILES) | {str(p.relative_to(ROOT)) for d in DIRS for p in (ROOT / d).glob("*.py")} | set(GENERATED)
            | set(OPTIONAL))


def check(d: Path, staged: bool) -> None:
    """The release directory (or what is staged of it) against its manifest and the package's definition."""
    import subprocess
    mf = d / "MANIFEST.sha256.json"
    if staged:  # the manifest the commit will hold: from the git index, not from the disk (project lead, 19.11.2026)
        top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=d, capture_output=True, text=True).stdout.strip()
        rel_mf = str(mf.resolve().relative_to(Path(top).resolve()))
        shown = subprocess.run(["git", "show", f":{rel_mf}"], cwd=top, capture_output=True)
        if shown.returncode != 0:
            sys.exit(f"{rel_mf} is not staged: stage the manifest together with the files")
        manifest = json.loads(shown.stdout)
    else:
        if not mf.is_file():
            sys.exit(f"{mf} missing: build the release with --out first")
        manifest = json.loads(mf.read_text())
    problems = []
    outside = sorted(f for f in manifest if f not in allowed_files())
    if outside:
        problems.append(f"files the package does not define: {outside}")
    rec = sorted(f for f in manifest if any(x in "/" + f for x in RECORD_LEVEL))
    if rec:
        problems.append(f"record-level looking files: {rec}")
    if staged:
        prefix = str(d.resolve().relative_to(Path(top).resolve()))
        prefix = "" if prefix == "." else prefix + "/"
        ls = subprocess.run(["git", "ls-files", "--stage", "--", str(d)], cwd=top, capture_output=True, text=True).stdout
        got = {}
        for ln in ls.splitlines():
            meta, path = ln.split("\t", 1)
            rel = path[len(prefix):] if path.startswith(prefix) else path
            if Path(rel).name in IGNORED or rel.startswith(".git/"):
                continue
            blob = subprocess.run(["git", "cat-file", "-p", meta.split()[1]], cwd=top, capture_output=True).stdout
            got[rel] = hashlib.sha256(blob).hexdigest()
    else:
        got = {str(p.relative_to(d)): sha(p) for p in d.rglob("*")
               if p.is_file() and not any(part in IGNORED for part in p.relative_to(d).parts)}
    extra = sorted(set(got) - set(manifest))
    lost = sorted(set(manifest) - set(got))
    changed = sorted(f for f in set(got) & set(manifest) if got[f] != manifest[f])
    if extra:
        problems.append(f"files not in the manifest: {extra}")
    if lost:
        problems.append(f"manifest files missing{' from the index' if staged else ''}: {lost}")
    if changed:
        problems.append(f"files that differ from the manifest: {changed}")
    if problems:
        sys.exit("release check failed:\n- " + "\n- ".join(problems))
    print(f"release check passed: {len(got)} files{' staged' if staged else ''}, all in the "
          f"{'staged ' if staged else ''}manifest with their sha256")


def write_manifest(out: Path) -> dict:
    manifest = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*"))
                if p.is_file() and not any(part in IGNORED for part in p.relative_to(out).parts)}
    (out / "MANIFEST.sha256.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
    return manifest


def refresh(out: Path) -> None:
    import re
    import subprocess
    if not (out / "MANIFEST.sha256.json").is_file():
        sys.exit(f"{out} is not a release (no MANIFEST.sha256.json); build one with --out")
    files = list(FILES) + sorted(str(p.relative_to(ROOT)) for d in DIRS for p in (ROOT / d).glob("*.py"))
    files += [f for f in OPTIONAL if (ROOT / f).is_file()]
    changed = []
    for f in files:
        src, dst = ROOT / f, out / f
        if not dst.is_file() or sha(dst) != sha(src):
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            changed.append(f)
    gi = out / ".gitignore"  # in a standalone repository data/local/ would protect the root only
    gtext = gi.read_text() if gi.is_file() else ""
    if "**/data/local/" not in gtext.split("\n"):
        gi.write_text(gtext + ("" if gtext.endswith("\n") or not gtext else "\n")
                      + "# local data at any depth (cards, keys, runs, SAFE extracts)\n**/data/local/\n")
        changed.append(".gitignore (**/data/local/)")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    rd = out / "README.md"
    if rd.is_file():
        t = rd.read_text()
        t2 = re.sub(r"Built from commit [0-9a-f]+", f"Built from commit {commit}", t)
        if t2 != t:
            rd.write_text(t2)
            changed.append("README.md (commit line)")
    m = write_manifest(out)
    print(json.dumps({"refreshed": changed, "files": len(m), "commit": commit}, ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out")
    ap.add_argument("--version")
    ap.add_argument("--check", default=None, help="check a release directory against its manifest (see above)")
    ap.add_argument("--staged", action="store_true", help="with --check: the files staged for commit")
    ap.add_argument("--refresh", default=None, help="update an existing release from the project (see above)")
    a = ap.parse_args()
    if a.check:
        check(Path(a.check), a.staged)
        return
    if a.refresh:
        refresh(Path(a.refresh))
        return
    if not a.out or not a.version:
        sys.exit("--out and --version are needed to build a release")
    out = Path(a.out).resolve()
    if out.is_relative_to(ROOT):
        sys.exit("--out must be outside the project repository")
    allowed = {".git", ".DS_Store"}  # a fresh git init; the file macOS Finder writes by itself
    extra = sorted(x.name for x in out.iterdir() if x.name not in allowed) if out.exists() else []
    if extra:
        sys.exit(f"{out} is not empty (only .git and .DS_Store may be there): {extra}")
    files = list(FILES) + sorted(str(p.relative_to(ROOT)) for d in DIRS for p in (ROOT / d).glob("*.py"))
    files += [f for f in OPTIONAL if (ROOT / f).is_file()]  # the compiled paper, when it has been built locally
    missing = [f for f in files if not (ROOT / f).is_file()]
    if missing:
        sys.exit(f"missing files: {missing}")
    bad = [f for f in files if f.startswith("data/local") or "safe_runs" in f]
    if bad:
        sys.exit(f"record-level files in the list: {bad}")
    for f in files:
        dst = out / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, dst)
    import subprocess
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    (out / "README.md").write_text(README.format(title=PAPER_TITLE, version=a.version, commit=commit))
    (out / ".gitignore").write_text(GITIGNORE)
    (out / "requirements.txt").write_text("numpy\nscipy\npandas\npytest\nanthropic==2.8.0\nopenai==3.53.0\n")
    (out / "CITATION.cff").write_text(
        "cff-version: 1.2.0\n"
        f"title: \"SAFE validation: code and data for '{PAPER_TITLE}'\"\n"
        "authors:\n  - family-names: Pepliński\n    given-names: Szymon P.\n    affiliation: Generatywnie\n"
        "    orcid: https://orcid.org/0009-0008-2928-0655\n"
        f"version: {a.version}\ndoi: 10.5281/zenodo.24396236\nlicense: Apache-2.0\n"
        "message: If you use this software or its results, please cite it as below; the SAFE data source must be cited as "
        "'EC/ECB Survey on the access to finance of enterprises'.\n")
    (out / ".zenodo.json").write_text(json.dumps({
        "title": f"SAFE validation: code and data for '{PAPER_TITLE}'",
        "upload_type": "software", "language": "eng", "version": a.version,
        "creators": [{"name": "Pepliński, Szymon P.", "affiliation": "Generatywnie", "orcid": "0009-0008-2928-0655"}],
        "license": "Apache-2.0",
        "keywords": ["AI agents", "language models", "survey simulation", "forecast validation", "proper scoring rules",
                     "prospectively specified", "SAFE", "access to finance", "Polish firms"],
        "description": ("Code, the analysis plan frozen before confirmatory model evaluation with its development amendments, the frozen "
                        "settings, aggregate reports and the paper sources of a validation of "
                        "language-model forecasts of firms' answers to the EC/ECB Survey on the access to finance of "
                        "enterprises (Poland). This paper uses data from the EC/ECB Survey on the access to finance of "
                        "enterprises. The SAFE microdata are not included; every number of the paper's tables can be read "
                        "from the aggregate reports (scripts/safe_paper_numbers.py); the README describes recomputation "
                        "and a new replication. Code Apache-2.0; texts, plan, reports and paper CC BY 4.0."),
    }, ensure_ascii=False, indent=1) + "\n")
    manifest = write_manifest(out)
    print(json.dumps({"out": str(out), "files": len(manifest), "commit": commit}, ensure_ascii=False))


if __name__ == "__main__":
    main()
