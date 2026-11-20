# SAFE profile validation: does a firm profile help AI agents forecast real survey answers?

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

Archived on Zenodo: https://doi.org/10.5281/zenodo.24396236 (all versions).

Version 1.0.2. Built from commit 674af667e17dbd2572afa66e60369372dd10f33d of the project repository.
