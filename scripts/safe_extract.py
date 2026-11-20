#!/usr/bin/env python3
"""SAFE validation: make the Polish extract of the ECB SAFE public use file, the source of every SAFE build.

The ECB publishes the SAFE microdata as one archive, ecb.SAFE_microdata.zip
(https://www.ecb.europa.eu/stats/ecb_surveys/safe/shared/pdf/ecb.SAFE_microdata.zip; access under the ECB's SAFE
microdata user agreement: scientific use, no commercial use, no re-identification, no linkage with other firm data,
citation and a copy of the work to the ECB). The extract is every record of safepanel_allrounds.csv with d0 == "PL",
all columns, read and written with pandas (read_csv with its defaults, to_csv without the index, gzip with a fixed
modification time), so the same archive gives the same bytes.

Checked (19.11.2026): the archive with SHA-256 64a118f1... (downloaded 12.11.2026; the server reported it last
modified on 23.10.2026) gives 16,421 records and exactly the extract every build of the validation used
(SHA-256 15fef143..., recorded as source_sha256 in data/reference/safe_frozen.json), with pandas 3.0.5 on Python 3.13
(zlib 1.3). Another pandas or zlib version may write other bytes from the same records; the script then reports the
SHA-256 of the uncompressed CSV as well, to tell a different compression from different records. A newer archive
(the ECB revises the file with each wave) gives a different extract: it can serve a new replication, not a repeat of
the frozen analysis.

    python3 scripts/safe_extract.py --zip data/local/safe/source/ecb.SAFE_microdata.zip \\
        --out data/local/safe/safe_poland_all_waves.csv.gz

Record-level output: --out inside the repository must be under data/local/ (not in git).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import zipfile
from pathlib import Path

VERSION = "0.1"
ROOT = Path(__file__).resolve().parents[1]
MEMBER = "safepanel_allrounds.csv"
COUNTRY = "PL"
# the gzip header's modification time of the extract the validation used (12.11.2026 15:50:01 UTC); a fixed value
# makes the output depend on the records only
MTIME = 1794498601
KNOWN = {"archive_sha256": "64a118f1bf2dddffe03a47b5a283a9b2a8705377f091b4a0b1a60906ccc10fc3",
         "extract_sha256": "15fef1438998053d13765e34c72cdbae6ebe1ed24c3b53ebbf569eb279974f1d",
         "csv_sha256": "5132104fc6655010b1f8433de8a2d4248c2473dc2bc236c62f8c0cd7588e6fc2",
         "records": 16421}


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True, help="the ECB archive ecb.SAFE_microdata.zip")
    ap.add_argument("--out", required=True, help="the extract (.csv.gz)")
    a = ap.parse_args()
    import pandas as pd
    out = Path(a.out).resolve()
    if out.is_relative_to(ROOT) and not out.is_relative_to(ROOT / "data/local"):
        sys.exit("the extract is record-level: --out inside the repository must be under data/local/")
    if out.exists():
        sys.exit(f"{a.out} exists: not overwritten")
    archive = sha_file(Path(a.zip))
    with zipfile.ZipFile(a.zip) as z:
        if MEMBER not in z.namelist():
            sys.exit(f"{a.zip}: no {MEMBER} in the archive")
        with z.open(MEMBER) as f:
            df = pd.read_csv(f)
    if "d0" not in df.columns:
        sys.exit(f"{MEMBER}: no country column d0")
    pl = df[df["d0"] == COUNTRY]
    csv_bytes = pl.to_csv(index=False).encode("utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    pl.to_csv(out, index=False, compression={"method": "gzip", "mtime": MTIME})
    if hashlib.sha256(gzip.decompress(out.read_bytes())).hexdigest() != hashlib.sha256(csv_bytes).hexdigest():
        sys.exit("the written extract does not decompress to the records")  # not expected; a guard
    res = {"script": f"safe_extract {VERSION}", "archive_sha256": archive, "records": int(len(pl)),
           "waves": {str(k): int(v) for k, v in sorted(pl["wave"].value_counts().items())},
           "extract_sha256": sha_file(out), "csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
           "pandas": pd.__version__, "python": sys.version.split()[0]}
    res["same_as_the_validation"] = {k: res[k] == KNOWN[k] for k in ("archive_sha256", "extract_sha256", "csv_sha256",
                                                                       "records")}
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
