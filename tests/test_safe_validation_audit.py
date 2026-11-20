"""SAFE validation audit (scripts/safe_validation_audit.py): value cleaning, Cramer's V, the signal check (a field
that determines the answer gives a large gain, noise gives none) and the refusal of a wave column answered in another
wave. The SAFE extract is not in git, so the checks run on synthetic rows."""
import csv
import gzip
import io
import random
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import safe_validation_audit as S  # noqa: E402


def test_norm_and_cramers_v():
    assert S.norm("2.0") == "2" and S.norm(" . ") == "" and S.norm("99.0") == "99"
    assert S.cramers_v([("a", "x"), ("b", "y")] * 10) == pytest.approx(1.0)
    assert S.cramers_v([("a", "x"), ("a", "y"), ("b", "x"), ("b", "y")] * 10) == pytest.approx(0.0)
    assert S.cramers_v([("a", "x")] * 5) is None


def _rows(n, seed, informative):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        y = rng.choice("123")
        f = y if informative else rng.choice("123")
        r = {c: "" for c in S.SIGNAL_FEATURES}
        r.update({"permid": str(i), "d1_rec": rng.choice("1234"), "d3_rec": rng.choice("1234"), "q4_a_rec": f, "y": y})
        out.append(r)
    return out


def test_signal_separates_information_from_noise():
    lab = lambda r: r["y"]  # noqa: E731
    good = S.signal(_rows(600, 1, True), lab, S.SIGNAL_FEATURES)
    noise = S.signal(_rows(600, 2, False), lab, S.SIGNAL_FEATURES)
    assert good["gain_best_over_marginal"] > 0.5
    assert abs(noise["gain_best_over_marginal"]) < 0.03
    assert good["n"] == 600 and good["categories"] == ["1", "2", "3"]


def test_refuses_a_column_answered_in_another_wave(tmp_path):
    cols = ["permid", "wave", "experiment_1", "d1_rec", "qa21_2024q3", "qa21_2025q3", "qa21_2026q3"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, cols)
    w.writeheader()
    w.writerow({"permid": "1", "wave": "32", "experiment_1": "2", "d1_rec": "1", "qa21_2024q3": "1"})
    w.writerow({"permid": "2", "wave": "36", "experiment_1": "2", "d1_rec": "1", "qa21_2024q3": "2"})
    w.writerow({"permid": "3", "wave": "40", "experiment_1": "2", "d1_rec": "1", "qa21_2026q3": "3"})
    src = tmp_path / "s.csv.gz"
    src.write_bytes(gzip.compress(buf.getvalue().encode()))
    p = subprocess.run([sys.executable, str(ROOT / "scripts/safe_validation_audit.py"), "--safe", str(src),
                        "--out", str(tmp_path / "o.json")], capture_output=True, text=True)
    assert p.returncode != 0 and "expected only in wave" in p.stderr


def test_signal_check_refuses_a_failed_fit(monkeypatch):
    """safe_signal_check: a fit that the optimiser reports as failed raises; in the inner choice the penalty is not
    eligible, and with no eligible penalty the script stops."""
    import numpy as np
    import safe_signal_check as C

    class R:
        x = np.zeros(6)
        fun = 1.0
        success = False
        status = 2
        message = "ABNORMAL"
        nit = 1
        jac = np.ones(6)

    monkeypatch.setattr(C, "minimize", lambda *a, **k: R())
    with pytest.raises(C.FitFailure):
        C.fit(np.ones((4, 1)), np.array([0, 1, 2, 0]), 3, 0.1, "t")
    assert C.DIAG[-1]["success"] is False and C.DIAG[-1]["message"] == "ABNORMAL"
    rows = [{"permid": str(i), **{f: "1" for f in C.CARD}} for i in range(40)]
    y = np.array([i % 3 for i in range(40)])
    with pytest.raises(SystemExit, match="no penalty"):
        C.nested(rows, y, 3, "logit_main", C.plain_scores, "t")


def test_band_scores_sum_ten_values_into_bands():
    import numpy as np
    import safe_signal_check as C
    cats = [str(i) for i in range(1, 11)]
    p = np.full((1, 10), 0.1)
    assert C.band_scores(cats)(p, np.array([0]))[0] == pytest.approx(np.log(0.3))
    assert C.band_scores(cats)(p, np.array([5]))[0] == pytest.approx(np.log(0.4))
