import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from selfpref_matched import mcnemar_exact  # noqa: E402
from summarize import rate, wilson  # noqa: E402


def test_wilson_matches_known_interval():
    lo, hi = wilson(50, 100)
    assert lo == pytest.approx(40.4, abs=0.2) and hi == pytest.approx(59.6, abs=0.2)


def test_wilson_handles_extremes():
    lo, hi = wilson(0, 30)
    assert lo == 0.0 and 8.0 < hi < 14.0
    lo, hi = wilson(30, 30)
    assert hi == 100.0 and 86.0 < lo < 92.0
    assert wilson(0, 0) == (0.0, 0.0)


def test_rate_reports_counts():
    r = rate(147, 1333)
    assert (r["k"], r["n"], r["pct"]) == (147, 1333, 11.0)


def test_mcnemar_exact_known_values():
    assert mcnemar_exact(18, 3) == pytest.approx(0.00149, abs=1e-4)  # the matched self-preference result
    assert mcnemar_exact(6, 6) == 1.0
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 0) == pytest.approx(0.0625)
