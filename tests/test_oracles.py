"""Tests for the three oracle scorers.

Oracles copied verbatim from ../flipgate/src/flipgate/scorers/
(2026-10-01) with only the import path changed (flipgate.* -> oracles
relative imports). Any behavior change vs flipgate is a bug: these tests
pin the oracle semantics both projects share.
"""

import pytest
from oraclebench.oracles.gsm8k import GSM8KScorer
from oraclebench.oracles.ifeval import IFEvalScorer
from oraclebench.oracles.fedproc import FedProcRegistryScorer


@pytest.fixture
def registry():
    return {"52.203-1", "52.212-4", "252.203-1"}


class TestGSM8KOracle:
    def test_correct(self):
        s = GSM8KScorer()
        assert s.score("", "Reasoning...\n#### 42", {"answer": "#### 42"}) == 1.0

    def test_incorrect(self):
        s = GSM8KScorer()
        assert s.score("", "#### 43", {"answer": "#### 42"}) == 0.0

    def test_binary(self):
        s = GSM8KScorer()
        assert s.score("", "no numbers here", {"answer": "#### 42"}) == 0.0


class TestIFEvalOracle:
    def test_no_comma(self):
        s = IFEvalScorer()
        ref = {"instruction_id_list": ["punctuation:no_comma"], "kwargs": [{}]}
        assert s.score("", "Hello world", ref) == 1.0
        assert s.score("", "Hello, world", ref) == 0.0

    def test_partial_credit(self):
        s = IFEvalScorer()
        ref = {"instruction_id_list": ["punctuation:no_comma", "keywords:existence"],
               "kwargs": [{}, {"keywords": ["python"]}]}
        assert s.score("", "I like java", ref) == 0.5


class TestFedProcOracle:
    def test_valid(self, registry):
        s = FedProcRegistryScorer(registry=registry)
        assert s.score("", "Per FAR 52.203-1, ...", {}) == 1.0

    def test_fabricated(self, registry):
        s = FedProcRegistryScorer(registry=registry)
        assert s.score("", "Per FAR 99.999-9, ...", {}) == 0.0

    def test_hallucinated_list(self, registry):
        s = FedProcRegistryScorer(registry=registry)
        assert s.get_hallucinated_clauses("FAR 52.203-1 and FAR 99.999-9") == ["99.999-9"]


class TestGSM8KOracleV2:
    ref = {"answer": "work\n#### 1,250"}

    def test_boxed_and_thousands_separator(self):
        from oraclebench.oracles.gsm8k import GSM8KScorer
        assert GSM8KScorer().score("q", "so \\(\\boxed{1,250}\\)", self.ref) == 1.0

    def test_v1_is_kept_and_misreads_commas(self):
        from oraclebench.oracles.gsm8k import GSM8KScorerV1
        assert GSM8KScorerV1().extract_answer("the answer is 1,250") == 1.0

    def test_truncated_text_scores_zero(self):
        from oraclebench.oracles.gsm8k import GSM8KScorer
        assert GSM8KScorer().score("q", "so the total is 80,000 + 120,0", self.ref) == 0.0
