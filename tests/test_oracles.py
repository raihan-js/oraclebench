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
