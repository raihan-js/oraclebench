"""Tests for the checker-first harness routing logic."""

import pytest
from oraclebench.harness import CheckerFirstHarness
from oraclebench.oracles.gsm8k import GSM8KScorer


class FakeScorer:
    def __init__(self, verdict):
        self._verdict = verdict
        self.calls = 0

    def score(self, prompt, answer, reference):
        self.calls += 1
        return 1.0 if self._verdict else 0.0


class TestRouting:
    def test_checker_covers(self):
        h = CheckerFirstHarness({"gsm8k": FakeScorer(True)})
        out = h.decide({"oracle": "gsm8k", "prompt": "p", "answer": "a",
                        "reference": {}, "provenance": "x"})
        assert out["verdict"] is True
        assert out["verdict_source"] == "checker"
        assert out["checker"] == "gsm8k"

    def test_no_checker_no_judge(self):
        h = CheckerFirstHarness({})
        out = h.decide({"oracle": None, "prompt": "p", "answer": "a",
                        "reference": {}, "provenance": "x"})
        assert out["verdict"] is None
        assert out["verdict_source"] == "no-coverage-no-judge"

    def test_falls_through_to_judge(self):
        calls = []

        def judge(prompt, answer, ref):
            calls.append((prompt, answer))
            return True, 0.5, 42

        h = CheckerFirstHarness({}, judge_fn=judge)
        out = h.decide({"oracle": None, "prompt": "p", "answer": "a",
                        "reference": {}, "provenance": "x"})
        assert out["verdict"] is True
        assert out["verdict_source"] == "judge"
        assert out["judge_seconds"] == 0.5
        assert out["judge_tokens"] == 42
        assert len(calls) == 1

    def test_judge_never_called_when_covered(self):
        def judge(prompt, answer, ref):
            raise AssertionError("judge must not be called")

        h = CheckerFirstHarness({"gsm8k": FakeScorer(False)}, judge_fn=judge)
        out = h.decide({"oracle": "gsm8k", "prompt": "p", "answer": "a",
                        "reference": {}, "provenance": "x"})
        assert out["verdict"] is False
        assert out["verdict_source"] == "checker"

    def test_real_gsm8k_oracle_end_to_end(self):
        h = CheckerFirstHarness({"gsm8k": GSM8KScorer()})
        out = h.decide({"oracle": "gsm8k", "prompt": "What is 2+2?",
                        "answer": "#### 4", "reference": {"answer": "#### 4"},
                        "provenance": "test"})
        assert out["verdict"] is True

    def test_summary_counts(self):
        h = CheckerFirstHarness({"gsm8k": FakeScorer(True)})
        h.decide({"oracle": "gsm8k", "prompt": "p", "answer": "a",
                  "reference": {}, "provenance": "x"})
        h.decide({"oracle": None, "prompt": "p", "answer": "a",
                  "reference": {}, "provenance": "y"})
        s = h.summary()
        assert s == {"checker": 1, "judge": 0, "uncovered_no_judge": 1,
                     "total": 2, "checker_share": 0.5}
