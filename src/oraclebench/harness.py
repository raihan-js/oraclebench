"""Checker-first harness: route to oracle when covered, judge otherwise.

The harness embodies one rule: never pay for (or risk) a judge verdict
where a deterministic checker gives ground truth. Every decision emits
a trace: which checker fired (or why none did), judge latency/tokens.
"""

import time
from typing import Any


class CheckerFirstHarness:
    """Routes items to checkers first, judges second.

    Args:
        scorers: dict oracle-name -> scorer with .score(prompt, answer, ref).
        judge_fn: callable(prompt, answer, reference) -> (verdict|None, seconds, tokens).
            Only called when no checker covers the item.
    """

    def __init__(self, scorers: dict, judge_fn=None):
        self.scorers = scorers
        self.judge_fn = judge_fn
        self.coverage = {"checker": 0, "judge": 0, "uncovered_no_judge": 0}

    def decide(self, item: dict) -> dict[str, Any]:
        """Return verdict + full trace for one item."""
        trace: dict[str, Any] = {"oracle": item.get("oracle"),
                                 "provenance": item.get("provenance")}
        oracle = item.get("oracle")
        scorer = self.scorers.get(oracle) if oracle else None

        if scorer is not None:
            t0 = time.time()
            score = scorer.score(item["prompt"], item["answer"], item.get("reference", {}))
            trace.update(checker=oracle, checker_seconds=round(time.time() - t0, 3),
                         verdict=(score == 1.0), verdict_source="checker")
            self.coverage["checker"] += 1
            return trace

        # No checker covers this item: fall through to judge (if available).
        trace["checker"] = None
        if self.judge_fn is None:
            trace.update(verdict=None, verdict_source="no-coverage-no-judge")
            self.coverage["uncovered_no_judge"] += 1
            return trace
        verdict, seconds, tokens = self.judge_fn(
            item["prompt"], item["answer"], item.get("reference", {}))
        trace.update(verdict=verdict, verdict_source="judge",
                     judge_seconds=round(seconds, 2), judge_tokens=tokens)
        self.coverage["judge"] += 1
        return trace

    def summary(self) -> dict:
        total = sum(self.coverage.values())
        return {**self.coverage, "total": total,
                "checker_share": self.coverage["checker"] / max(total, 1)}
