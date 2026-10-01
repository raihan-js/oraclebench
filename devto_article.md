# My LLM Judge Approved a Fake Legal Clause 13% of the Time (and a Smaller One 36%)

*Grading small open judges against deterministic oracles — plus the harness that makes most judge calls unnecessary.*

---

> Scope note, first paragraph as promised: every judge here is ≤3B parameters running locally. Nothing below carries over to frontier judges. Synthetic corruptions and natural errors are reported separately throughout.

## The Setup

Everyone uses LLM-as-judge. Existing benchmarks (JudgeBench, RewardBench) already measure judges in general. What they don't have is what I have: a **real legal oracle**. My definition of a hallucinated FAR clause — "a clause number not in the 1,032-clause eCFR registry" — needs no human and no judge. It's ground truth by construction.

OracleBench grades two small open judges (Qwen2.5-3B and Qwen2.5-0.5B) against three deterministic oracles: the FAR/DFARS registry, GSM8K arithmetic, and 25 rule-based IFEval checkers. Item bank: 1,655 verified items (608 natural errors, 620 synthetic corruptions, 427 correct answers), each with frozen provenance.

## False-Accept Rates

| Judge | Overall | 95% CI | GSM8K | IFEval | FedProc |
|---|---|---|---|---|---|
| Qwen2.5-3B | 13.2% | [11.4%, 15.1%] | 15.8% | 22.1% | 0.5% |
| Qwen2.5-0.5B | 36.1% | [33.4%, 38.7%] | 94.0% | 13.9% | 2.0% |

Three things stand out:

1. **The 0.5B judge rubber-stamps arithmetic** — 94% false-accept on wrong math. It says CORRECT to nearly everything on GSM8K (98.7% true-accept too). It isn't judging; it's agreeing.
2. **Both judges blanket-reject clause citations** — 0.5%/2.0% false-accept looks great until you see 0.0% true-accept. They reject all clause answers, right or wrong. Low false-accept via zero discrimination.
3. **Synthetic corruptions are far easier to catch** (2.7% vs 23.8% for the 3B judge). If you only test judges on obvious corruptions, you flatter them ~9×. Always keep a natural slice.

## Pairwise Judging Collapses

Asked to pick A or B, **both judges pick B 85–92% of the time regardless of correctness** (wrong-pick rate 15% when B is right vs 85%+ when B is wrong — it never evaluates, it position-picks). Padding the correct answer with fluff makes it worse (15%→50% wrong-picks). If you run pairwise evals with small judges, swap positions and average — or better, don't run them at all.

## The Harness: Checkers First, Judges Last

The constructive result. A 60-line router: run applicable checkers first, call the judge only where no checker covers the item, trace every decision.

| Setup | Errors | Judge calls | Judge GPU-s |
|---|---|---|---|
| Checker-first | **0/1655 (0.0%)** | 100 | 15 |
| Judge-only | 429/1655 (25.9%) | 1,655 | 275 |

94.3% of items route to checkers. **18× fewer judge calls at zero error cost** — because oracles are ground truth where they apply. The remaining 100 uncovered items (subjective summary preferences) still need judges; that's the honest boundary of the method.

## Limitations

- Judges ≤3B local; no frontier claims.
- Synthetic corruptions easier than natural errors; slices separate.
- FedProc low-FA is blanket rejection, stated plainly.
- Judge prompts frozen (v1 in repo); different prompts, different numbers.
- Qwen judging Qwen outputs measured; cross-family judging not tested.

---

*Repo: github.com/raihan-js/oraclebench · Data: huggingface.co/datasets/raihan-js/oraclebench-items · 14 tests green. The harness wins by construction wherever a checker exists — the false-accept rates are the results.*
