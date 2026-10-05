![oraclebench results](https://raw.githubusercontent.com/raihan-js/oraclebench/HEAD/images/oraclebench.png)

# Small LLM Judges Approved 11% and 41% of Wrong Answers. Then I Fixed My Own Pairwise Test.

*Grading small open judges against deterministic oracles, plus the harness that makes most judge calls unnecessary.*

---

> Scope note, first paragraph as promised: every judge here is ≤3B parameters running locally. Nothing below carries over to frontier judges. Synthetic corruptions and natural errors are reported separately throughout.

## The Setup

Everyone uses LLM-as-judge. Existing benchmarks (JudgeBench, RewardBench) already measure judges in general. What they don't have is what I have: a **real legal oracle**. My definition of a hallucinated FAR clause, "a clause number not in the FAR/DFARS registry (1,128 entries from eCFR Title 48)", needs no human and no judge. It is ground truth by construction.

OracleBench grades two small open judges (Qwen2.5-3B and Qwen2.5-0.5B) against three deterministic oracles: the FAR/DFARS registry, GSM8K arithmetic, and 25 rule-based IFEval checkers. The item bank has 1,760 verified items (511 natural errors, 822 synthetic corruptions, 427 correct answers), each with frozen provenance.

## What I had to redo

The first version of this article is not the one you are reading. Re-checking my own raw data turned up three problems:

1. **The GSM8K "wrong" labels were partly wrong.** They came from a FlipGate run that capped generation at 256 tokens. Re-scoring with a robust answer extractor found the right answer in 51 of the 398 items I had labelled wrong. I re-ran FlipGate with a 1,024-token cap (0% truncated), rebuilt the item bank from it, and re-ran every judge stage.
2. **The pairwise probe was flawed.** It paired a wrong answer with a correct answer to a *different* question, so it could only measure position bias, and I wrongly concluded the judges "cannot evaluate".
3. **The self-preference comparison confounded own-vs-other with question difficulty.**

All numbers below are from the rebuilt bank and the fixed probes.

## False-accept rates

False-accept is the share of oracle-wrong answers the judge called CORRECT. The headline is dominated by arithmetic for the small judge, so read the columns.

| Judge | Overall | 95% CI | GSM8K | IFEval | FedProc |
|---|---|---|---|---|---|
| Qwen2.5-3B | 11.0% (147/1,333) | [9.5%, 12.8%] | 10.1% | 21.4% | 0.5% |
| Qwen2.5-0.5B | 40.6% (541/1,333) | [38.0%, 43.2%] | 94.8% | 13.7% | 1.0% |

Three things stand out:

1. **The 0.5B judge rubber-stamps arithmetic**: 94.8% false-accept and 100% true-accept on GSM8K. It is agreeing, not judging.
2. **Both judges blanket-reject clause citations.** 0.5% and 1.0% false-accept look great until you see 0.0% true-accept: they reject every clause answer, right or wrong. Low false-accept through zero discrimination.
3. **Synthetic corruptions flatter the judge.** The 3B judge falsely accepts 3.3% of synthetic corruptions but 23.5% of natural errors, about 7×. If you only test judges on obvious corruptions, you will think they are better than they are. Keep a natural slice.

## Pairwise judging, with matched pairs

For each pair the judge saw both orders, plain and with the right answer padded with filler. This time the right answer answers the *same* prompt as the wrong one: GSM8K and IFEval pairs are minimal pairs (one number changed, one rule broken), FedProc pairs are a natural wrong answer against the correct one for the same record. 236 pairs, 944 judgements per judge.

| Judge | Picks the right answer | Right answer in slot B | Right answer in slot A | Right answer padded |
|---|---|---|---|---|
| Qwen2.5-3B | **80.5%** [76.6%, 83.8%] | 94.1% | 66.8% | 74.1% |
| Qwen2.5-0.5B | 53.8% [49.3%, 58.3%] | 86.9% | 20.8% | 53.7% |

The 3B judge does discriminate (87.8% on GSM8K pairs, 87.4% on IFEval, 56.5% on FedProc), but it leans toward slot B: 94% when the right answer is there, 67% when it is in slot A. The 0.5B judge is at chance on correctness and picks slot B 83% of the time. Padding the right answer with filler costs the 3B judge about 6 points. The old, flawed probe had made both judges look equally broken ("pick B 85-92%"); fixing it changes the 3B conclusion from "cannot evaluate" to "can, with a position bias you should randomise away".

## Self-preference

A common worry is that judges favour their own outputs. The 3B judge accepts the 3B model's wrong answers more than the 0.5B model's:

| Judge | Comparison | Own model's wrong answers | Other model's wrong answers | Test |
|---|---|---|---|---|
| Qwen2.5-3B | different questions | 19.7% (40/203) | 0.9% (1/113) | Fisher p = 1e-7 |
| Qwen2.5-3B | **same 165 questions**, both models wrong | **14.5%** (24/165) [10.0%, 20.7%] | 5.5% (9/165) [2.9%, 10.0%] | exact McNemar p = 0.0015 |
| Qwen2.5-0.5B | same 165 questions | 92.7% | 92.7% | p = 1.0 |

The first row is the comparison I would have reported: a 20-fold gap. It is not evidence, because the two sets are different questions with different difficulty. The second row holds the questions fixed (both models answered the same 165 questions wrongly), and the gap shrinks to 2.6× while staying significant. That is consistent with self-preference, but this design cannot separate it from the 3B model's errors being subtler than the 0.5B model's. The 0.5B judge approves nearly everything, so there is nothing to detect. Only same-family judges were tested; a cross-family judge would settle it.

## The harness: checkers first, judges last

The constructive result. A small router: run the applicable checkers first, call the judge only where no checker covers the item, trace every decision.

| Setup | Errors | Judge calls | Judge GPU-s | Judge tokens |
|---|---|---|---|---|
| Checker-first | **0/1,760 (0.0%)** | 100 | 22 | 300 |
| Judge-only | 412/1,760 (23.4%) | 1,760 | 448 | 5,403 |

94.6% of the 1,860 items route to checkers: **17.6× fewer judge calls and about 20× less judge time**. The zero errors are by construction, since an oracle is ground truth where it applies; the point is the routing and the per-decision trace, which tell you which answers still need a judge.

## Limitations

- Judges ≤3B and local; no frontier claims.
- Judge prompts frozen (v1 in the repo); different prompts, different numbers.
- Natural GSM8K errors come from one model (Qwen2.5-3B, bf16, 1,024-token cap); all judges are Qwen models, so self-preference is confounded with error subtlety.
- FedProc's low false-accept is blanket rejection, stated plainly.
- 236 matched pairs, only 54 of them FedProc; one pairwise template.

---

*Repo: github.com/raihan-js/oraclebench · Data: huggingface.co/datasets/raihan-js/oraclebench-items · 21 tests green.*
