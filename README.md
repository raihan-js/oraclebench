# OracleBench

![OracleBench results](images/oraclebench.png)

Grade small open LLM-as-judge setups against deterministic oracles — and ship the harness that makes most judge calls unnecessary.

Scope, first: both judges are ≤3B local models (Qwen2.5-3B-Instruct and Qwen2.5-0.5B-Instruct). Nothing here carries over to frontier judges. Natural errors and synthetic corruptions are always reported separately.

## What changed on 2026-10-05

The first version's GSM8K "wrong" labels came from a FlipGate run with a 256-token generation cap; 51 of its 398 "wrong" answers were actually right. FlipGate was re-run with a 1,024-token cap (0% truncated), the item bank was rebuilt from that run, and every judge stage was re-run. Two other flaws surfaced while re-checking and were fixed: the pairwise probe paired a wrong answer with a correct answer to a *different* question, and the self-preference comparison compared different question sets. The earlier numbers (3B 13.2%, 0.5B 36.1%; "pairwise broken"; "no self-preference") are superseded by everything below. The inputs of the first version are kept on Hugging Face under their original revision.

## Results (1,760 oracle-checked items: 511 natural errors, 822 synthetic corruptions, 427 correct)

All numbers are recomputed from the per-item files by `python scripts/summarize.py` (`results/summary.json`).

### Pointwise: false-accept (judge says CORRECT on an oracle-wrong answer)

| Judge | Overall | 95% CI | GSM8K | IFEval | FedProc |
|---|---|---|---|---|---|
| Qwen2.5-3B | 11.0% (147/1,333) | [9.5%, 12.8%] | 10.1% | 21.4% | 0.5% |
| Qwen2.5-0.5B | 40.6% (541/1,333) | [38.0%, 43.2%] | 94.8% | 13.7% | 1.0% |

True-accept on correct answers: 3B 84.0% (GSM8K), 60.7% (IFEval), 0.0% (FedProc); 0.5B 100.0%, 28.0%, 0.0%.

- **The 0.5B judge rubber-stamps arithmetic**: 94.8% false-accept and 100% true-accept on GSM8K. It is agreeing, not judging.
- **Both judges blanket-reject clause citations**: 0.5% and 1.0% false-accept look excellent until you see 0.0% true-accept. Low false-accept through zero discrimination.
- **Synthetic corruptions flatter the judge**: the 3B judge falsely accepts 3.3% of synthetic corruptions (27/822) but 23.5% of natural errors (120/511), about 7×. Keep a natural slice.
- Unparseable judge output is counted as "not accepted" (3B: 5 cases, 0.5B: 35).

### Pairwise, with matched pairs (wrong and right answer to the same prompt)

For each pair the judge sees both orders, plain and with the right answer padded with filler. 236 pairs, 944 judgements per judge: GSM8K and IFEval pairs are minimal pairs (the corruption changes one number or breaks one rule), FedProc pairs are a natural wrong answer vs the correct one for the same record.

| Judge | Picks the right answer | Right answer in slot B | Right answer in slot A | Right answer padded |
|---|---|---|---|---|
| Qwen2.5-3B | **80.5%** [76.6%, 83.8%] | 94.1% | 66.8% | 74.1% |
| Qwen2.5-0.5B | 53.8% [49.3%, 58.3%] | 86.9% | 20.8% | 53.7% |

- The 3B judge discriminates (87.8% on GSM8K pairs, 87.4% on IFEval, 56.5% on FedProc) but has a clear position bias: 94% when the right answer is in slot B, 67% when it is in slot A.
- The 0.5B judge is at chance on correctness (53.8%) and picks slot B 83% of the time: pure position.
- Padding the right answer with filler lowers the 3B judge's pick-right rate by about 6 points.
- An earlier version of this probe paired a wrong answer with a right answer to a *different* question. That version only measures position bias (the judges picked B 76% and 91% of the time) and wrongly suggested the 3B judge cannot discriminate. The unmatched files are kept (`*_pairwise.jsonl`) and labelled as a position probe.

### Self-preference (do judges accept their own wrong answers more?)

| Judge | Comparison | Own model's wrong answers | Other model's wrong answers | Test |
|---|---|---|---|---|
| Qwen2.5-3B | different questions | 19.7% (40/203) [14.8%, 25.7%] | 0.9% (1/113) [0.2%, 4.8%] | Fisher p = 1.0e-7 |
| Qwen2.5-3B | **same 165 questions** (both models wrong) | **14.5%** (24/165) [10.0%, 20.7%] | 5.5% (9/165) [2.9%, 10.0%] | exact McNemar p = 0.0015 (18 vs 3 discordant) |
| Qwen2.5-0.5B | same 165 questions | 92.7% (153/165) | 92.7% (153/165) | p = 1.0 |

The first comparison confounds own-vs-other with question difficulty, so the 20-fold gap is not evidence. Holding the questions fixed (`scripts/selfpref_matched.py`), the 3B judge still accepts its own wrong answers 2.6× as often as the 0.5B model's wrong answers to the same questions. That is consistent with self-preference, but this design cannot separate it from the 3B model's errors being subtler than the 0.5B model's. The 0.5B judge accepts nearly everything, so there is nothing to detect. A cross-family judge would settle it; none was tested.

### Checker-first harness (oracles first, judge only on uncovered items)

| Setup | Errors | Judge calls | Judge GPU-s | Judge tokens |
|---|---|---|---|---|
| Harness | 0/1,760 (0.0%) | 100 | 22 | 300 |
| Judge-only | 412/1,760 (23.4%) | 1,760 | 448 | 5,403 |

94.6% of the 1,860 items route to a checker. 17.6× fewer judge calls and about 20× less judge time. The 0 errors are by construction: where an oracle applies it is the ground truth. The harness's value is the routing and the per-decision traces, not the zero.

## Status

21 pytest tests; results above; [dataset on Hugging Face](https://huggingface.co/datasets/raihan-js/oraclebench-items). See `AGENTS.md`.

```bash
python scripts/summarize.py                         # every table above
python scripts/selfpref_recheck.py                  # unmatched self-preference
python scripts/selfpref_matched.py generate         # GPU; then run_judges.py on the output, then:
python scripts/selfpref_matched.py analyze
python scripts/run_pairwise.py --judge qwen3b --matched --n 300
```

## Limitations (read before citing)

- Judges are ≤3B local models. Frontier judges may behave very differently.
- Judge prompts are frozen (v1); different prompts give different numbers.
- Natural GSM8K errors come from one model (Qwen2.5-3B, bf16, 1,024-token cap); IFEval and FedProc natural errors from the same family.
- Self-preference could not be separated from error subtlety; all judges are Qwen models.
- FedProc 0.5% / 1.0% false-accept is blanket rejection, not discrimination (0.0% true-accept).
- Pairwise: 236 matched pairs, FedProc has only 54; one prompt template.
- The robust GSM8K extractor is a heuristic, but it finds the right answer in 0 of the 503 oracle-wrong GSM8K items in the rebuilt bank.

## License

MIT
