# OracleBench

Grade small open LLM-as-judge setups against deterministic oracles — and ship the harness that makes most judge calls unnecessary.

## Results

**False-accept rate** (judge says CORRECT on oracle-wrong items):

| Judge | Overall | 95% CI | GSM8K | IFEval | FedProc |
|---|---|---|---|---|---|
| Qwen2.5-3B | 13.2% | [11.4%, 15.1%] | 15.8% | 22.1% | 0.5% |
| Qwen2.5-0.5B | 36.1% | [33.4%, 38.7%] | 94.0% | 13.9% | 2.0% |

The 0.5B judge rubber-stamps arithmetic (94% false-accept) while both judges blanket-reject clause citations (0.0% true-accept). Judge quality is task-specific, not general.

**Pairwise judging is broken**: both judges pick answer B 85–92% of the time regardless of correctness. Verbosity padding on the correct answer makes it worse.

**Self-preference: not detected.** Qwen-3B judge false-accepts 15.8% on Qwen-3B outputs vs 12.3% on Qwen-0.5B outputs (CIs overlap); Qwen-0.5B judge 94.0% vs 92.6% (CIs overlap). Generator identity doesn't matter — judge capability dominates. The 0.5B judge rubber-stamps everything from both generators.

**Checker-first harness** (oracles first, judge only on uncovered):

| Setup | Errors | Judge calls | Judge GPU-s |
|---|---|---|---|
| Harness | 0/1655 (0.0%) | 100 | 15 |
| Judge-only | 429/1655 (25.9%) | 1,655 | 275 |

94.3% of items route to checkers. 18× fewer judge calls at zero error cost — because oracles are ground truth where they apply.

## Status

Milestones 1–3 complete. Write-up pending. See `AGENTS.md`.

## Limitations (read before citing)

- Judges are ≤3B local models — nothing here carries to frontier judges.
- Synthetic corruptions are easier to catch than natural errors (2.7% vs 23.8% FA); slices reported separately, always.
- FedProc "0.5% FA" is blanket rejection, not discrimination (0.0% true-accept).

## License

MIT
