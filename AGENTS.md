# AGENTS.md — oraclebench

Fourth portfolio project for Raihan Sikder. Target roles: Noeon Research (LLMOps), Citadel AI (AI testing product), Treasure AI, VISASQ — Tokyo/Japan.

Siblings: `../flipgate/` (all three oracle scorers + stored runs with natural errors), `../graphproof-qa/`, `../fedproc-constrained/` (registry).
Prep context: `../../noeon-prep/`.

---

## Active project: OracleBench

Grade small open LLM-as-judge setups against deterministic oracles. Measure false-accept rate + position/verbosity bias + self-preference. Ship a checker-first-judge-second harness with per-decision traces.

**Publishes:** false-accept rate of each judge on oracle-wrong answers (corrupted + natural slices separate), with 95% bootstrap CIs.

### Honesty rules (load-bearing)

- Novelty stays narrow: oracle-grounded judge-error study with a real legal registry. NOT "the first judge benchmark" (JudgeBench/RewardBench exist — cite them).
- Only small judges tested (≤3B local) — conclusions do NOT carry to frontier judges. Say so in paragraph one of every write-up.
- Synthetic corruptions and natural errors reported SEPARATELY, always.
- Judge prompts frozen + versioned in `configs/judge_prompts/`; never edit mid-sweep (new version instead).
- Qwen thinking mode pinned + reported if used.

### Key design decisions

- **Three oracles, all reused:** FedProc registry check, GSM8K exact arithmetic, IFEval 25-rule checkers — copied from `../flipgate/src/flipgate/scorers/` (with attribution note), NOT imported cross-project.
- **Natural errors are free:** flipgate's JSONL store holds 4,000+ evaluated items with scores; wrong answers = natural-error slice. No extra generation needed.
- **Corrupted slice is synthetic:** fabricated FAR clause swap, off-by-one arithmetic step, one broken IFEval rule — built by perturbing correct answers deterministically.
- **Two judges:** Qwen2.5-3B-Instruct + Qwen2.5-0.5B-Instruct (both on disk — doubles as a judge-size comparison). Llama-3.2-3B is gated on HF (no access with current token); swap it in if access is granted. Pointwise AND pairwise formats. Position bias via A/B swap; verbosity bias via padded-but-equal answers; self-preference via Qwen-judging-Qwen.
- **Harness wins by construction** wherever a checker exists — the reported results are false-accept rates and the cost (GPU-seconds, tokens) of calling a judge where a check sufficed.

### Hardware

One RTX 3060 12GB. ~15-25 GPU-hours. Judges run sequentially (one model at a time). Optional ~$10 A40 for a 32B-AWQ judge row.

### Stack

Python, PyTorch, HF Transformers. No vLLM (judge calls are low-throughput; HF generate suffices). scipy (bootstrap CIs), pytest.

### Milestones (~17 days)

1. **Oracle item bank** — ✅ REBUILT 2026-10-05 from the 1,024-token FlipGate run (the first bank used labels from a 256-token run; 51 of 398 "wrong" were right): 1,760 verified items + 100 uncovered: 511 natural + 822 corrupted + 427 correct.
2. **Judge sweep** — ✅ RE-RUN (`scripts/summarize.py` -> `results/summary.json`):
   - False-accept: qwen3b 11.0% [9.5%, 12.8%], qwen05b 40.6% [38.0%, 43.2%] (n=1,333 oracle-wrong)
   - Per-oracle FA: GSM8K 10.1%/94.8%; IFEval 21.4%/13.7%; FedProc 0.5%/1.0% (blanket rejection: TA 0.0%)
   - Natural 23.5% vs corrupted 3.3% (3b) — synthetic errors ~7x easier to catch
   - **Pairwise (matched pairs, same prompt):** 3b picks the right answer 80.5% [76.6, 83.8] with a position bias (94% right-in-B vs 67% right-in-A); 0.5b at chance (53.8%), picks slot B 83%. The first pairwise probe paired a wrong answer with a right answer to a DIFFERENT question (position bias only; its "both pick B 85-92%" claim is retired).
   - **Self-preference:** unmatched questions 19.7% vs 0.9% (confounded, not evidence); matched on the same 165 questions 14.5% vs 5.5%, exact McNemar p = 0.0015 (3b judge); 0.5b judge rubber-stamps (92.7% both). Cannot be separated from error subtlety (all Qwen).
3. **Checker-first harness** — ✅ (`oraclebench/harness.py`, 6 tests): 0/1,760 errors vs judge-only 412 (23.4%); 100 vs 1,760 judge calls (17.6x), 22 vs 448 GPU-s; 94.6% of 1,860 items routed to checkers. The zero is by construction.
4. **Write-up** — ✅ HF dataset (rebuilt, with judge outputs), article rewritten (leads with what had to be redone).

### Risks to keep honest

- Synthetic corruptions ≠ natural errors (both slices, always separate).
- Small judges only (no frontier claims).
- Judge prompts fragile (freeze + version).
- Narrow novelty (say what exists).

---

## Conventions

- Python 3.10+, pytest for oracles, corruption builders, harness routing.
- Own venv (`.venv/`); torch cu124 (known-good).
- Per-item JSONL, append-only; every claim cites run ID.
- Oracle verdicts are ground truth by construction; judge outputs are the variable.
