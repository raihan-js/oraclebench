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
- **Two judges:** Qwen2.5-3B-Instruct (have it) + Llama-3.2-3B-Instruct (~6GB download). Pointwise AND pairwise formats. Position bias via A/B swap; verbosity bias via padded-but-equal answers; self-preference via Qwen-judging-Qwen.
- **Harness wins by construction** wherever a checker exists — the reported results are false-accept rates and the cost (GPU-seconds, tokens) of calling a judge where a check sufficed.

### Hardware

One RTX 3060 12GB. ~15-25 GPU-hours. Judges run sequentially (one model at a time). Optional ~$10 A40 for a 32B-AWQ judge row.

### Stack

Python, PyTorch, HF Transformers. No vLLM (judge calls are low-throughput; HF generate suffices). scipy (bootstrap CIs), pytest.

### Milestones (~17 days)

1. **Oracle item bank** (5d) — ~2,000 (prompt, answer, oracle verdict, provenance) across 3 oracles; corrupted + natural slices separate; HF dataset with provenance column.
2. **Judge sweep** (5d) — pointwise + pairwise per judge; accuracy, false-accept, position/verbosity/self-preference bias; 95% CIs; frozen prompts.
3. **Checker-first harness** (4d) — `oraclebench` library (checkers first, judge only on uncovered); per-decision traces; error + GPU-seconds comparison.
4. **Write-up** (3d) — repo, HF dataset, dev.to post with Limitations (small judges, synthetic corruptions, slices separate).

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
