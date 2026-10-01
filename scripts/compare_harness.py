#!/usr/bin/env python3
"""Harness vs judge-only comparison on bank + uncovered items.

Measures: error rate, judge calls, GPU-seconds, tokens.
The harness routes covered items to oracles (ground truth) and only
uncovered items to the judge. Judge-only calls the judge for everything.
Usage: PYTHONPATH=src python scripts/compare_harness.py
"""
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from oraclebench.harness import CheckerFirstHarness
from oraclebench.judges.prompts import render_pointwise
from oraclebench.oracles.fedproc import FedProcRegistryScorer
from oraclebench.oracles.gsm8k import GSM8KScorer
from oraclebench.oracles.ifeval import IFEvalScorer

MODEL = "../flipgate/data/models/Qwen--Qwen2.5-3B-Instruct"


def main() -> None:
    with open("../flipgate/data/eval/far_registry.json") as f:
        import re
        registry = {re.sub(r"^(?:FAR|DFARS)\s+", "", c, flags=re.IGNORECASE)
                    for c in json.load(f)}
    scorers = {"gsm8k": GSM8KScorer(), "ifeval": IFEvalScorer(),
               "fedproc": FedProcRegistryScorer(registry=registry)}

    items = [json.loads(l) for l in open("data/items.jsonl")]
    uncovered = [json.loads(l) for l in open("data/items_uncovered.jsonl")]
    # uncovered items are summary pairs; grade answer_a as the candidate
    for it in uncovered:
        it["answer"] = it.pop("answer_a")
        it.pop("answer_b", None)
    # uncovered items have no oracle verdict; harness routes them to judge
    for it in uncovered:
        it["oracle_verdict"] = None
    all_items = items + uncovered
    print(f"Bank: {len(items)}, uncovered: {len(uncovered)}, total: {len(all_items)}",
          flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda",
        pad_token_id=tok.pad_token_id)
    model.eval()

    def ref_summary(ref: dict) -> str:
        if ref.get("answer"):
            return f"Expected answer: {ref['answer']}"
        if "instruction_id_list" in ref:
            return "Must satisfy: " + "; ".join(ref["instruction_id_list"])
        return "Must cite a real FAR/DFARS clause number."

    def judge_fn(prompt, answer, ref):
        import re as _re
        text = ("You are a strict grader. Reply with exactly one word: CORRECT or WRONG.\n"
                f"\nQuestion:\n{prompt}\n\nCandidate answer:\n{answer}\n"
                f"\nReference:\n{ref_summary(ref)}\n\nVerdict:")
        messages = [{"role": "user", "content": text}]
        formatted = tok.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=True)
        inputs = tok(formatted, return_tensors="pt", truncation=True,
                     max_length=4096 - 32).to(model.device)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        dt = time.time() - t0
        resp = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        m = _re.search(r"\b(CORRECT|WRONG)\b", resp, _re.IGNORECASE)
        return (m.group(1).upper() == "CORRECT" if m else None,
                dt, int(out.shape[1] - inputs.input_ids.shape[1]))

    # --- harness run ---
    h = CheckerFirstHarness(scorers, judge_fn=judge_fn)
    traces = [h.decide(it) for it in all_items]
    with open("data/harness_traces.jsonl", "w") as f:
        for t in traces:
            f.write(json.dumps(t) + "\n")

    # --- scoring: only items WITH oracle verdict count toward error ---
    scored = [t for t, it in zip(traces, all_items) if it["oracle_verdict"] is not None]
    h_err = sum(1 for t, it in zip(traces, all_items)
                if it["oracle_verdict"] is not None
                and t["verdict"] is not None and bool(t["verdict"]) != it["oracle_verdict"])
    h_abs = sum(1 for t in traces if t["verdict"] is None)
    j_calls = sum(1 for t in traces if t["verdict_source"] == "judge")
    j_secs = sum(t.get("judge_seconds", 0) for t in traces)
    j_toks = sum(t.get("judge_tokens", 0) for t in traces)

    # --- judge-only baseline: what if the judge graded every scored item? ---
    # (measured directly: run judge over scored items)
    print("Judge-only baseline on scored items...", flush=True)
    j_err, j_abs, b_secs, b_toks = 0, 0, 0.0, 0
    for it in all_items:
        if it["oracle_verdict"] is None:
            continue
        v, s, tk = judge_fn(it["prompt"], it["answer"], it.get("reference", {}))
        b_secs += s
        b_toks += tk
        if v is None:
            j_abs += 1
        elif bool(v) != it["oracle_verdict"]:
            j_err += 1
    n_scored = sum(1 for it in all_items if it["oracle_verdict"] is not None)

    print(f"\n=== Harness vs judge-only (n={n_scored} scored + {len(uncovered)} uncovered) ===")
    print(f"Harness:      errors {h_err}/{n_scored} = {100*h_err/n_scored:.1f}%, "
          f"judge calls {j_calls}, judge GPU-s {j_secs:.0f}, judge tokens {j_toks}")
    print(f"Judge-only:   errors {j_err}/{n_scored} = {100*j_err/n_scored:.1f}%, "
          f"judge calls {n_scored}, judge GPU-s {b_secs:.0f}, judge tokens {b_toks}")
    print(f"Coverage: {h.summary()}")
    with open("data/harness_comparison.json", "w") as f:
        json.dump({"harness_errors": h_err, "judge_errors": j_err, "n_scored": n_scored,
                   "harness_judge_calls": j_calls, "judge_only_calls": n_scored,
                   "harness_judge_secs": round(j_secs, 1), "judge_only_secs": round(b_secs, 1),
                   "harness_judge_toks": j_toks, "judge_only_toks": b_toks,
                   "coverage": h.summary()}, f, indent=2)
    print("Saved traces + comparison.", flush=True)


if __name__ == "__main__":
    main()
