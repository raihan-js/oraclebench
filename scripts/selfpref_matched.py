#!/usr/bin/env python3
"""Question-matched self-preference control.

The first self-preference comparison judged Qwen-3B's wrong answers (on the questions Qwen-3B gets wrong) against
Qwen-0.5B's wrong answers (on a different random set of questions), so question difficulty and error type were
confounded with "own vs other". Here both models answer the SAME questions: the natural GSM8K errors of Qwen-3B.
We keep the questions where Qwen-0.5B is also wrong and compare, per judge, how often it accepts each model's
wrong answer to the same question (exact McNemar on the discordant pairs).

  python scripts/selfpref_matched.py generate     # needs the GPU; writes data/selfpref_matched_items.jsonl
  python scripts/run_judges.py --judge qwen3b  --items data/selfpref_matched_items.jsonl --out-dir data/judge_runs_selfpref_matched
  python scripts/run_judges.py --judge qwen05b --items data/selfpref_matched_items.jsonl --out-dir data/judge_runs_selfpref_matched
  python scripts/selfpref_matched.py analyze      # writes results/selfpref_matched.json
"""
import json
import sys
from math import comb, sqrt
from pathlib import Path

GEN_05B = "data/models/Qwen--Qwen2.5-0.5B-Instruct"
ITEMS = "data/items.jsonl"
OUT_ITEMS = "data/selfpref_matched_items.jsonl"
JUDGE_DIR = Path("data/judge_runs_selfpref_matched")


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(100 * max(0, c - h), 1), round(100 * min(1, c + h), 1)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def generate(max_new_tokens=1024):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from oraclebench.oracles.gsm8k import GSM8KScorer

    items = [json.loads(line) for line in open(ITEMS)]
    big = [it for it in items if it["oracle"] == "gsm8k" and it["provenance"].startswith("natural")
           and str(it["oracle_verdict"]) == "False"]
    print(f"{len(big)} Qwen-3B natural GSM8K errors", flush=True)
    tok = AutoTokenizer.from_pretrained(GEN_05B, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(GEN_05B, dtype=torch.bfloat16, device_map="cuda",
                                                 pad_token_id=tok.pad_token_id)
    model.eval()
    scorer = GSM8KScorer()
    kept = skipped_trunc = right = 0
    with open(OUT_ITEMS, "w") as f:
        for n, it in enumerate(big):
            q = it["prompt"]
            formatted = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
            inputs = tok(formatted, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            gen = out[0][inputs.input_ids.shape[1]:].tolist()
            if len(gen) >= max_new_tokens and tok.eos_token_id not in gen:
                skipped_trunc += 1
                continue
            resp = tok.decode(gen, skip_special_tokens=True)
            ref = it["reference"]
            ok = scorer.score(q, resp, ref if isinstance(ref, dict) else {"answer": ref}) == 1.0
            if ok:
                right += 1
                continue  # 0.5B got it right: not a matched error
            kept += 1
            f.write(json.dumps({"oracle": "gsm8k", "prompt": q, "answer": resp, "reference": ref,
                                "oracle_verdict": False, "provenance": "selfpref_matched:Qwen2.5-0.5B",
                                "answer_3b": it["answer"], "provenance_3b": it["provenance"]}) + "\n")
            if (n + 1) % 25 == 0:
                print(f"  {n+1}/{len(big)} kept {kept}", flush=True)
    print(f"matched questions (both wrong): {kept}; 0.5B right: {right}; 0.5B truncated at cap: {skipped_trunc}", flush=True)


def analyze():
    items = [json.loads(line) for line in open(OUT_ITEMS)]
    main_items = [json.loads(line) for line in open(ITEMS)]
    by_prompt = {it["prompt"]: i for i, it in enumerate(main_items)
                 if it["oracle"] == "gsm8k" and it["provenance"].startswith("natural")}
    out = {"n_questions": len(items), "judges": {}}
    for judge in ("qwen3b", "qwen05b"):
        main_rows = [json.loads(line) for line in open(f"data/judge_runs/{judge}_pointwise.jsonl")]
        small_rows = [json.loads(line) for line in open(JUDGE_DIR / f"{judge}_pointwise.jsonl")]
        assert len(small_rows) == len(items), (len(small_rows), len(items))  # run_judges writes rows in item order
        acc3, acc05 = [], []
        for it, r05 in zip(items, small_rows):
            acc3.append(main_rows[by_prompt[it["prompt"]]]["judge_verdict"] == "CORRECT")
            acc05.append(r05["judge_verdict"] == "CORRECT")
        n = len(acc3)
        k3, k05 = sum(acc3), sum(acc05)
        b = sum(1 for x, y in zip(acc3, acc05) if x and not y)   # accepts 3B's answer only
        c = sum(1 for x, y in zip(acc3, acc05) if y and not x)   # accepts 0.5B's answer only
        out["judges"][judge] = {
            "n": n, "accepts_3b_wrong": [k3, wilson(k3, n)], "accepts_05b_wrong": [k05, wilson(k05, n)],
            "only_3b_accepted": b, "only_05b_accepted": c, "mcnemar_p": mcnemar_exact(b, c)}
        print(f"{judge:8} n={n} accepts Qwen-3B's wrong answer {k3} ({100*k3/n:.1f}% {wilson(k3, n)}) vs Qwen-0.5B's wrong answer "
              f"{k05} ({100*k05/n:.1f}% {wilson(k05, n)}); discordant 3B-only {b}, 0.5B-only {c}; exact McNemar p = {mcnemar_exact(b, c):.3g}")
    Path("results").mkdir(exist_ok=True)
    json.dump(out, open("results/selfpref_matched.json", "w"), indent=2)


if __name__ == "__main__":
    {"generate": generate, "analyze": analyze}[sys.argv[1]]()
