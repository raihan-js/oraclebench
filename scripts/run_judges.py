#!/usr/bin/env python3
"""Judge sweep runner (resumable).

Pointwise: each judge grades every item (CORRECT/WRONG). Unparseable
judge output counts as ABSTAIN — never as accept.
Pairwise/bias probes run on fixed subsamples (see --mode).
Usage:
  PYTHONPATH=src python scripts/run_judges.py --judge qwen3b --mode pointwise
  PYTHONPATH=src python scripts/run_judges.py --judge qwen05b --mode pointwise
"""
import argparse
import json
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from oraclebench.judges.prompts import render_pointwise

JUDGES = {
    "qwen3b": "../flipgate/data/models/Qwen--Qwen2.5-3B-Instruct",
    "qwen05b": "data/models/Qwen--Qwen2.5-0.5B-Instruct",
}


def parse_pointwise(response: str) -> str | None:
    """Extract CORRECT/WRONG verdict; None if unparseable (abstain)."""
    m = re.search(r"\b(CORRECT|WRONG)\b", response, re.IGNORECASE)
    return m.group(1).upper() if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True, choices=list(JUDGES))
    ap.add_argument("--mode", default="pointwise", choices=["pointwise"])
    ap.add_argument("--items", default="data/items.jsonl")
    ap.add_argument("--out-dir", default="data/judge_runs")
    ap.add_argument("--max-tokens", type=int, default=64)
    args = ap.parse_args()

    import os
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = f"{args.out_dir}/{args.judge}_{args.mode}.jsonl"
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    done.add((d["prompt"], d["answer"]))
                except Exception:
                    pass
    print(f"Resuming {out_path}: {len(done)} done", flush=True)

    with open(args.items) as f:
        items = [json.loads(line) for line in f]
    todo = [it for it in items if (it["prompt"], it["answer"]) not in done]
    print(f"Remaining: {len(todo)}/{len(items)}", flush=True)
    if not todo:
        print("Nothing to do.", flush=True)
        return

    print(f"Loading judge {args.judge}...", flush=True)
    model_path = JUDGES[args.judge]
    tok = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="cuda",
        pad_token_id=tok.pad_token_id)
    model.eval()
    print("Judge loaded", flush=True)

    start = time.time()
    with open(out_path, "a") as f:
        for n, it in enumerate(todo, 1):
            prompt = render_pointwise(it["prompt"], it["answer"], it["reference"])
            messages = [{"role": "user", "content": prompt}]
            formatted = tok.apply_chat_template(messages, tokenize=False,
                                                add_generation_prompt=True)
            inputs = tok(formatted, return_tensors="pt", truncation=True,
                         max_length=4096 - args.max_tokens).to(model.device)
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=args.max_tokens,
                                     do_sample=False)
            dt = time.time() - t0
            resp = tok.decode(out[0][inputs.input_ids.shape[1]:],
                              skip_special_tokens=True)
            verdict = parse_pointwise(resp)
            f.write(json.dumps({
                "oracle": it["oracle"],
                "provenance": it["provenance"],
                "oracle_verdict": it["oracle_verdict"],
                "judge": args.judge,
                "judge_verdict": verdict,  # CORRECT, WRONG, or None (abstain)
                "judge_response": resp[:300],
                "seconds": round(dt, 2),
            }) + "\n")
            if n % 50 == 0:
                print(f"  Progress: {len(done)+n}/{len(items)}", flush=True)

    print(f"Done in {time.time()-start:.0f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
