#!/usr/bin/env python3
"""Self-preference probe: do Qwen judges favor Qwen outputs?

Generates answers with Qwen-0.5B on GSM8K questions (oracle-scored),
then has BOTH judges grade BOTH generators' outputs:
  Qwen3B judge x {Qwen3B outputs, Qwen05B outputs}
  Qwen05B judge x {Qwen3B outputs, Qwen05B outputs}
Self-preference = higher accept rate on own-family outputs, all else equal.
Usage: PYTHONPATH=src python scripts/run_selfpref.py [--n 200]
"""
import argparse
import json
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from oraclebench.oracles.gsm8k import GSM8KScorer

GEN_05B = "data/models/Qwen--Qwen2.5-0.5B-Instruct"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=21)
    args = ap.parse_args()

    import random
    rng = random.Random(args.seed)
    gsm8k = load_dataset("openai/gsm8k", "main", split="test")
    idxs = rng.sample(range(len(gsm8k)), args.n)

    tok = AutoTokenizer.from_pretrained(GEN_05B, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        GEN_05B, dtype=torch.bfloat16, device_map="cuda",
        pad_token_id=tok.pad_token_id)
    model.eval()

    scorer = GSM8KScorer()
    out_path = "data/selfpref_items.jsonl"
    done = set()
    try:
        with open(out_path) as f:
            for line in f:
                done.add(json.loads(line)["id"])
    except FileNotFoundError:
        pass

    with open(out_path, "a") as f:
        for n, i in enumerate(idxs):
            qid = f"gsm8k-test-{i}"
            if qid in done:
                continue
            item = gsm8k[i]
            messages = [{"role": "user", "content": item["question"]}]
            formatted = tok.apply_chat_template(messages, tokenize=False,
                                                add_generation_prompt=True)
            inputs = tok(formatted, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            resp = tok.decode(out[0][inputs.input_ids.shape[1]:],
                              skip_special_tokens=True)
            score = scorer.score(item["question"], resp, {"answer": item["answer"]})
            f.write(json.dumps({"id": qid, "oracle": "gsm8k",
                                "prompt": item["question"], "answer": resp,
                                "reference": {"answer": item["answer"]},
                                "oracle_verdict": score == 1.0,
                                "provenance": "selfpref:Qwen2.5-0.5B",
                                "generator": "qwen05b"}) + "\n")
            if (n + 1) % 25 == 0:
                print(f"  {n+1}/{len(idxs)}", flush=True)
    print(f"Saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
