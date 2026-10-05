#!/usr/bin/env python3
"""Pairwise bias probes (resumable): position swap + verbosity + self-preference.

For each sampled item (wrong answer W + correct answer C on same prompt type):
  - order AB (W first) and BA (C first) -> position bias
  - verbosity: padded correct vs plain correct -> verbosity bias
Self-preference: Qwen-3B judging Qwen-3B vs Qwen-0.5B outputs (from provenance).
Usage: PYTHONPATH=src python scripts/run_pairwise.py --judge qwen3b --n 200
"""
import argparse
import json
import random
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from oraclebench.judges.prompts import render_pairwise

JUDGES = {
    "qwen3b": "../flipgate/data/models/Qwen--Qwen2.5-3B-Instruct",
    "qwen05b": "data/models/Qwen--Qwen2.5-0.5B-Instruct",
}

FILLER = (" Note that this response was generated carefully with attention to "
          "detail, following all instructions precisely and completely.")


def parse_pairwise(response: str) -> str | None:
    m = re.search(r"\b([AB])\b", response, re.IGNORECASE)
    return m.group(1).upper() if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True, choices=list(JUDGES))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out-dir", default="data/judge_runs")
    ap.add_argument("--matched", action="store_true",
                    help="pair each wrong answer with a correct answer to the SAME prompt (the default pairs a wrong "
                         "answer with a correct answer from a different item, so it only measures position/verbosity bias)")
    args = ap.parse_args()

    import os
    rng = random.Random(args.seed)
    # pair wrong + correct items with same oracle
    wrong, correct = [], []
    with open("data/items.jsonl") as f:
        for line in f:
            it = json.loads(line)
            (wrong if not it["oracle_verdict"] else correct).append(it)
    by_oracle_w = {}
    by_oracle_c = {}
    for it in wrong:
        by_oracle_w.setdefault(it["oracle"], []).append(it)
    for it in correct:
        by_oracle_c.setdefault(it["oracle"], []).append(it)

    pairs = []
    per = args.n // 3
    if args.matched:
        by_prompt = {}
        for it in correct:
            by_prompt.setdefault((it["oracle"], it["prompt"]), it)
        for oracle in ["gsm8k", "ifeval", "fedproc"]:
            ws = [w for w in by_oracle_w.get(oracle, []) if (oracle, w["prompt"]) in by_prompt]
            rng.shuffle(ws)
            for w in ws[:per]:
                c = by_prompt[(oracle, w["prompt"])]
                pairs.append({"oracle": oracle, "prompt": w["prompt"], "wrong": w["answer"],
                              "right": c["answer"], "reference": w["reference"],
                              "wrong_provenance": w["provenance"]})
    else:
      for oracle in ["gsm8k", "ifeval", "fedproc"]:
        ws = by_oracle_w.get(oracle, [])[:]
        cs = by_oracle_c.get(oracle, [])[:]
        rng.shuffle(ws)
        rng.shuffle(cs)
        for w, c in zip(ws[:per], cs[:per]):
            # use wrong item's prompt with correct answer from same oracle pool
            pairs.append({"oracle": oracle, "prompt": w["prompt"],
                          "wrong": w["answer"], "right": c["answer"],
                          "reference": w["reference"]})

    out_path = f"{args.out_dir}/{args.judge}_pairwise{'_matched' if args.matched else ''}.jsonl"
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    done.add((d["prompt"], d["order"]))
                except Exception:
                    pass
    print(f"Pairs: {len(pairs)}, done: {len(done)}", flush=True)

    tok = AutoTokenizer.from_pretrained(JUDGES[args.judge], padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        JUDGES[args.judge], dtype=torch.bfloat16, device_map="cuda",
        pad_token_id=tok.pad_token_id)
    model.eval()

    def ask(prompt_text):
        messages = [{"role": "user", "content": prompt_text}]
        formatted = tok.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=True)
        inputs = tok(formatted, return_tensors="pt", truncation=True,
                     max_length=4096 - 32).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        resp = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return parse_pairwise(resp), resp[:100]

    with open(out_path, "a") as f:
        for it in pairs:
            for order, (a_txt, b_txt, a_is_wrong) in [
                    ("AB", (it["wrong"], it["right"], True)),
                    ("BA", (it["right"], it["wrong"], False))]:
                if (it["prompt"], order) in done:
                    continue
                # verbosity probe: pad the RIGHT answer on BA order
                verdict, raw = ask(render_pairwise(it["prompt"], a_txt, b_txt, it["reference"]))
                f.write(json.dumps({"oracle": it["oracle"], "order": order,
                                    "picked_wrong": (verdict == "A") == a_is_wrong
                                    if verdict else None,
                                    "raw": raw}) + "\n")
                # verbosity: same pair, right answer padded
                padded = it["right"] + FILLER
                vpa, vpb = (it["wrong"], padded) if order == "AB" else (padded, it["wrong"])
                v_verdict, v_raw = ask(render_pairwise(it["prompt"], vpa, vpb, it["reference"]))
                f.write(json.dumps({"oracle": it["oracle"], "order": order + "+verbose",
                                    "picked_wrong": (v_verdict == "A") == (order == "AB")
                                    if v_verdict else None,
                                    "raw": v_raw}) + "\n")
    print(f"Saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
