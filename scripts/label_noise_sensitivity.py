"""Label-noise sensitivity for the GSM8K 'oracle-wrong' items.

The natural GSM8K errors come from FlipGate's bf16 run, which used max_new_tokens=256 and a
strict answer extractor, so some items labelled 'oracle-wrong' are actually correct (the
answer is present but the extractor missed it, e.g. boxed or thousands separators).
This script re-scores those items with a more robust extractor and recomputes each judge's
false-accept rate with the mislabelled items excluded.

    python scripts/label_noise_sensitivity.py [--items data/items.jsonl] [--runs data/judge_runs]

Items: https://huggingface.co/datasets/raihan-js/oraclebench-items
"""
import argparse
import json
import re
from pathlib import Path

NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def to_num(s):
    try:
        return float(s.replace(",", "").replace("$", "").strip().rstrip("."))
    except ValueError:
        return None


def robust_extract(resp):
    boxed = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", resp)
    if boxed and (m := NUM.findall(boxed[-1].replace("\\,", ""))):
        return to_num(m[-1])
    if m := re.search(r"####\s*(-?\d[\d,]*\.?\d*)", resp):
        return to_num(m.group(1))
    if m := re.search(r"answer is[^\d\-]{0,20}(-?\d[\d,]*\.?\d*)", resp, re.I):
        return to_num(m.group(1))
    nums = NUM.findall(resp)
    return to_num(nums[-1]) if nums else None


def gold(item):
    ref = item["reference"]
    text = ref.get("answer") if isinstance(ref, dict) else ref
    m = re.search(r"####\s*(-?\d[\d,]*\.?\d*)", str(text))
    return to_num(m.group(1)) if m else to_num(str(text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="data/items.jsonl")
    ap.add_argument("--runs", default="data/judge_runs")
    args = ap.parse_args()
    items = [json.loads(line) for line in open(args.items)]
    mislabelled = set()
    for idx, it in enumerate(items):
        if it["oracle"] == "gsm8k" and str(it["oracle_verdict"]) == "False":
            p, g = robust_extract(it["answer"]), gold(it)
            if p is not None and g is not None and abs(p - g) < 1e-6:
                mislabelled.add(idx)
    n_gsm_wrong = sum(1 for it in items if it["oracle"] == "gsm8k" and str(it["oracle_verdict"]) == "False")
    n_nat = sum(1 for it in items if it["oracle"] == "gsm8k" and it["provenance"].startswith("natural") and str(it["oracle_verdict"]) == "False")
    print(f"GSM8K oracle-wrong items: {n_gsm_wrong} ({n_nat} natural); robust extractor finds the right answer in {len(mislabelled)} of them\n")
    print(f"{'judge':8} {'slice':8} {'false-accept (as published)':>28} {'excluding mislabelled':>24}")
    for judge in ("qwen3b", "qwen05b"):
        rows = [json.loads(line) for line in open(Path(args.runs) / f"{judge}_pointwise.jsonl")]
        for name, pred in (("GSM8K", lambda it: it["oracle"] == "gsm8k"), ("overall", lambda it: True)):
            sel = [(i, r) for i, (it, r) in enumerate(zip(items, rows)) if pred(it) and str(it["oracle_verdict"]) == "False"]
            clean = [(i, r) for i, r in sel if i not in mislabelled]
            f = lambda L: 100 * sum(1 for _, r in L if r["judge_verdict"] == "CORRECT") / len(L)
            print(f"{judge:8} {name:8} {f(sel):>20.1f}% (n={len(sel):4d}) {f(clean):>14.1f}% (n={len(clean):4d})")


if __name__ == "__main__":
    main()
