"""Self-preference re-check with matched baselines, Wilson CIs and Fisher tests.

Each judge is compared on answers from its own model vs the other model:
  qwen3b : own = Qwen2.5-3B natural GSM8K errors (data/items.jsonl), other = Qwen2.5-0.5B answers (selfpref set)
  qwen05b: own = Qwen2.5-0.5B answers (selfpref set),               other = Qwen2.5-3B natural GSM8K errors
Reported with the published labels and with the 51 mislabelled GSM8K items excluded
(see label_noise_sensitivity.py).

    python scripts/selfpref_recheck.py [--items data/items.jsonl] [--runs data/judge_runs] [--selfpref data/selfpref_items.jsonl]
"""
import argparse
import json
from math import comb, sqrt
from pathlib import Path

from label_noise_sensitivity import gold, robust_extract


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def fisher(a, n1, b, n2):
    k, tot = a + b, n1 + n2
    pm = lambda x: comb(n1, x) * comb(n2, k - x) / comb(tot, k)
    p0 = pm(a)
    return sum(pm(x) for x in range(max(0, k - n2), min(k, n1) + 1) if pm(x) <= p0 * (1 + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="data/items.jsonl")
    ap.add_argument("--runs", default="data/judge_runs")
    ap.add_argument("--selfpref", default="data/selfpref_items.jsonl")
    args = ap.parse_args()
    items = [json.loads(line) for line in open(args.items)]
    sp = [json.loads(line) for line in open(args.selfpref)]
    mislabelled = {i for i, it in enumerate(items)
                   if it["oracle"] == "gsm8k" and str(it["oracle_verdict"]) == "False"
                   and (lambda p, g: p is not None and g is not None and abs(p - g) < 1e-6)(robust_extract(it["answer"]), gold(it))}
    runs = Path(args.runs)
    for judge, own_is_3b in (("qwen3b", True), ("qwen05b", False)):
        main_rows = [json.loads(line) for line in open(runs / f"{judge}_pointwise.jsonl")]
        sp_rows = [json.loads(line) for line in open(runs / f"selfpref_{judge}.jsonl")]
        def natural(excl):
            return [r for i, (r, it) in enumerate(zip(main_rows, items))
                    if it["oracle"] == "gsm8k" and it["provenance"].startswith("natural")
                    and str(it["oracle_verdict"]) == "False" and not (excl and i in mislabelled)]
        small = [r for r, it in zip(sp_rows, sp) if str(it["oracle_verdict"]) == "False"]
        acc = lambda rows: (sum(1 for r in rows if r["judge_verdict"] == "CORRECT"), len(rows))
        for label, excl in (("published labels", False), ("51 mislabelled excluded", True)):
            big = natural(excl)
            own, other = (acc(big), acc(small)) if own_is_3b else (acc(small), acc(big))
            (k1, n1), (k2, n2) = own, other
            a, b = wilson(k1, n1), wilson(k2, n2)
            print(f"{judge:8} {label:24} own {k1}/{n1} = {100*k1/n1:.1f}% [{a[0]:.1f}, {a[1]:.1f}]   other {k2}/{n2} = {100*k2/n2:.1f}% [{b[0]:.1f}, {b[1]:.1f}]   Fisher p = {fisher(k1, n1, k2, n2):.3g}")


if __name__ == "__main__":
    main()
