#!/usr/bin/env python3
"""Recompute every number in the README from the judge-run files (Wilson 95% CIs).

Writes results/summary.json and prints the tables.

  python scripts/summarize.py [--items data/items.jsonl] [--runs data/judge_runs]
"""
import argparse
import json
from collections import defaultdict
from math import sqrt
from pathlib import Path

JUDGES = ["qwen3b", "qwen05b"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0.0, c - h), 100 * min(1.0, c + h))


def rate(k, n):
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "pct": round(100 * k / n, 1) if n else None, "ci95": [round(lo, 1), round(hi, 1)]}


def fmt(r):
    return f"{r['pct']}% [{r['ci95'][0]}, {r['ci95'][1]}] ({r['k']}/{r['n']})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="data/items.jsonl")
    ap.add_argument("--runs", default="data/judge_runs")
    ap.add_argument("--out", default="results/summary.json")
    args = ap.parse_args()
    items = [json.loads(line) for line in open(args.items)]
    out = {"n_items": len(items), "pointwise": {}, "pairwise": {}}
    wrong = lambda it: str(it["oracle_verdict"]) == "False"
    slices = ["gsm8k", "ifeval", "fedproc"]

    for j in JUDGES:
        rows = [json.loads(line) for line in open(Path(args.runs) / f"{j}_pointwise.jsonl")]
        assert len(rows) == len(items), (j, len(rows), len(items))
        acc = lambda r: r["judge_verdict"] == "CORRECT"
        d = {}
        w = [(r, it) for r, it in zip(rows, items) if wrong(it)]
        c = [(r, it) for r, it in zip(rows, items) if not wrong(it)]
        d["false_accept_overall"] = rate(sum(acc(r) for r, _ in w), len(w))
        d["true_accept_overall"] = rate(sum(acc(r) for r, _ in c), len(c))
        for s in slices:
            ws = [(r, it) for r, it in w if it["oracle"] == s]
            cs = [(r, it) for r, it in c if it["oracle"] == s]
            d[f"false_accept_{s}"] = rate(sum(acc(r) for r, _ in ws), len(ws))
            d[f"true_accept_{s}"] = rate(sum(acc(r) for r, _ in cs), len(cs))
        for prov in ("natural", "corrupted"):
            wp = [(r, it) for r, it in w if it["provenance"].startswith(prov)]
            d[f"false_accept_{prov}"] = rate(sum(acc(r) for r, _ in wp), len(wp))
        unparsed = sum(1 for r in rows if r["judge_verdict"] not in ("CORRECT", "WRONG"))
        d["unparsed_verdicts"] = unparsed
        out["pointwise"][j] = d

        prow = [json.loads(line) for line in open(Path(args.runs) / f"{j}_pairwise.jsonl")]
        pd = {"n": len(prow), "unparsed": sum(r["picked_wrong"] is None for r in prow)}
        def b_pick(r):  # picked the answer in position B
            return (r["order"].startswith("AB") and r["picked_wrong"] is False) or (r["order"].startswith("BA") and r["picked_wrong"] is True)
        valid = [r for r in prow if r["picked_wrong"] is not None]
        pd["picks_B_all"] = rate(sum(b_pick(r) for r in valid), len(valid))
        for cond, sel in (("plain", lambda r: "+verbose" not in r["order"]), ("verbose_padded_correct", lambda r: "+verbose" in r["order"])):
            v = [r for r in valid if sel(r)]
            pd[f"picks_B_{cond}"] = rate(sum(b_pick(r) for r in v), len(v))
            pd[f"picks_wrong_{cond}"] = rate(sum(r["picked_wrong"] for r in v), len(v))
        for s in slices:
            v = [r for r in valid if r["oracle"] == s and "+verbose" not in r["order"]]
            pd[f"picks_B_{s}"] = rate(sum(b_pick(r) for r in v), len(v))
        out["pairwise"][j] = pd

    # Matched pairs: the right answer answers the same prompt as the wrong one (run_pairwise.py --matched)
    out["pairwise_matched"] = {}
    for j in JUDGES:
        path = Path(args.runs) / f"{j}_pairwise_matched.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(line) for line in open(path)]
        valid = [r for r in rows if r["picked_wrong"] is not None]
        md = {"n_rows": len(rows), "unparsed": len(rows) - len(valid)}
        for cond, sel in (("plain", lambda r: "+verbose" not in r["order"]),
                          ("verbose", lambda r: "+verbose" in r["order"])):
            v = [r for r in valid if sel(r)]
            md[f"picks_right_{cond}"] = rate(sum(not r["picked_wrong"] for r in v), len(v))
            md[f"picks_B_{cond}"] = rate(sum(((r["order"].startswith("AB") and r["picked_wrong"] is False)
                                              or (r["order"].startswith("BA") and r["picked_wrong"] is True)) for r in v), len(v))
            # accuracy when the right answer is in slot B (AB) vs slot A (BA)
            ab = [r for r in v if r["order"].startswith("AB")]
            ba = [r for r in v if r["order"].startswith("BA")]
            md[f"picks_right_when_right_is_B_{cond}"] = rate(sum(not r["picked_wrong"] for r in ab), len(ab))
            md[f"picks_right_when_right_is_A_{cond}"] = rate(sum(not r["picked_wrong"] for r in ba), len(ba))
        for sl in slices:
            v = [r for r in valid if r["oracle"] == sl and "+verbose" not in r["order"]]
            md[f"picks_right_plain_{sl}"] = rate(sum(not r["picked_wrong"] for r in v), len(v))
        out["pairwise_matched"][j] = md

    Path(args.out).parent.mkdir(exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)

    print(f"items: {len(items)}")
    print("\nPointwise false-accept (judge says CORRECT on oracle-wrong items), Wilson 95%:")
    for j in JUDGES:
        d = out["pointwise"][j]
        print(f"  {j:8} overall {fmt(d['false_accept_overall'])}")
        for s in slices:
            print(f"           {s:8} FA {fmt(d['false_accept_'+s])}   true-accept {fmt(d['true_accept_'+s])}")
        print(f"           natural FA {fmt(d['false_accept_natural'])}   corrupted FA {fmt(d['false_accept_corrupted'])}   unparsed {d['unparsed_verdicts']}")
    print("\nPairwise (n per judge = 792; each pair judged in both orders, plain and with the correct answer padded):")
    for j in JUDGES:
        d = out["pairwise"][j]
        print(f"  {j:8} picks B overall {fmt(d['picks_B_all'])}; plain {fmt(d['picks_B_plain'])}; verbose {fmt(d['picks_B_verbose_padded_correct'])}; unparsed {d['unparsed']}")
        print(f"           picks the WRONG answer: plain {fmt(d['picks_wrong_plain'])}; verbose {fmt(d['picks_wrong_verbose_padded_correct'])}")
        print("           picks B by slice (plain): " + "; ".join(f"{s} {d['picks_B_'+s]['pct']}%" for s in slices))
    for j, d in out["pairwise_matched"].items():
        print(f"\nMatched pairwise ({j}): wrong and right answers to the SAME prompt; {d['n_rows']} judgements, {d['unparsed']} unparsed")
        print(f"  picks the right answer: plain {fmt(d['picks_right_plain'])}; right padded with filler {fmt(d['picks_right_verbose'])}")
        print(f"  plain, right answer in slot B {fmt(d['picks_right_when_right_is_B_plain'])}; in slot A {fmt(d['picks_right_when_right_is_A_plain'])}; picks slot B {fmt(d['picks_B_plain'])}")
        print("  plain accuracy by slice: " + "; ".join(f"{sl} {d['picks_right_plain_'+sl]['pct']}% (n={d['picks_right_plain_'+sl]['n']})" for sl in slices))
    print("saved ->", args.out)


if __name__ == "__main__":
    main()
