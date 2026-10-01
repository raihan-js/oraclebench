#!/usr/bin/env python3
"""Build the oracle item bank: (prompt, answer, oracle_verdict, provenance).

Two slices, ALWAYS kept separate:
  natural:   wrong answers harvested from flipgate's stored runs
             (real model errors with oracle scores != 1.0)
  corrupted: correct answers perturbed deterministically
             (fabricated FAR clause swap, off-by-one arithmetic step,
              one broken IFEval rule)

Target: ~2,000 items across 3 oracles. Writes data/items.jsonl.
Usage: PYTHONPATH=src python scripts/build_items.py [--n-natural N] [--seed 42]
"""
import argparse
import json
import random
import re

from oraclebench.oracles.fedproc import FedProcRegistryScorer
from oraclebench.oracles.gsm8k import GSM8KScorer
from oraclebench.oracles.ifeval import IFEvalScorer

FLIPGATE_RESULTS = "../flipgate/data/results"
REGISTRY_PATH = "../flipgate/data/eval/far_registry.json"


def _find_run(dataset: str, model_sub: str, min_items: int):
    """Find a flipgate run dir by dataset + model substring + size."""
    from pathlib import Path
    base = Path(FLIPGATE_RESULTS)
    for run_dir in sorted(base.iterdir()):
        if not run_dir.is_dir():
            continue
        meta_path = run_dir / "metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        if (meta.get("dataset") == dataset
                and model_sub in run_dir.name
                and meta.get("num_items", 0) >= min_items):
            return run_dir
    return None


def _gsm8k_answers() -> dict[str, str]:
    """Map flipgate item_id (item_NNNN over test-split prefix) -> gold answer."""
    from datasets import load_dataset
    gsm8k = load_dataset("openai/gsm8k", "main", split="test")
    return {f"item_{i:04d}": gsm8k[i]["answer"] for i in range(len(gsm8k))}


def natural_gsm8k(n: int, rng: random.Random) -> list[dict]:
    """Harvest naturally-wrong GSM8K answers from flipgate's bf16 1000-item run.

    Re-scores with the CURRENT local scorer (never trust stored scores:
    scorer versions evolve; the bank must agree with this repo's oracles).
    """
    from oraclebench.oracles.gsm8k import GSM8KScorer
    scorer = GSM8KScorer()
    target = _find_run("gsm8k", "bf16", 1000)
    if target is None:
        print("  no 1000-item bf16 run found, skipping natural/gsm8k")
        return []
    answers = _gsm8k_answers()
    wrong = []
    with open(target / "gsm8k.jsonl") as f:
        for line in f:
            it = json.loads(line)
            ref = {"answer": answers.get(it["item_id"], "")}
            if scorer.score(it["prompt"], it["response"], ref) != 1.0:
                wrong.append({
                    "oracle": "gsm8k",
                    "prompt": it["prompt"],
                    "answer": it["response"],
                    "reference": ref,
                    "oracle_verdict": False,
                    "provenance": f"natural:{target.name}:{it['item_id']}",
                })
    rng.shuffle(wrong)
    print(f"  natural/gsm8k: {len(wrong)} wrong answers available, taking {min(n, len(wrong))}")
    return wrong[:n]


def corrupted_gsm8k(n: int, rng: random.Random) -> list[dict]:
    """Off-by-one arithmetic: take correct answers, shift final number by ±1."""
    target = _find_run("gsm8k", "bf16", 1000)
    if target is None:
        print("  no 1000-item bf16 run found, skipping corrupted/gsm8k")
        return []
    answers = _gsm8k_answers()
    out = []
    with open(target / "gsm8k.jsonl") as f:
        for line in f:
            it = json.loads(line)
            if it["score"] != 1.0:
                continue
            # FlipGate bf16 responses use LaTeX \boxed{N}; fall back to #### N
            m = re.search(r"\\boxed\{(\d+\.?\d*)\}", it["response"])
            if m:
                val = float(m.group(1))
                delta = rng.choice([-1, 1])
                corrupted = (it["response"][:m.start(1)] + str(int(val + delta))
                             + it["response"][m.end(1):])
            else:
                m2 = re.search(r"####\s*(\d+\.?\d*)", it["response"])
                if not m2:
                    continue
                val = float(m2.group(1))
                delta = rng.choice([-1, 1])
                corrupted = (it["response"][:m2.start(1)] + str(int(val + delta))
                             + it["response"][m2.end(1):])
            out.append({
                "oracle": "gsm8k",
                "prompt": it["prompt"],
                "answer": corrupted,
                "reference": {"answer": answers.get(it["item_id"], "")},
                "oracle_verdict": False,
                "provenance": f"corrupted:off-by-one:{target.name}:{it['item_id']}",
            })
            if len(out) >= n:
                break
    print(f"  corrupted/gsm8k: built {len(out)}")
    return out


def natural_fedproc(n: int, rng: random.Random) -> list[dict]:
    """Naturally-hallucinated clauses from flipgate FedProc runs (AWQ/GPTQ)."""
    from pathlib import Path
    base = Path(FLIPGATE_RESULTS)
    out = []
    for run_dir in sorted(base.iterdir()):
        if not run_dir.is_dir():
            continue
        if "AWQ" not in run_dir.name and "GPTQ" not in run_dir.name:
            continue
        meta_path = run_dir / "metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        if meta.get("dataset") != "fedproc":
            continue
        with open(run_dir / "fedproc.jsonl") as f:
            for line in f:
                it = json.loads(line)
                if it["score"] != 1.0:
                    out.append({
                        "oracle": "fedproc",
                        "prompt": it["prompt"],
                        "answer": it["response"],
                        "reference": {},
                        "oracle_verdict": False,
                        "provenance": f"natural:{run_dir.name}:{it['item_id']}",
                    })
    rng.shuffle(out)
    print(f"  natural/fedproc: {len(out)} available, taking {min(n, len(out))}")
    return out[:n]


def corrupted_fedproc(n: int, rng: random.Random, registry: set) -> list[dict]:
    """Swap a real cited clause for a fabricated but plausible one."""
    reg = sorted(registry)
    out = []
    # fabricate by perturbing real IDs: change last digit group
    # stride coprime-ish to spread across registry; loop until n built
    i, guard = 0, 0
    while len(out) < n and guard < len(reg) * 3:
        real = reg[(i * 7919) % len(reg)]
        i += 1
        guard += 1
        m = re.match(r"(\d{2,3}\.\d{3}-)(\d+)([a-z]?)((?:\([a-z0-9]+\))*)$", real)
        if not m:
            continue
        prefix, num, letter, suffix = m.groups()
        fake = f"{prefix}{int(num) + 900}{letter}{suffix}"
        assert fake not in registry, f"collision: {fake}"
        prompt = (f"A federal contract needs a clause covering {real}. "
                  "Which FAR or DFARS clause number applies?")
        out.append({
            "oracle": "fedproc",
            "prompt": prompt,
            "answer": f"According to FAR {fake}, the requirement applies.",
            "reference": {},
            "oracle_verdict": False,
            "provenance": f"corrupted:fabricated-clause:{real}->{fake}",
        })
    print(f"  corrupted/fedproc: built {len(out)}")
    return out


def _ifeval_data() -> list[dict]:
    """Load cached IFEval input file (index-matched to flipgate item_NNNN ids)."""
    import os
    path = os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--google--IFEval"
        "/snapshots/966cd89545d6b6acfd7638bc708b98261ca58e84/ifeval_input_data.jsonl")
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def natural_ifeval(n: int, rng: random.Random) -> list[dict]:
    """Naturally-failing IFEval responses from flipgate bf16 run."""
    target = _find_run("ifeval", "bf16", 500)
    if target is None:
        print("  no 500+-item bf16 IFEval run found, skipping natural/ifeval")
        return []
    data = _ifeval_data()
    wrong = []
    with open(target / "ifeval.jsonl") as f:
        for line in f:
            it = json.loads(line)
            if it["score"] != 1.0:
                idx = int(it["item_id"].split("_")[1])
                ref = data[idx]
                wrong.append({
                    "oracle": "ifeval",
                    "prompt": it["prompt"],
                    "answer": it["response"],
                    "reference": {"instruction_id_list": ref["instruction_id_list"],
                                  "kwargs": ref["kwargs"]},
                    "oracle_verdict": False,
                    "provenance": f"natural:{target.name}:{it['item_id']}",
                })
    rng.shuffle(wrong)
    print(f"  natural/ifeval: {len(wrong)} available, taking {min(n, len(wrong))}")
    return wrong[:n]


def corrupted_ifeval(n: int, rng: random.Random) -> list[dict]:
    """Break one rule in a correct response: insert comma / delete keyword / truncate."""
    target = _find_run("ifeval", "bf16", 500)
    if target is None:
        print("  no 500+-item bf16 IFEval run found, skipping corrupted/ifeval")
        return []
    data = _ifeval_data()
    out = []
    with open(target / "ifeval.jsonl") as f:
        for line in f:
            it = json.loads(line)
            if it["score"] != 1.0:
                continue
            idx = int(it["item_id"].split("_")[1])
            ref = data[idx]
            resp = it["response"]
            broken = None
            for iid, kw in zip(ref["instruction_id_list"], ref["kwargs"]):
                if iid == "punctuation:no_comma" and "," not in resp:
                    broken = resp + ", oops"
                    break
                if iid == "keywords:existence":
                    for k in kw.get("keywords", []):
                        if k.lower() in resp.lower():
                            broken = re.sub(re.escape(k), "", resp, count=1,
                                            flags=re.IGNORECASE)
                            break
                    if broken:
                        break
            if broken is None:
                words = resp.split()
                if len(words) > 6:
                    broken = " ".join(words[:3])
            if broken is None:
                continue
            out.append({
                "oracle": "ifeval",
                "prompt": it["prompt"],
                "answer": broken,
                "reference": {"instruction_id_list": ref["instruction_id_list"],
                              "kwargs": ref["kwargs"]},
                "oracle_verdict": False,
                "provenance": f"corrupted:broken-rule:{target.name}:{it['item_id']}",
            })
            if len(out) >= n:
                break
    print(f"  corrupted/ifeval: built {len(out)}")
    return out
    """Naturally-hallucinated clauses from flipgate FedProc runs (AWQ/GPTQ)."""
    from pathlib import Path
    base = Path(FLIPGATE_RESULTS)
    out = []
    for run_dir in base.iterdir():
        if not run_dir.is_dir() or ("AWQ" not in run_dir.name and "GPTQ" not in run_dir.name):
            continue
        meta_path = run_dir / "metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        if meta.get("dataset") != "fedproc":
            continue
        with open(run_dir / "fedproc.jsonl") as f:
            for line in f:
                it = json.loads(line)
                if it["score"] != 1.0:
                    out.append({
                        "oracle": "fedproc",
                        "prompt": it["prompt"],
                        "answer": it["response"],
                        "reference": {},
                        "oracle_verdict": False,
                        "provenance": f"natural:{run_dir.name}:{it['item_id']}",
                    })
    rng.shuffle(out)
    print(f"  natural/fedproc: {len(out)} available, taking {min(n, len(out))}")
    return out[:n]


def correct_slice(rng: random.Random, per_oracle: int = 150) -> list[dict]:
    """Correct answers (oracle_verdict True) from flipgate runs, re-scored."""
    from oraclebench.oracles.fedproc import FedProcRegistryScorer
    from oraclebench.oracles.gsm8k import GSM8KScorer
    from oraclebench.oracles.ifeval import IFEvalScorer
    from pathlib import Path

    with open(REGISTRY_PATH) as f:
        import json as _json
        registry = {_canonical(c) for c in _json.load(f)}
    scorers = {"gsm8k": GSM8KScorer(), "ifeval": IFEvalScorer(),
               "fedproc": FedProcRegistryScorer(registry=registry)}
    answers = _gsm8k_answers()
    data = _ifeval_data()
    out: list[dict] = []
    counts = {"gsm8k": 0, "ifeval": 0, "fedproc": 0}
    base = Path(FLIPGATE_RESULTS)
    for run_dir in sorted(base.iterdir()):
        if not run_dir.is_dir() or "bf16" not in run_dir.name:
            continue
        meta_path = run_dir / "metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        ds = meta.get("dataset")
        if ds not in ("gsm8k", "ifeval", "fedproc") or counts[ds] >= per_oracle:
            continue
        path = run_dir / f"{ds}.jsonl"
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                if counts[ds] >= per_oracle:
                    break
                it = json.loads(line)
                if ds == "gsm8k":
                    ref = {"answer": answers.get(it["item_id"], "")}
                elif ds == "ifeval":
                    idx = int(it["item_id"].split("_")[1])
                    ref = {"instruction_id_list": data[idx]["instruction_id_list"],
                           "kwargs": data[idx]["kwargs"]}
                else:
                    ref = {}
                if scorers[ds].score(it["prompt"], it["response"], ref) == 1.0:
                    out.append({"oracle": ds, "prompt": it["prompt"],
                                "answer": it["response"], "reference": ref,
                                "oracle_verdict": True,
                                "provenance": f"correct:{run_dir.name}:{it['item_id']}"})
                    counts[ds] += 1
    print(f"  correct slice: {counts}")
    return out


def _canonical(c: str) -> str:
    import re as _re
    return _re.sub(r"^(?:FAR|DFARS)\s+", "", c.strip(), flags=_re.IGNORECASE)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-natural", type=int, default=600)
    ap.add_argument("--n-corrupted", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/items.jsonl")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    with open(REGISTRY_PATH) as f:
        registry = set(json.load(f))

    items: list[dict] = []
    # natural slice (3 oracles)
    items += natural_gsm8k(args.n_natural // 3, rng)
    items += natural_fedproc(args.n_natural // 3, rng)
    items += natural_ifeval(args.n_natural // 3, rng)
    # corrupted slice (3 oracles)
    items += corrupted_gsm8k(args.n_corrupted // 3, rng)
    with open(REGISTRY_PATH) as f:
        import json as _json
        registry = set(_json.load(f))
    # canonicalize (strip FAR/DFARS prefixes like the scorer does)
    registry = {re.sub(r"^(?:FAR|DFARS)\s+", "", c, flags=re.IGNORECASE) for c in registry}
    items += corrupted_fedproc(args.n_corrupted // 3, rng, registry)
    items += corrupted_ifeval(args.n_corrupted // 3, rng)

    # Correct-verdict slice (judge accuracy needs both classes, not just
    # false-accepts): correct answers re-scored with current oracles.
    items += correct_slice(rng, per_oracle=150)

    # Final gate: re-score EVERYTHING with this repo's current oracles.
    # Stored scores and older scorer versions are never trusted; the bank
    # must agree with the oracles that will grade judges against it.
    # (This caught 84 stale-score mismatches on the first build.)
    from oraclebench.oracles.fedproc import FedProcRegistryScorer
    from oraclebench.oracles.gsm8k import GSM8KScorer
    from oraclebench.oracles.ifeval import IFEvalScorer
    scorers = {"gsm8k": GSM8KScorer(), "ifeval": IFEvalScorer(),
               "fedproc": FedProcRegistryScorer(registry=set(registry))}
    kept, dropped = [], 0
    for it in items:
        s = scorers[it["oracle"]].score(it["prompt"], it["answer"], it["reference"])
        if (s == 1.0) == it["oracle_verdict"]:
            kept.append(it)
        else:
            dropped += 1
    print(f"Oracle agreement gate: kept {len(kept)}, dropped {dropped} stale")
    items = kept

    with open(args.out, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    from collections import Counter
    print(f"\nWrote {len(items)} items -> {args.out}")
    print(Counter((it["oracle"], it["provenance"].split(":")[0]) for it in items))


if __name__ == "__main__":
    main()
