"""Score evaluation runs on the held-out splits.

usage: python -m attrib.compare --splits data/splits results/<run_a> results/<run_b> ...

For id_test and ood_test separately:
  Who    agent accuracy, averaged over frameworks; only multi-agent systems count (the harness treats smolagents,
         alfagent and mathchat as single-agent, so in-domain Who covers debate + macnet only)
  When   step accuracy, averaged over frameworks
  What   error-mode macro-F1 over the gold modes present
  Joint  agent, step and mode all correct on the same trace, averaged over frameworks
  mode   pooled error-mode accuracy
  predR  share of predictions that are reasoning modes (R.*)
  omRec / omPre  exact-mode recall / precision on omission and process modes (A.3, V.1, V.2, C.1, C.2, C.3, PL.1, PL.2)
"""
import argparse
import collections
import glob
import json
from pathlib import Path

SINGLE = {"agentoccam", "alfagent", "eva", "gemini", "mathchat", "openai_cua", "pixelcraft", "smolagents"}
OMISSION = {"A.3", "V.1", "V.2", "C.1", "C.2", "C.3", "PL.1", "PL.2"}


def load(d):
    return {r["trace_id"]: r for f in glob.glob(f"{d}/**/*.jsonl", recursive=True) for line in open(f) for r in [json.loads(line)]}


def ok(r, k):
    return bool((r.get("score") or {}).get(k))


def summarize(results, keep):
    rs = [results[i] for i in keep if i in results]
    by = collections.defaultdict(list)
    for r in rs:
        by[r["framework"]].append(r)
    mean = lambda x: sum(x) / len(x) if x else float("nan")
    who = [100 * sum(ok(r, "agent") for r in v) / len(v) for f, v in by.items() if f not in SINGLE]
    when = [100 * sum(ok(r, "step") for r in v) / len(v) for v in by.values()]
    joint = [100 * sum(all(ok(r, k) for k in ("agent", "step", "mode")) for r in v) / len(v) for v in by.values()]
    gold = [r["ground_truth"].get("mode") for r in rs]
    pred = [((r.get("prediction") or {}).get("error_mode") or "") for r in rs]
    f1 = []
    for c in sorted(set(gold)):
        tp = sum(a == c and b == c for a, b in zip(gold, pred))
        fp = sum(a != c and b == c for a, b in zip(gold, pred))
        fn = sum(a == c and b != c for a, b in zip(gold, pred))
        f1.append(0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    n = max(1, len(rs))
    return dict(n=len(rs), who=mean(who), when=mean(when), what=100 * mean(f1), joint=mean(joint), mode=100 * sum(a == b for a, b in zip(gold, pred)) / n,
                predR=100 * sum(x.startswith("R.") for x in pred) / n,
                omRec=100 * sum(a == b for a, b in zip(gold, pred) if a in OMISSION) / max(1, sum(a in OMISSION for a in gold)),
                omPre=100 * sum(a == b for a, b in zip(gold, pred) if b in OMISSION) / max(1, sum(b in OMISSION for b in pred)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    for split in ("id_test", "ood_test"):
        keep = (Path(a.splits) / f"{split}_ids.txt").read_text().split()
        print(f"== {split}")
        print(f"{'run':44s} {'n':>5s} {'Who':>6s} {'When':>6s} {'What':>6s} {'Joint':>6s} {'mode':>6s} {'predR':>6s} {'omRec':>6s} {'omPre':>6s}")
        for d in a.runs:
            x = summarize(load(d), keep)
            print(f"{Path(d).name[:44]:44s} {x['n']:5d} " + " ".join(f"{x[k]:6.1f}" for k in ("who", "when", "what", "joint", "mode", "predR", "omRec", "omPre")))


if __name__ == "__main__":
    main()
