"""SFT targets for the two supervised baselines.

usage:
  python -m attrib.sft_targets gold --rows data/built/train.json --data-root whowhen-pro --out data/built/sft_gold.json
  python -m attrib.sft_targets rs   --rows data/built/train.json --teacher-results results/<model>_gold_step_mode_train --out data/built/sft_rs.json

gold: the gold agent/step/mode in the official answer format, with a one-sentence reason taken from the taxonomy
      definition (the release has no written reasons). Multi-agent gold (debate lists two agents) uses the first
      listed agent, which the scorer accepts. Every target scores fully correct under the official scorer.
rs:   rejection sampling from the privileged teacher: the same model answering WITH the gold_step_mode note
      (run attrib.eval_vllm --hint gold_step_mode on the train ids, greedy), keeping only answers whose agent,
      step and mode are all correct.
"""
import argparse
import glob
import json
import re

import yaml

from attrib.prepare import gold_coord


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["gold", "rs"])
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", help="gold: the whowhen-pro folder (for taxonomy.yaml)")
    ap.add_argument("--teacher-results", help="rs: results folder of the teacher-with-note run on the train ids")
    a = ap.parse_args()
    rows = json.load(open(a.rows))
    if a.kind == "gold":
        tax = yaml.safe_load(open(f"{a.data_root}/taxonomy.yaml"))
        out = []
        for r in rows:
            step, agent = gold_coord(r["framework"], r["gt"])
            agent = agent.split(", ")[0]
            mode = r["gt"]["mode"]
            desc = " ".join(tax[mode]["description"].split())
            first = re.split(r"(?<=[.!?])\s", desc)[0]
            out.append(dict(r, target=f"Agent Name: {agent}\nStep Number: {step}\nError Mode: {mode}\nReason: {tax[mode]['name']}: {first}"))
    else:
        good = {}
        for f in glob.glob(f"{a.teacher_results}/**/*.jsonl", recursive=True):
            for line in open(f):
                x = json.loads(line)
                s = x.get("score") or {}
                if x.get("raw_output") and all(bool(s.get(k)) for k in ("agent", "step", "mode")):
                    good[x["trace_id"]] = x["raw_output"].strip()
        out = [dict(r, target=good[r["id"]]) for r in rows if r["id"] in good]
    json.dump(out, open(a.out, "w"))
    print(a.kind, "targets", len(out), "of", len(rows))


if __name__ == "__main__":
    main()
